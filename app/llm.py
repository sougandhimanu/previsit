"""All model calls go through Nebius Token Factory (OpenAI-compatible API).

Models are NVIDIA Nemotron. Exact model IDs change, so they are discovered at
runtime from the Token Factory catalog (GET /v1/models). You can pin them with
TEXT_MODEL and VISION_MODEL in .env. Set MOCK_LLM=1 to run without a key.
"""
import base64
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict

from fastapi import HTTPException
from openai import OpenAI

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
log = logging.getLogger("uvicorn.error")  # shows in the uvicorn terminal

EXTRACT_SYSTEM = """You read medical documents (lab reports, prescriptions, discharge notes) and
copy their contents into structured JSON. You only COPY what is written. You never diagnose, never
interpret, never recommend treatment, and never invent values. If something is not in the document,
use null or an empty list. Return ONLY one JSON object, no prose, no markdown.
Field rules:
- conditions: ONLY diagnoses or medical history written in the document. A test, measurement or
  result is never a condition, even if it is abnormal. If the document names no diagnosis, use [].
- lab_results: every test with a value. "value" is the number or result only; the unit goes in "unit".
- source_quote: the exact words from the document that the item came from.
- key_notes: clinical instructions, findings or follow-up that are not already in another field.
  Never patient name, date of birth, addresses, registration numbers, report dates, or header/footer
  text."""

EXTRACT_SCHEMA = """{
  "report_type": "lab | prescription | discharge | imaging | other",
  "report_date": "YYYY-MM-DD or null",
  "provider": "clinic/hospital/lab name or null",
  "medications": [{"name": "", "dose": "", "frequency": "", "source_quote": ""}],
  "allergies": [{"substance": "", "reaction": "", "source_quote": ""}],
  "conditions": [{"name": "", "source_quote": ""}],
  "lab_results": [{"test": "", "value": "", "unit": "", "reference_range": "", "flag": "high | low | normal | null", "source_quote": ""}],
  "key_notes": ["short factual statements copied from the document"]
}"""

SUMMARY_SYSTEM = """You prepare a one-page pre-visit brief for a clinician from patient-shared records.
Rules: use ONLY the data given. Do not diagnose. Do not recommend treatment, tests, lifestyle changes or
medication changes. Do not add facts that are not in the data. Keep it short and scannable.

Write exactly these four sections in this order. Each section is the heading alone on its own line
(no colon, no capitals, no markdown), then "- " bullet lines, then one blank line:
Reason for visit
Key history
Medications
Allergies

What goes where:
- Key history: diagnoses and past medical history only. Never test results or symptoms.
- Medications: each medicine once, with dose and how often as written.
If a section has no data, write "- Not shared"."""

_client = None
_models_cache: dict = {"at": 0.0, "ids": []}
_lock = threading.Lock()

# --- per-IP daily cap so a public demo cannot burn through credits -------------
_usage: dict = defaultdict(lambda: [0, 0])  # ip -> [day, count]


def mock_enabled() -> bool:
    return os.getenv("MOCK_LLM", "").strip().lower() in ("1", "true", "yes")


def charge(ip: str) -> None:
    limit = int(os.getenv("DAILY_LLM_CALLS_PER_IP", "25"))
    day = int(time.time() // 86400)
    with _lock:
        rec = _usage[ip]
        if rec[0] != day:
            rec[0], rec[1] = day, 0
        if rec[1] >= limit:
            raise HTTPException(
                429,
                f"Demo limit reached ({limit} AI requests per day per visitor). "
                "This keeps the free demo available for everyone.",
            )
        rec[1] += 1


def client() -> OpenAI:
    global _client
    if _client is None:
        key = os.getenv("NEBIUS_API_KEY", "").strip()
        if not key:
            raise HTTPException(
                503, "NEBIUS_API_KEY is not set. Add it to .env (or set MOCK_LLM=1 to try the UI)."
            )
        _client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL", BASE_URL), api_key=key)
    return _client


def list_models(force: bool = False) -> list:
    if not force and time.time() - _models_cache["at"] < 600 and _models_cache["ids"]:
        return _models_cache["ids"]
    ids = [m.id for m in client().models.list().data]
    _models_cache.update(at=time.time(), ids=ids)
    return ids


_BAD = ("embed", "rerank", "guard", "safety", "reward", "parse")


def _nemotron(ids: list) -> list:
    return [i for i in ids if "nemotron" in i.lower() and not any(b in i.lower() for b in _BAD)]


def pick_text_model() -> str:
    pinned = os.getenv("TEXT_MODEL", "").strip()
    if pinned:
        return pinned
    cands = [i for i in _nemotron(list_models()) if not re.search(r"vl|vision|omni", i, re.I)]
    for pref in ("nano", "lightning", "super", "ultra"):  # cheapest/fastest first
        for c in cands:
            if pref in c.lower():
                return c
    if cands:
        return cands[0]
    raise HTTPException(
        503, "No Nemotron text model found in the Token Factory catalog. Set TEXT_MODEL in .env."
    )


def pick_vision_model():
    pinned = os.getenv("VISION_MODEL", "").strip()
    if pinned:
        return pinned
    cands = [i for i in _nemotron(list_models()) if re.search(r"vl|vision|omni", i, re.I)]
    return cands[0] if cands else None


def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    return text


def _parse_json(text: str) -> dict:
    text = _strip(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _chat(model: str, messages: list, max_tokens: int = 1200) -> str:
    try:
        r = client().chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=max_tokens,
            # copying facts needs no reasoning; thinking tokens otherwise eat max_tokens
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except HTTPException:
        raise
    except Exception as e:  # network, auth, bad model id...
        raise HTTPException(502, f"Token Factory request failed ({model}): {e}")
    used = r.usage.completion_tokens if r.usage else "?"
    log.info("%s: %s / %d output tokens (input %s)", model, used, max_tokens, r.usage.prompt_tokens if r.usage else "?")
    if r.choices[0].finish_reason == "length":
        log.warning("%s: answer cut off at %d tokens, raise max_tokens", model, max_tokens)
    return r.choices[0].message.content or ""


# Identity and header/footer lines that models copy into key_notes despite being told not to.
_NOISE = re.compile(r"^\s*(patient( name)?|name|dob|d\.o\.b\.?|date of birth|collected|reported|(report )?date(?=\s*:)|"
                    r"reg(istration)?\.?\s*(no|number)\.?|mrn|id)\b|synthetic data|for demo only", re.I)


def _split_unit(lab: dict) -> dict:
    """Models sometimes repeat the unit inside the value ("8.1 %" + "%"); keep it in one place."""
    value, unit = str(lab.get("value") or "").strip(), str(lab.get("unit") or "").strip()
    if unit and value.lower().endswith(unit.lower()) and value[: -len(unit)].strip():
        lab = {**lab, "value": value[: -len(unit)].strip()}
    return lab


def _normalise(d: dict) -> dict:
    out = {
        "report_type": d.get("report_type") or "other",
        "report_date": d.get("report_date"),
        "provider": d.get("provider"),
        "medications": d.get("medications") or [],
        "allergies": d.get("allergies") or [],
        "conditions": d.get("conditions") or [],
        "lab_results": [_split_unit(lab) for lab in d.get("lab_results") or [] if isinstance(lab, dict)],
        "key_notes": [n for n in d.get("key_notes") or [] if isinstance(n, str) and n.strip() and not _NOISE.search(n)],
    }
    return out


def extract_from_text(text: str, filename: str = "") -> dict:
    if mock_enabled():
        return mock_extract(filename or text)
    model = pick_text_model()
    user = f"Document text:\n\n{text[:12000]}\n\nReturn JSON in exactly this shape:\n{EXTRACT_SCHEMA}"
    out = _run_extract(model, [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": user}])
    return _grounded(out, text)


def _squash(s) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", str(s or "").lower()).strip()


# For text we can check the model's work: drop any item whose quote and name are not in the document,
# so an invented diagnosis or value never reaches the patient's check screen.
_MAIN = {"medications": "name", "allergies": "substance", "conditions": "name", "lab_results": "test"}


def _grounded(ex: dict, text: str) -> dict:
    doc = _squash(text)
    for key, main in _MAIN.items():
        ex[key] = [i for i in ex[key] if (_squash(i.get("source_quote")) and _squash(i.get("source_quote")) in doc)
                   or (_squash(i.get(main)) and _squash(i.get(main)) in doc)]
    return ex


def extract_from_image(data: bytes, mime: str, filename: str = "") -> dict:
    if mock_enabled():
        return mock_extract(filename)
    model = pick_vision_model()
    if not model:
        raise HTTPException(
            422,
            "No Nemotron vision model was found in the catalog, so photos can't be read yet. "
            "Run `python check_models.py` to see options and set VISION_MODEL in .env, "
            "or upload a text/PDF report, or paste the report text instead.",
        )
    b64 = base64.b64encode(data).decode()
    content = [
        {"type": "text", "text": f"Read this document image and return JSON in exactly this shape:\n{EXTRACT_SCHEMA}"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
    return _run_extract(model, [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": content}])


def _run_extract(model: str, messages: list) -> dict:
    raw = _chat(model, messages)
    try:
        return _normalise(_parse_json(raw))
    except (json.JSONDecodeError, AttributeError):
        retry = messages + [
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": "That was not valid JSON. Return ONLY the JSON object."},
        ]
        raw2 = _chat(model, retry)
        try:
            return _normalise(_parse_json(raw2))
        except (json.JSONDecodeError, AttributeError):
            raise HTTPException(502, "The model did not return readable structured data. Try again or paste the text.")


def summarize(shared: dict) -> str:
    if mock_enabled():
        return mock_summary(shared)
    model = pick_text_model()
    user = "Shared patient data (JSON):\n" + json.dumps(shared, ensure_ascii=False)[:14000]
    raw = _chat(model, [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user}], 600)
    return _strip(raw) + "\n\n" + notable(shared) + "\n\n" + to_confirm(shared)


# The last two brief sections are exact filtering and comparison, so they are computed, not generated.
def notable(shared: dict) -> str:
    labs = [" ".join(x for x in [f"{lab.get('test')}:", lab.get("value"), lab.get("unit"), f"({lab['flag']})",
                                 f"[{r['report_date']}]" if r.get("report_date") else ""] if x)
            for r in shared.get("reports", []) for lab in r.get("lab_results", []) if lab.get("flag") in ("high", "low")]
    return "Notable results\n" + "\n".join(f"- {x}" for x in labs or ["None flagged high or low" if shared.get("reports") else "Not shared"])


def to_confirm(shared: dict) -> str:
    """'Things to confirm' is exact comparison, not writing, so it is computed rather than generated:
    the same medicine with different doses across reports, and medicines missing a dose or frequency."""
    items, doses = [], {}
    for r in shared.get("reports", []):
        src = " ".join(x for x in [(r.get("report_type") or "report").capitalize(), r.get("report_date")] if x)
        for m in r.get("medications", []):
            name = (m.get("name") or "").strip()
            if not name:
                continue
            doses.setdefault(name.lower(), (name, {}))[1].setdefault((m.get("dose") or "").strip(), src)
            missing = [f for f in ("dose", "frequency") if not (m.get(f) or "").strip()]
            if missing:
                items.append(f"{name}: no {' or '.join(missing)} given ({src})")
    for name, by_dose in doses.values():
        given = {d: s for d, s in by_dose.items() if d}
        if len(given) > 1:
            items.append(f"{name}: different doses, " + " vs ".join(f"{d} ({s})" for d, s in given.items()))
    return "Things to confirm with the patient\n" + "\n".join(f"- {i}" for i in items or ["Nothing to confirm"])


# --- deterministic fallbacks (also used by MOCK_LLM so the UI can be tried offline) --------

def mock_extract(hint: str) -> dict:
    h = (hint or "").lower()
    if "rx" in h or "prescription" in h:
        return _normalise({
            "report_type": "prescription", "report_date": "2026-09-14", "provider": "Riverside Family Clinic (sample)",
            "medications": [
                {"name": "Metformin", "dose": "500 mg", "frequency": "twice daily", "source_quote": "Metformin 500 mg BID"},
                {"name": "Lisinopril", "dose": "10 mg", "frequency": "once daily", "source_quote": "Lisinopril 10 mg OD"},
            ],
            "allergies": [{"substance": "Penicillin", "reaction": "rash", "source_quote": "Allergy: Penicillin (rash)"}],
        })
    if "discharge" in h:
        return _normalise({
            "report_type": "discharge", "report_date": "2026-07-02", "provider": "St. Mary Hospital (sample)",
            "conditions": [{"name": "Type 2 diabetes", "source_quote": "Dx: Type 2 diabetes mellitus"},
                           {"name": "Hypertension", "source_quote": "Hypertension"}],
            "key_notes": ["Admitted for 2 days for high blood sugar", "Follow up with primary doctor in 2 weeks"],
        })
    return _normalise({
        "report_type": "lab", "report_date": "2026-09-10", "provider": "City Diagnostics Lab (sample)",
        "lab_results": [
            {"test": "HbA1c", "value": "8.1", "unit": "%", "reference_range": "4.0-5.6", "flag": "high", "source_quote": "HbA1c 8.1 % (4.0-5.6) H"},
            {"test": "Fasting glucose", "value": "148", "unit": "mg/dL", "reference_range": "70-99", "flag": "high", "source_quote": "Fasting glucose 148 mg/dL H"},
            {"test": "Creatinine", "value": "0.9", "unit": "mg/dL", "reference_range": "0.6-1.3", "flag": "normal", "source_quote": "Creatinine 0.9 mg/dL"},
        ],
    })


def mock_summary(shared: dict) -> str:
    p = shared.get("profile", {})

    def lines(key):
        v = p.get(key)
        return "\n".join(f"- {x}" for x in v) if v else "Not shared"

    return (
        f"Reason for visit\n{p.get('todays_concern') or 'Not shared'}\n\n"
        f"Key history\n{lines('conditions')}\n\n"
        f"Medications\n{lines('medications')}\n\n"
        f"Allergies\n{lines('allergies')}\n\n"
        + notable(shared) + "\n\n" + to_confirm(shared)
    )
