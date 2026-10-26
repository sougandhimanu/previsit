"""PreVisit API.

Patients keep their own record, turn paper/PDF reports into structured facts with NVIDIA Nemotron on
Nebius Token Factory, and share chosen sections with a doctor through an expiring, revocable link.
Every doctor view is written to an access log the patient can see.

DEMO ONLY: use synthetic data. There is no real authentication or clinical safety certification here.
"""
import hashlib
import io
import json
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, llm

APP_DIR = Path(__file__).parent
STATIC = APP_DIR / "static"
SAMPLES = APP_DIR / "samples"
MAX_UPLOAD = 6 * 1024 * 1024
ALLOWED = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    "application/pdf": ".pdf", "text/plain": ".txt",
}
SHAREABLE = ["allergies", "medications", "conditions", "todays_concern", "notes", "reports"]
RETENTION_HOURS = float(os.getenv("DEMO_RETENTION_HOURS", "48"))

EMERGENCY = re.compile(
    r"chest (pain|pressure|tight)|trouble breathing|can'?t breathe|short(ness)? of breath|"
    r"face droop|slurred speech|sudden (weakness|numbness|confusion)|severe bleeding|"
    r"unconscious|seizure|overdose|suicid|want to die|worst headache",
    re.I,
)

app = FastAPI(title="PreVisit", version="0.1.0")
db.init()
db.purge_old(RETENTION_HOURS)


# ---------------------------------------------------------------- helpers
def _ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")) or "?"


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _device(request: Request) -> str:
    return _hash(_ip(request) + "|" + request.headers.get("user-agent", ""))[:6]


def patient_or_401(token: Optional[str]):
    if not token:
        raise HTTPException(401, "Missing patient token")
    row = db.one("SELECT * FROM patients WHERE token_hash = ?", (_hash(token),))
    if not row:
        raise HTTPException(401, "Unknown patient token (the demo may have been reset)")
    return row


def profile_of(row) -> dict:
    p = json.loads(row["profile"] or "{}")
    return {
        "name": row["name"], "dob": row["dob"],
        "allergies": p.get("allergies", []), "medications": p.get("medications", []),
        "conditions": p.get("conditions", []), "todays_concern": p.get("todays_concern", ""),
        "notes": p.get("notes", ""),
    }


def report_json(r, include_extracted=True) -> dict:
    d = {"id": r["id"], "filename": r["filename"], "mime": r["mime"], "status": r["status"],
         "created_at": r["created_at"], "has_file": bool(r["path"])}
    if include_extracted:
        d["extracted"] = json.loads(r["extracted"] or "{}")
    return d


def _dedupe_add(existing: List[str], new: List[str]) -> List[str]:
    seen = {e.strip().lower() for e in existing}
    out = list(existing)
    for n in new:
        n = n.strip()
        if n and n.lower() not in seen:
            out.append(n)
            seen.add(n.lower())
    return out


def _med_line(m: dict) -> str:
    return " ".join(x for x in [m.get("name"), m.get("dose"), m.get("frequency")] if x).strip()


def _merge_meds(existing: List[str], meds: List[dict]) -> List[str]:
    """Same medicine and dose counts as one entry; keep the more detailed wording."""
    out = list(existing)
    for m in meds:
        line = _med_line(m)
        key = " ".join(x for x in [m.get("name"), m.get("dose")] if x).strip().lower()
        if not line:
            continue
        hit = next((i for i, e in enumerate(out) if key and e.strip().lower().startswith(key)), None)
        if hit is None:
            out = _dedupe_add(out, [line])
        elif len(line) > len(out[hit]):
            out[hit] = line
    return out


# ---------------------------------------------------------------- models
class ProfileIn(BaseModel):
    name: str = Field("", max_length=120)
    dob: str = Field("", max_length=20)
    allergies: List[str] = []
    medications: List[str] = []
    conditions: List[str] = []
    todays_concern: str = Field("", max_length=1500)
    notes: str = Field("", max_length=3000)


class ConfirmIn(BaseModel):
    extracted: dict
    add_to_profile: bool = True


class TextIn(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000)
    label: str = "Pasted text"


class SampleIn(BaseModel):
    name: str


class ShareIn(BaseModel):
    doctor_name: str = Field(..., min_length=1, max_length=120)
    sections: List[str]
    report_ids: List[str] = []
    expires_minutes: int = Field(240, ge=5, le=10080)


# ---------------------------------------------------------------- config / patients
@app.get("/api/config")
def config():
    mock = llm.mock_enabled()
    info = {"mock": mock, "retention_hours": RETENTION_HOURS, "text_model": None, "vision_model": None}
    if not mock and os.getenv("NEBIUS_API_KEY"):
        try:
            info["text_model"] = llm.pick_text_model()
            info["vision_model"] = llm.pick_vision_model()
        except HTTPException as e:
            info["error"] = e.detail
    return info


@app.post("/api/patients")
def create_patient(sample: bool = False):
    pid, token = uuid.uuid4().hex[:12], secrets.token_urlsafe(24)
    profile = {}
    name, dob = "", ""
    if sample:
        name, dob = "Asha Verma (sample)", "1958-04-12"
        profile = {
            "allergies": [], "medications": [], "conditions": [],
            "todays_concern": "Feeling more tired than usual and very thirsty for the last two weeks.",
            "notes": "",
        }
    db.run("INSERT INTO patients (id, token_hash, name, dob, profile, created_at) VALUES (?,?,?,?,?,?)",
           (pid, _hash(token), name, dob, json.dumps(profile), time.time()))
    return {"patient_id": pid, "token": token}


@app.get("/api/me")
def me(x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    reports = db.many("SELECT * FROM reports WHERE patient_id = ? ORDER BY created_at DESC", (row["id"],))
    return {"profile": profile_of(row), "reports": [report_json(r) for r in reports]}


@app.put("/api/me/profile")
def save_profile(body: ProfileIn, x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    clean = lambda xs: [x.strip() for x in xs if x.strip()][:60]
    profile = {
        "allergies": clean(body.allergies), "medications": clean(body.medications),
        "conditions": clean(body.conditions), "todays_concern": body.todays_concern.strip(),
        "notes": body.notes.strip(),
    }
    db.run("UPDATE patients SET name=?, dob=?, profile=? WHERE id=?",
           (body.name.strip(), body.dob.strip(), json.dumps(profile), row["id"]))
    warning = None
    if EMERGENCY.search(body.todays_concern):
        warning = ("What you described can be an emergency. Do not wait in a queue or for a message: "
                   "call your local emergency number or go to the nearest emergency department now.")
    return {"ok": True, "emergency_warning": warning}


# ---------------------------------------------------------------- reports
def _save_report(row, filename: str, mime: str, data: Optional[bytes], extracted: dict) -> dict:
    rid, path = uuid.uuid4().hex[:12], None
    if data is not None:
        path = str(db.UPLOAD_DIR / f"{rid}{ALLOWED.get(mime, '.bin')}")
        Path(path).write_bytes(data)
    db.run("INSERT INTO reports (id, patient_id, filename, mime, path, extracted, status, created_at) "
           "VALUES (?,?,?,?,?,?,?,?)",
           (rid, row["id"], filename[:160], mime, path, json.dumps(extracted), "draft", time.time()))
    return report_json(db.one("SELECT * FROM reports WHERE id = ?", (rid,)))


def _extract(data: bytes, mime: str, filename: str, ip: str) -> dict:
    llm.charge(ip)
    if mime in ("image/png", "image/jpeg", "image/webp"):
        return llm.extract_from_image(data, mime, filename)
    if mime == "application/pdf":
        from pypdf import PdfReader
        try:
            text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            raise HTTPException(422, "Could not read this PDF.")
        if len(text.strip()) < 20:
            if llm.mock_enabled():
                return llm.mock_extract(filename)
            raise HTTPException(422, "This PDF looks scanned (no text). Upload a photo/PNG of the page instead.")
        return llm.extract_from_text(text, filename)
    text = data.decode("utf-8", errors="replace")
    return llm.extract_from_text(text, filename)


@app.post("/api/me/reports")
async def upload_report(request: Request, file: UploadFile = File(...), x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    mime = (file.content_type or "").lower()
    if mime not in ALLOWED:
        raise HTTPException(415, "Upload a PNG, JPG, WEBP, PDF or TXT file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File too large (6 MB max).")
    extracted = _extract(data, mime, file.filename or "report", _ip(request))
    return _save_report(row, file.filename or "report", mime, data, extracted)


@app.post("/api/me/reports/text")
def paste_report(body: TextIn, request: Request, x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    llm.charge(_ip(request))
    extracted = llm.extract_from_text(body.text, body.label)
    return _save_report(row, body.label, "text/plain", body.text.encode(), extracted)


@app.get("/api/samples")
def list_samples():
    return [p.name for p in sorted(SAMPLES.glob("*.png"))]


@app.post("/api/me/reports/sample")
def use_sample(body: SampleIn, request: Request, x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    path = (SAMPLES / Path(body.name).name)
    if not path.exists() or path.suffix != ".png":
        raise HTTPException(404, "Unknown sample")
    data = path.read_bytes()
    extracted = _extract(data, "image/png", path.name, _ip(request))
    return _save_report(row, path.name, "image/png", data, extracted)


@app.put("/api/me/reports/{rid}/confirm")
def confirm_report(rid: str, body: ConfirmIn, x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    rep = db.one("SELECT * FROM reports WHERE id = ? AND patient_id = ?", (rid, row["id"]))
    if not rep:
        raise HTTPException(404, "Report not found")
    ex = llm._normalise(body.extracted)
    db.run("UPDATE reports SET extracted=?, status='confirmed' WHERE id=?", (json.dumps(ex), rid))
    if body.add_to_profile:
        p = json.loads(row["profile"] or "{}")
        p["medications"] = _merge_meds(p.get("medications", []), ex["medications"])
        p["allergies"] = _dedupe_add(
            p.get("allergies", []),
            [f"{a.get('substance','')}" + (f" ({a['reaction']})" if a.get("reaction") else "") for a in ex["allergies"]],
        )
        p["conditions"] = _dedupe_add(p.get("conditions", []), [c.get("name", "") for c in ex["conditions"]])
        db.run("UPDATE patients SET profile=? WHERE id=?", (json.dumps(p), row["id"]))
    return {"ok": True}


@app.delete("/api/me/reports/{rid}")
def delete_report(rid: str, x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    rep = db.one("SELECT * FROM reports WHERE id = ? AND patient_id = ?", (rid, row["id"]))
    if not rep:
        raise HTTPException(404, "Report not found")
    if rep["path"]:
        Path(rep["path"]).unlink(missing_ok=True)
    db.run("DELETE FROM reports WHERE id = ?", (rid,))
    return {"ok": True}


@app.get("/api/me/reports/{rid}/file")
def my_report_file(rid: str, t: Optional[str] = Query(None), x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token or t)
    rep = db.one("SELECT * FROM reports WHERE id = ? AND patient_id = ?", (rid, row["id"]))
    if not rep or not rep["path"]:
        raise HTTPException(404, "No file")
    return FileResponse(rep["path"], media_type=rep["mime"])


# ---------------------------------------------------------------- sharing
@app.post("/api/me/shares")
def create_share(body: ShareIn, request: Request, x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    sections = [s for s in body.sections if s in SHAREABLE]
    if not sections:
        raise HTTPException(422, "Choose at least one thing to share.")
    report_ids: List[str] = []
    if "reports" in sections:
        for rid in body.report_ids:
            r = db.one("SELECT id FROM reports WHERE id=? AND patient_id=? AND status='confirmed'", (rid, row["id"]))
            if r:
                report_ids.append(rid)
        if not report_ids:
            raise HTTPException(422, "Pick at least one confirmed report to share, or untick Reports.")
    sid, token = uuid.uuid4().hex[:12], secrets.token_urlsafe(18)
    db.run("INSERT INTO shares (id, patient_id, token, doctor_name, sections, report_ids, expires_at, created_at) "
           "VALUES (?,?,?,?,?,?,?,?)",
           (sid, row["id"], token, body.doctor_name.strip(), json.dumps(sections), json.dumps(report_ids),
            time.time() + body.expires_minutes * 60, time.time()))
    return {"id": sid, "token": token}


def share_json(s) -> dict:
    now = time.time()
    state = "revoked" if s["revoked"] else ("expired" if s["expires_at"] < now else "active")
    return {"id": s["id"], "token": s["token"], "doctor_name": s["doctor_name"],
            "sections": json.loads(s["sections"]), "report_ids": json.loads(s["report_ids"]),
            "expires_at": s["expires_at"], "state": state, "created_at": s["created_at"]}


@app.get("/api/me/shares")
def my_shares(x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    rows = db.many("SELECT * FROM shares WHERE patient_id = ? ORDER BY created_at DESC", (row["id"],))
    return [share_json(s) for s in rows]


@app.post("/api/me/shares/{sid}/revoke")
def revoke_share(sid: str, x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    s = db.one("SELECT id FROM shares WHERE id=? AND patient_id=?", (sid, row["id"]))
    if not s:
        raise HTTPException(404, "Share not found")
    db.run("UPDATE shares SET revoked=1 WHERE id=?", (sid,))
    return {"ok": True}


@app.get("/api/me/access-log")
def access_log(x_patient_token: Optional[str] = Header(None)):
    row = patient_or_401(x_patient_token)
    rows = db.many(
        "SELECT a.at, a.action, a.device, s.doctor_name FROM access_log a JOIN shares s ON s.id = a.share_id "
        "WHERE a.patient_id = ? ORDER BY a.at DESC LIMIT 200", (row["id"],))
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- doctor view
def _active_share(token: str):
    s = db.one("SELECT * FROM shares WHERE token = ?", (token,))
    if not s:
        raise HTTPException(404, "This link is not valid.")
    if s["revoked"]:
        raise HTTPException(410, "The patient has revoked access to this record.")
    if s["expires_at"] < time.time():
        raise HTTPException(410, "This link has expired. Ask the patient to share again.")
    return s


def _log(s, action: str, request: Request):
    db.run("INSERT INTO access_log (share_id, patient_id, at, action, device) VALUES (?,?,?,?,?)",
           (s["id"], s["patient_id"], time.time(), action, _device(request)))


@app.get("/api/shared/{token}")
def shared_view(token: str, request: Request):
    s = _active_share(token)
    p = db.one("SELECT * FROM patients WHERE id = ?", (s["patient_id"],))
    sections = json.loads(s["sections"])
    full = profile_of(p)
    shown = {k: full[k] for k in ("allergies", "medications", "conditions", "todays_concern", "notes") if k in sections}
    reports = []
    if "reports" in sections:
        for rid in json.loads(s["report_ids"]):
            r = db.one("SELECT * FROM reports WHERE id=? AND patient_id=? AND status='confirmed'", (rid, p["id"]))
            if r:
                d = report_json(r)
                d.update(d.pop("extracted"))
                reports.append(d)
    payload_for_ai = {"profile": shown, "reports": [
        {k: v for k, v in r.items() if k not in ("id", "filename", "mime", "status", "created_at", "has_file")}
        for r in reports]}
    summary, summary_note = s["summary"], None
    if not summary:
        try:
            llm.charge(_ip(request))
            summary = llm.summarize(payload_for_ai)
            db.run("UPDATE shares SET summary=? WHERE id=?", (summary, s["id"]))
        except HTTPException as e:
            summary, summary_note = llm.mock_summary(payload_for_ai), f"AI summary unavailable ({e.detail}). Showing a rule-based brief."
    _log(s, "Opened the shared record", request)
    return {
        "patient": {"name": p["name"], "dob": p["dob"]}, "doctor_name": s["doctor_name"],
        "expires_at": s["expires_at"], "sections": sections, "profile": shown, "reports": reports,
        "summary": summary, "summary_note": summary_note,
        "disclaimer": "Patient-provided and AI-organised. Verify against the original documents.",
    }


@app.get("/api/shared/{token}/report/{rid}/file")
def shared_report_file(token: str, rid: str, request: Request):
    s = _active_share(token)
    if "reports" not in json.loads(s["sections"]) or rid not in json.loads(s["report_ids"]):
        raise HTTPException(404, "Not shared")
    rep = db.one("SELECT * FROM reports WHERE id=? AND patient_id=?", (rid, s["patient_id"]))
    if not rep or not rep["path"]:
        raise HTTPException(404, "No file")
    _log(s, f"Viewed original: {rep['filename']}", request)
    return FileResponse(rep["path"], media_type=rep["mime"])


# ---------------------------------------------------------------- pages
@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/doctor")
def doctor_page():
    return FileResponse(STATIC / "doctor.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
