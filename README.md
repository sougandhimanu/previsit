# PreVisit

**Hospitals share records with each other. They can't see the paper reports patients carry.**
PreVisit lets a patient photograph those reports, have NVIDIA Nemotron turn them into a clean, checkable
record, and send their doctor an expiring link *before* the visit. The patient chooses exactly what is shared,
can switch it off at any time, and sees every time it is opened.

Built for the Nebius x NVIDIA Global AI Hackathon, **Best Apps and Agents** track.

**Live demo:** https://previsit-1oqp.onrender.com (click **Try with a sample patient**; the first visit may take
up to a minute while the server wakes)

> **Demo only.** Use synthetic data. This is not a medical device, gives no diagnosis or treatment advice,
> and has no real authentication or compliance certification (HIPAA etc.).

## What it does

1. **Profile once.** Allergies, medications, conditions, today's concern. Reused at every visit.
2. **Photograph a report.** A Nemotron model on Nebius Token Factory extracts medications, allergies,
   conditions and lab values as structured JSON. Each item keeps a `source_quote` from the document.
3. **Patient verifies.** The original image sits beside the extracted fields. Fix or delete anything wrong
   before saving. Confirmed facts merge into the profile without duplicates.
4. **Share a link.** Pick sections and specific reports, set an expiry (1 hour to 7 days). Private notes are
   off by default. Doctors need no account.
5. **Doctor view.** A one-page brief written by Nemotron *only from the shared data*, plus the structured
   facts and a link to the original document. The brief never diagnoses or recommends treatment.
6. **Access log + revoke.** The patient sees each time the record or an original was opened, and can revoke.

## How Nemotron and Nebius are used

| Step | Model on Token Factory | Why this model |
|---|---|---|
| Read a report photo | **Nemotron 3.5 Super VL** (`nvidia/Nemotron-3.5-Super-VL`) | The Nemotron model that reads images and has a public pay-per-token endpoint. Reads tables of lab values from a photo. |
| Read PDF / pasted text | **Nemotron 3 Nano 30B A3B** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) | Copying facts into a fixed schema needs accuracy, not deep reasoning. Nano is the cheapest Nemotron ($0.06 / $0.24 per 1M tokens), about 5x cheaper than Super. |
| Doctor's pre-visit brief | **Nemotron 3 Nano 30B A3B** | Short, structured summary of already-checked data. Fast and cheap. |

Measured on the three sample reports and one pasted report (thinking off, see below):

| Call | Model | Time | Output tokens |
|---|---|---|---|
| Lab report photo | Super VL | 2.4 s | 542 |
| Prescription photo | Super VL | 1.3 s | 317 |
| Discharge summary photo | Super VL | 1.1 s | 206 |
| Pasted report text | Nano | 2.0 s | 458 |
| Doctor's brief | Nano | 1.0 s | 226 |

How the calls are set up (`app/llm.py`):

- All calls go to `https://api.tokenfactory.nebius.com/v1/` with the standard OpenAI Python SDK, because
  Token Factory is OpenAI-compatible. Switching models is a config change, not a code change.
- **Thinking is turned off** (`chat_template_kwargs: {"enable_thinking": false}`). With reasoning on, both
  models spent the whole token budget on hidden reasoning and returned cut-off JSON on 3 of 4 reports. Off,
  answers were 3-5x shorter and every report parsed. (`reasoning_effort` and a `/no_think` system prompt had no
  effect in our tests.)
- Temperature 0, a strict JSON schema with a `source_quote` per item, and one automatic retry on invalid JSON.
- Every call logs its token use (`used / max_tokens`) and warns when an answer is cut off, so limits are set
  from real numbers.
- Model IDs are pinned with `TEXT_MODEL` / `VISION_MODEL`. If unset, the app picks Nemotron models from the
  live catalog (`GET /v1/models`).

Token Factory let us compare Nemotron sizes and variants behind one key, with no GPUs to manage.

## Safety choices

- The model only copies what a document says; it is told never to interpret, diagnose or invent.
- **Human in the loop:** nothing is shareable until the patient confirms it against the original.
- Deterministic emergency check on "today's concern" (chest pain, trouble breathing, stroke signs, etc.)
  shows a "call emergency services" warning. It does not use the model.
- Doctors see a disclaimer: patient-provided and AI-organised, verify against originals.
- Public-demo protections: per-visitor daily AI call cap, 6 MB upload limit, file-type allowlist, automatic
  deletion of demo data after 48 h, share links are random 144-bit tokens.

## Run it

Needs Python 3.12 and a [Nebius Token Factory](https://tokenfactory.nebius.com) API key.

**1. Install**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Create a file named `.env`** in the project folder (it is git-ignored, never commit it):

```
NEBIUS_API_KEY=your-token-factory-key
TEXT_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
VISION_MODEL=nvidia/Nemotron-3.5-Super-VL
MOCK_LLM=0
DAILY_LLM_CALLS_PER_IP=25
DEMO_RETENTION_HOURS=48
```

| Variable | Meaning |
|---|---|
| `NEBIUS_API_KEY` | Your Token Factory key |
| `TEXT_MODEL`, `VISION_MODEL` | Model IDs to use. Leave empty to pick Nemotron models from the catalog automatically. |
| `MOCK_LLM` | `1` runs the whole app with rule-based fake extraction and no AI calls (no key needed) |
| `DAILY_LLM_CALLS_PER_IP` | AI calls each visitor may make per day, to protect credits |
| `DEMO_RETENTION_HOURS` | Demo data older than this is deleted |
| `DATA_DIR` | Optional. Where the database and uploads are stored (default `data/`) |

**3. Check your models and run**

```bash
python check_models.py --probe   # lists Nemotron models your key can use and which accept images
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and click **Try with a sample patient**. Three synthetic reports can be read with
one click on the Reports tab (regenerate them with `python gen_samples.py`).

No key yet? Set `MOCK_LLM=1` in `.env` to run the whole UI with rule-based fake extraction.
Run `python smoke_test.py` to check the full flow (profile, upload, confirm, share, privacy, expiry, revoke,
access log, rate cap) without any network.

If your catalog has no Nemotron *vision* model, photos can't be read. Upload a text-based PDF or paste the
report text instead, or set `VISION_MODEL` to a vision-capable Nemotron model.

### Deploy for judges

The live demo runs on [Render](https://render.com) as a Python web service. Any host that runs a Python web
process works (Railway, Fly.io, a small VM).

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/healthz` |
| Environment variables | The `.env` values above as secrets, plus `PYTHON_VERSION=3.12.7` |
| Persistent data (optional) | Attach a small disk at `/var/data` and set `DATA_DIR=/var/data`. Without one, demo data resets on restart. |

## Project layout

```
app/main.py        API: profile, reports, sharing, access log, doctor view
app/llm.py         Token Factory client, model discovery, extraction + summary prompts, rate cap
app/db.py          SQLite
app/static/        patient app (index.html), doctor view (doctor.html)
app/samples/       synthetic sample reports
check_models.py    find usable Nemotron models
smoke_test.py      end-to-end test (mock mode)
```

## Limits and next steps

- No Epic/FHIR integration; real use needs hospital-system integration and compliance review.
- Demo identity is a random token in the browser (no accounts, no recovery).
- Possible next: hospital-side queue/token notifications, anonymised trend view for hospitals, FHIR export.

## License

MIT. See `LICENSE`.
