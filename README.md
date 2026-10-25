# PreVisit

**Hospitals share records with each other. They can't see the paper reports patients carry.**
PreVisit lets a patient photograph those reports, have NVIDIA Nemotron turn them into a clean, checkable
record, and send their doctor an expiring link *before* the visit. The patient chooses exactly what is shared,
can switch it off at any time, and sees every time it is opened.

Built for the Nebius x NVIDIA Global AI Hackathon, **Best Apps and Agents** track.

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

| Step | Call | Model |
|---|---|---|
| Read a report photo | Token Factory `chat.completions` with an image | Nemotron vision (VL) model |
| Read PDF / pasted text | Token Factory `chat.completions` | Nemotron text model |
| Doctor's pre-visit brief | Token Factory `chat.completions` | Nemotron text model |

All calls go to `https://api.tokenfactory.nebius.com/v1/` through the OpenAI-compatible API. Exact model IDs
are discovered at runtime from the catalog (`GET /v1/models`), or pinned with `TEXT_MODEL` / `VISION_MODEL`.
Temperature is 0, and invalid JSON triggers one automatic retry.
Token Factory let us swap and compare Nemotron sizes without managing GPUs (see feedback in the submission).

## Safety choices

- The model only copies what a document says; it is told never to interpret, diagnose or invent.
- **Human in the loop:** nothing is shareable until the patient confirms it against the original.
- Deterministic emergency check on "today's concern" (chest pain, trouble breathing, stroke signs, etc.)
  shows a "call emergency services" warning. It does not use the model.
- Doctors see a disclaimer: patient-provided and AI-organised, verify against originals.
- Public-demo protections: per-visitor daily AI call cap, 6 MB upload limit, file-type allowlist, automatic
  deletion of demo data after 48 h, share links are random 144-bit tokens.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then paste your Token Factory key into .env
python check_models.py --probe  # lists Nemotron models and which accept images
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and click **Try with a sample patient**. Three synthetic reports can be read with
one click on the Reports tab (regenerate them with `python gen_samples.py`).

No key yet? `MOCK_LLM=1 uvicorn app.main:app` runs the whole UI with rule-based fake extraction.
Run `python smoke_test.py` to check the full flow (profile, upload, confirm, share, privacy, expiry, revoke,
access log, rate cap) without any network.

If your catalog has no Nemotron *vision* model, photos can't be read. Upload a text-based PDF or paste the
report text instead, or set `VISION_MODEL` to a vision-capable Nemotron model.

### Deploy for judges

Judges must be able to use the demo for free until Dec 15, 2026. Any host that runs a Python web process works
(Render, Fly.io, Railway, a small VM). Set `NEBIUS_API_KEY` as a secret environment variable, start command
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and attach a small persistent disk for `data/` (or accept
that demo data resets on restart). Keep `DAILY_LLM_CALLS_PER_IP` low to protect credits.

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
