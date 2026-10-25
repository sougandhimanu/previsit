"""End-to-end test of the whole flow with MOCK_LLM=1 (no network, no key)."""
import os
import shutil
import tempfile

os.environ["MOCK_LLM"] = "1"
os.environ["DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

c = TestClient(app)


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


r = c.post("/api/patients?sample=true"); ok(r.status_code == 200, "create sample patient")
H = {"X-Patient-Token": r.json()["token"]}

ok(c.get("/api/me").status_code == 401, "no token is rejected")
ok(c.get("/api/me", headers=H).json()["profile"]["name"].startswith("Asha"), "profile loads")

# upload each sample, confirm, merge into profile
samples = c.get("/api/samples").json(); ok(len(samples) == 3, f"3 samples listed {samples}")
for s in samples:
    d = c.post("/api/me/reports/sample", headers=H, json={"name": s}); ok(d.status_code == 200 and d.json()["status"] == "draft", f"read {s}")
    ok(c.put(f"/api/me/reports/{d.json()['id']}/confirm", headers=H, json={"extracted": d.json()["extracted"]}).status_code == 200, f"confirm {s}")
me = c.get("/api/me", headers=H).json()
p = me["profile"]
ok(any("Metformin" in m for m in p["medications"]), f"meds merged into profile: {p['medications']}")
ok(any("Penicillin" in a for a in p["allergies"]), f"allergy merged: {p['allergies']}")
ok("Type 2 diabetes" in p["conditions"], f"conditions merged: {p['conditions']}")
ok(len(p["medications"]) == 2, "no duplicate meds")

# upload validation
ok(c.post("/api/me/reports", headers=H, files={"file": ("x.exe", b"MZ", "application/x-msdownload")}).status_code == 415, "bad file type rejected")
ok(c.post("/api/me/reports", headers=H, files={"file": ("big.png", b"0" * (7 * 1024 * 1024), "image/png")}).status_code == 413, "oversize rejected")

# emergency gate
r = c.put("/api/me/profile", headers=H, json={**{k: p[k] for k in ("name", "dob", "allergies", "medications", "conditions", "notes")}, "todays_concern": "Crushing chest pain since an hour"})
ok(r.json()["emergency_warning"], "emergency wording triggers warning")
c.put("/api/me/profile", headers=H, json={**{k: p[k] for k in ("name", "dob", "allergies", "medications", "conditions")}, "todays_concern": "Tired and thirsty", "notes": "PRIVATE-NOTE"})

# share: notes NOT included
rid = [x["id"] for x in me["reports"] if x["extracted"]["report_type"] == "lab"][0]
ok(c.post("/api/me/shares", headers=H, json={"doctor_name": "Dr Rao", "sections": []}).status_code == 422, "empty share rejected")
ok(c.post("/api/me/shares", headers=H, json={"doctor_name": "Dr Rao", "sections": ["reports"], "report_ids": []}).status_code == 422, "reports section needs a report")
sh = c.post("/api/me/shares", headers=H, json={"doctor_name": "Dr Rao", "sections": ["allergies", "medications", "todays_concern", "reports"], "report_ids": [rid], "expires_minutes": 30}).json()

d = c.get(f"/api/shared/{sh['token']}"); ok(d.status_code == 200, "doctor opens link without account")
j = d.json()
ok("notes" not in j["profile"] and "PRIVATE-NOTE" not in d.text, "private notes are NOT shared")
ok("conditions" not in j["profile"], "unshared section is hidden")
ok(len(j["reports"]) == 1 and j["reports"][0]["lab_results"], "only the chosen report is shared")
ok("Reason for visit" in j["summary"], "summary produced")
f = c.get(f"/api/shared/{sh['token']}/report/{rid}/file"); ok(f.status_code == 200 and f.headers["content-type"] == "image/png", "doctor can open original")
other = [x["id"] for x in me["reports"] if x["id"] != rid][0]
ok(c.get(f"/api/shared/{sh['token']}/report/{other}/file").status_code == 404, "unshared report file blocked")
ok(c.get("/api/shared/not-a-token").status_code == 404, "bad token 404")

log = c.get("/api/me/access-log", headers=H).json()
ok(len(log) == 2 and {x["action"] for x in log} == {"Opened the shared record", "Viewed original: sample_lab_report.png"}, f"access log recorded: {[x['action'] for x in log]}")

# isolation between patients
r2 = c.post("/api/patients").json(); H2 = {"X-Patient-Token": r2["token"]}
ok(c.get("/api/me", headers=H2).json()["reports"] == [], "another patient sees none of it")
ok(c.get("/api/me/access-log", headers=H2).json() == [], "another patient's log is empty")
ok(c.delete(f"/api/me/reports/{rid}", headers=H2).status_code == 404, "cannot delete someone else's report")

# revoke + expiry
ok(c.post(f"/api/me/shares/{sh['id']}/revoke", headers=H).status_code == 200, "revoke")
ok(c.get(f"/api/shared/{sh['token']}").status_code == 410, "revoked link stops working")
sh2 = c.post("/api/me/shares", headers=H, json={"doctor_name": "Dr Kim", "sections": ["allergies"], "expires_minutes": 5}).json()
from app import db  # noqa: E402
db.run("UPDATE shares SET expires_at = 1 WHERE id = ?", (sh2["id"],))
ok(c.get(f"/api/shared/{sh2['token']}").status_code == 410, "expired link stops working")
states = {s["doctor_name"]: s["state"] for s in c.get("/api/me/shares", headers=H).json()}
ok(states == {"Dr Rao": "revoked", "Dr Kim": "expired"}, f"states shown to patient {states}")

# pages
ok(c.get("/").status_code == 200 and c.get("/doctor").status_code == 200, "pages served")

# daily cap
os.environ["DAILY_LLM_CALLS_PER_IP"] = "2"
from app import llm  # noqa: E402
llm._usage.clear()
codes = [c.post("/api/me/reports/sample", headers=H, json={"name": samples[0]}).status_code for _ in range(4)]
ok(codes[-1] == 429 and codes[0] == 200, f"per-visitor AI cap enforced {codes}")

# deleting wipes file
shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)
print("\nALL CHECKS PASSED")
