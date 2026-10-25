"""Generate synthetic sample reports (PNG). All names/values are fake."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "app" / "samples"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def page(name, title, org, lines):
    img = Image.new("RGB", (900, 1100), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 900, 90], fill=(30, 70, 110))
    d.text((30, 25), org, fill="white", font=font(30, True))
    d.text((30, 110), title, fill=(20, 20, 20), font=font(26, True))
    y = 170
    for ln in lines:
        bold = ln.startswith("#")
        ln = ln.lstrip("# ")
        d.text((30, y), ln, fill=(20, 20, 20), font=font(22, bold))
        y += 38
    d.text((30, 1050), "SAMPLE - SYNTHETIC DATA FOR DEMO ONLY", fill=(150, 0, 0), font=font(18, True))
    img.save(OUT / name)


page("sample_lab_report.png", "LABORATORY REPORT", "City Diagnostics Lab", [
    "Patient: Asha Verma    DOB: 1958-04-12",
    "Collected: 2026-09-10    Reported: 2026-09-11",
    "",
    "# Test                 Result      Reference     Flag",
    "HbA1c                  8.1 %       4.0 - 5.6     H",
    "Fasting glucose        148 mg/dL   70 - 99       H",
    "Creatinine             0.9 mg/dL   0.6 - 1.3",
    "Total cholesterol      212 mg/dL   < 200         H",
    "Sodium                 139 mmol/L  135 - 145",
])

page("sample_prescription.png", "PRESCRIPTION", "Riverside Family Clinic", [
    "Patient: Asha Verma    DOB: 1958-04-12",
    "Date: 2026-09-14",
    "",
    "Rx 1: Metformin 500 mg - take twice daily with meals",
    "Rx 2: Lisinopril 10 mg - take once daily in the morning",
    "",
    "Allergy: Penicillin (rash)",
    "",
    "Dr. R. Menon    Reg. no. 000000 (sample)",
])

page("sample_discharge_summary.png", "DISCHARGE SUMMARY", "St. Mary Hospital", [
    "Patient: Asha Verma    DOB: 1958-04-12",
    "Admitted: 2026-06-30    Discharged: 2026-07-02",
    "",
    "# Diagnoses",
    "Type 2 diabetes mellitus",
    "Hypertension",
    "",
    "# Hospital course",
    "Admitted with high blood sugar. Treated and stabilised.",
    "",
    "# Follow-up",
    "See primary doctor within 2 weeks.",
])
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
