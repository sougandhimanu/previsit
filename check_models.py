"""Check your Token Factory key and see which NVIDIA Nemotron models you can use.

    python check_models.py            # list Nemotron models in the catalog
    python check_models.py --probe    # also test a tiny text call and a tiny image call on each

Run this first. It tells you what to put in TEXT_MODEL and VISION_MODEL (.env).
"""
import base64
import io
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw

load_dotenv()
key = os.getenv("NEBIUS_API_KEY", "").strip()
if not key:
    sys.exit("NEBIUS_API_KEY is missing. Copy .env.example to .env and paste your key.")

client = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"), api_key=key)
ids = sorted(m.id for m in client.models.list().data)
nemo = [i for i in ids if "nemotron" in i.lower()]
print(f"{len(ids)} models in catalog, {len(nemo)} Nemotron:")
for i in nemo:
    print("  ", i)
if not nemo:
    print("No Nemotron model found. Other NVIDIA models:", [i for i in ids if "nvidia" in i.lower()])
    sys.exit(0)

if "--probe" in sys.argv:
    img = Image.new("RGB", (160, 80), "white")
    ImageDraw.Draw(img).text((20, 30), "HbA1c 8.1", fill="black")
    buf = io.BytesIO(); img.save(buf, "PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    for m in nemo:
        if any(x in m.lower() for x in ("embed", "rerank", "guard", "safety", "reward", "parse")):
            continue
        try:
            r = client.chat.completions.create(model=m, max_tokens=20, messages=[{"role": "user", "content": "Say OK."}])
            print(f"[text ok]   {m}: {(r.choices[0].message.content or '').strip()[:40]!r}")
        except Exception as e:
            print(f"[text FAIL] {m}: {str(e)[:100]}")
            continue
        try:
            r = client.chat.completions.create(model=m, max_tokens=30, messages=[{"role": "user", "content": [
                {"type": "text", "text": "What number is written in the image?"},
                {"type": "image_url", "image_url": {"url": data_url}}]}])
            print(f"[image ok]  {m}: {(r.choices[0].message.content or '').strip()[:60]!r}")
        except Exception as e:
            print(f"[image no]  {m}: {str(e)[:80]}")
