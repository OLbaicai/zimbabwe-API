import base64, io, time
from pathlib import Path
import requests
from PIL import Image, ImageOps

API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME = "HunyuanOCR"
API_KEY = "sk-placeholder-key"
IMAGE_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\身份证ID-金属\金属ID")
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\身份证ID-金属\金属ID-PROMPT10-OUT")
MAX_COUNT = 0
TIMEOUT = 120
MAX_RETRIES = 2
RETRY_DELAY = 3

# OCR 提示词 --- 策略：综合最优 = PROMPT2英文结构 + PROMPT3性别强调 + 允许id_number带空格
PROMPT = """Extract these 5 fields from this Zimbabwe ID card. Return ONLY a raw JSON object.

{
  "id_number": "",
  "surname": "",
  "first_name": "",
  "birth_date": "",
  "gender": ""
}

EXTRACTION RULES:

1. id_number — The national ID number at the top of the card.
   - It looks like "02 033711 E 02" on the card (spaces between digit groups).
   - Copy ALL digits and the single uppercase letter exactly as you see them. Spaces are OK, we will clean them later.
   - Do NOT drop leading zeros. Do NOT add or remove characters.
   - Ignore any "CIT" annotation text.

2. surname — The line DIRECTLY BELOW the id_number, in ALL CAPS.
   - Example: "NGULUBE", "MOYO", "SIBANDA"
   - WARNING: "ZIMBABWE" is the country name at the very top — NEVER use it as surname.

3. first_name — The line DIRECTLY BELOW the surname, in ALL CAPS.
   - Example: "ORIEL", "SIBONOKUHLE", "RODNEY JAMES"
   - Keep compound names with spaces intact.

4. birth_date — Below the first_name. Output as DD.MM.YYYY with dots.
   - Example: "12.01.1961" (NOT "12 01 1961", NOT "12/01/1961")

5. gender — Look carefully at the right side of the card, near the id_number row.
   - There is a single letter: "M" or "F"
   - "M" -> output "male" ; "F" -> output "female"
   - IMPORTANT: Look at this letter VERY carefully. The letter F means female, M means male.
   - If you are not 100% sure about the gender letter, leave the field empty ("").

OUTPUT: ONLY the raw JSON. No markdown fences, no explanations.
"""

def _prepare_image(image_path):
    suffix = image_path.suffix.lower()
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    fmt = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    if fmt == "JPEG" and img.mode != "RGB":
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode == "P":
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return buf.getvalue(), mime

def _call_api(image_path):
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"API error: {e}") from e

def ocr_with_retry(image_path):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            return _call_api(image_path)
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES + 1:
                time.sleep(RETRY_DELAY)
    raise last_exc

def main():
    if not IMAGE_DIR.exists():
        print(f"[ERROR] {IMAGE_DIR}")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)
    if MAX_COUNT > 0:
        images = images[:MAX_COUNT]
    total = len(images)
    width = len(str(total))
    print(f"DIR: {IMAGE_DIR}")
    print(f"OUT: {OUTPUT_DIR}")
    print(f"COUNT: {total}\n")
    success = skip = fail = 0
    fails = []
    for idx, img_path in enumerate(images, 1):
        prefix = f"[{idx:{width}d}/{total}]"
        out_path = OUTPUT_DIR / (img_path.stem + ".txt")
        if out_path.exists():
            print(f"{prefix} SKIP: {img_path.name}")
            skip += 1
            continue
        print(f"{prefix} PROC: {img_path.name}", end="", flush=True)
        try:
            result = ocr_with_retry(img_path)
            out_path.write_text(result, encoding="utf-8")
            print(f" -> OK")
            success += 1
        except Exception as exc:
            print(f" -> FAIL: {exc}")
            fail += 1
            fails.append((img_path.name, str(exc)))
    print(f"\n{'─'*55}")
    print(f"DONE  OK:{success} SKIP:{skip} FAIL:{fail} TOTAL:{total}")

if __name__ == "__main__":
    main()
