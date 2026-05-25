import base64, io, json, re, time
from pathlib import Path
import requests
from PIL import Image, ImageOps

API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"
IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID")
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY1\txt\金属ID-PROMPT16-OUT")
MAX_COUNT, TIMEOUT, MAX_RETRIES, RETRY_DELAY = 0, 120, 2, 3

# PROMPT16: PROMPT15基础 + id_number算法式步骤 + gender中文强调
PROMPT = """Extract fields from this Zimbabwe ID card. Output ONLY a raw JSON object (no markdown, no ```json, no explanation).

{
  "id_number": "",
  "surname": "",
  "first_name": "",
  "birth_date": "",
  "gender": ""
}

STRICT EXTRACTION RULES:

1. id_number — 11 characters total. Extract using these steps:
   Step A: Ignore ALL spaces, dashes, dots, and the word "CIT" on the ID line.
   Step B: Find exactly 8 consecutive digits. These are the first 8 characters of the ID.
   Step C: The character IMMEDIATELY after those 8 digits must be an uppercase letter (A-Z). That is character 9.
   Step D: The 2 characters after that letter must be digits. Those are characters 10-11.
   Step E: Combine: the 8 digits + the uppercase letter + the 2 digits. Output exactly 11 characters.
   Step F: Do NOT drop any of these 11 characters. Do NOT add extras. 11 characters exactly.

2. surname — The ALL-UPPERCASE line directly BELOW id_number. Usually a single short word.
   WARNING: "ZIMBABWE" at the top of the card is the COUNTRY, NOT a surname. Never extract it.

3. first_name — The ALL-UPPERCASE line directly BELOW the surname. Keep spaces for compound names.

4. birth_date — Located BELOW first_name. Output format: DD.MM.YYYY (dots, 4-digit year).
   If year is 2 digits: 00-30→20XX, 31-99→19XX.

5. gender — Look for a single letter "F" or "M" near the id_number area. Output exactly what you see.
   If the letter is unclear or missing, leave empty. Do NOT guess.

OUTPUT: ONLY the raw JSON. No markdown. No explanation.
"""

# ═══════════════ 后处理 ═══════════════

def _clean_id_number(raw: str) -> str:
    if not raw: return ""
    s = re.sub(r'\bCIT\b', '', raw, flags=re.IGNORECASE)
    s = re.sub(r'[^A-Za-z0-9]', '', s)
    m = re.search(r'(\d{8})([A-Za-z])(\d{2})', s)
    if m: return m.group(1) + m.group(2).upper() + m.group(3)
    if len(s) >= 11:
        c = s[:11].upper()
        if sum(1 for x in c if x.isdigit()) >= 9: return c
    return s.upper()

def _clean_name(raw: str) -> str:
    if not raw: return ""
    s = raw.strip()
    return "" if s.upper() == "ZIMBABWE" else s

def _clean_birth_date(raw: str) -> str:
    if not raw: return ""
    s = re.sub(r'[\s\-/]+', '.', raw.strip())
    parts = s.split('.')
    if len(parts) != 3: return raw.strip()
    d, m, y = [re.sub(r'\D', '', p) for p in parts]
    if not d or not m or not y: return raw.strip()
    d, m = d.zfill(2), m.zfill(2)
    if len(y) == 2: y = f"19{y}" if int(y) > 30 else f"20{y}"
    return f"{d}.{m}.{y}"

def _clean_gender(raw: str) -> str:
    if not raw: return ""
    g = raw.strip().upper()
    if g in ("F", "FEMALE"): return "female"
    if g in ("M", "MALE"): return "male"
    return ""

def _post_process(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try: data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try: data = json.loads(m.group())
            except json.JSONDecodeError: return {}
        else: return {}
    if not isinstance(data, dict): return {}
    return {
        "id_number": _clean_id_number(data.get("id_number", "")),
        "surname": _clean_name(data.get("surname", "")),
        "first_name": _clean_name(data.get("first_name", "")),
        "birth_date": _clean_birth_date(data.get("birth_date", "")),
        "gender": _clean_gender(data.get("gender", "")),
    }

# ═══════════════ 基础设施 ═══════════════

def _prepare_image(image_path: Path) -> tuple[bytes, str]:
    suffix = image_path.suffix.lower()
    img = Image.open(image_path); img = ImageOps.exif_transpose(img)
    fmt = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    if fmt == "JPEG" and img.mode != "RGB":
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1]); img = bg
        elif img.mode == "P":
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1]); img = bg
        else: img = img.convert("RGB")
    buf = io.BytesIO(); img.save(buf, format=fmt, quality=95)
    return buf.getvalue(), mime

def _call_api(image_path: Path) -> str:
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": PROMPT},
    ]}], "temperature": 0.0, "max_tokens": 4096}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def ocr_with_retry(image_path: Path) -> str:
    last_exc = None
    for a in range(1, MAX_RETRIES + 2):
        try: return _call_api(image_path)
        except Exception as exc:
            last_exc = exc
            if a <= MAX_RETRIES: time.sleep(RETRY_DELAY)
    raise last_exc

def main() -> None:
    if not IMAGE_DIR.exists(): print(f"[错误] 图片目录不存在: {IMAGE_DIR}"); return
    if not IMAGE_DIR.is_dir(): print(f"[错误] 路径不是目录: {IMAGE_DIR}"); return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)
    if not images: print(f"[警告] 未找到图片: {IMAGE_DIR}"); return
    if MAX_COUNT > 0: images = images[:MAX_COUNT]

    total = len(images); width = len(str(total))
    print(f"图片目录 : {IMAGE_DIR}\n输出目录 : {OUTPUT_DIR}\n处理数量 : {total} 张\n")
    sc = sk = fc = pf = 0; fl = []

    for idx, ip in enumerate(images, 1):
        prefix = f"[{idx:{width}d}/{total}]"
        op = OUTPUT_DIR / (ip.stem + ".txt")
        if op.exists(): print(f"{prefix} 跳过: {ip.name}"); sk += 1; continue
        print(f"{prefix} 处理: {ip.name}", end="", flush=True)
        try:
            raw = ocr_with_retry(ip)
            cleaned = _post_process(raw)
            if not cleaned: op.write_text(raw, encoding="utf-8"); print(f"  → 后处理失败"); pf += 1; continue
            op.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
            print(f"  → 已保存"); sc += 1
        except Exception as exc:
            print(f"  → 失败: {exc}"); fc += 1; fl.append((ip.name, str(exc)))

    print(f"\n{'─'*55}\n处理完成  成功:{sc} 跳过:{sk} 后处理失败:{pf} 请求失败:{fc} 共:{total}")
    if fl: print(f"\n失败列表:"); [print(f"  ✗ {n}\n      {e}") for n, e in fl]

if __name__ == "__main__": main()
