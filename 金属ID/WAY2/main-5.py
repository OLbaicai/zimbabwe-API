import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image, ImageOps

# =========================================================
# 基础配置
# =========================================================

API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME = "HunyuanOCR"
API_KEY = "sk-placeholder-key"

IMAGE_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID")
OCR_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-OUT")
DEAL_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-5-DEAL-OUT")

MAX_COUNT = 0
TIMEOUT = 120
MAX_RETRIES = 2
RETRY_DELAY = 3
WORKERS = 5

# =========================================================
# Prompt
# =========================================================

STRUCTURED_PROMPT = """
You are extracting fields from a Zimbabwe national ID card.

Return ONLY one JSON object with exactly these keys:
id_number, surname, first_name, birth_date, gender

Rules:
- id_number:
  format = 8 digits + 1 uppercase letter + 2 digits
  remove spaces, hyphens, slashes, dots and CIT suffix

- surname:
  family name only

- first_name:
  given names only

- birth_date:
  normalize to DD.MM.YYYY

- gender:
  output ONLY:
  male
  female

Important:
- NEVER output explanations
- NEVER hallucinate
- If field missing -> ""

OCR summary:
{ocr_summary}
"""

# =========================================================
# 正则
# =========================================================

_ID_RE_PRIMARY = re.compile(
    r"(\d{2})[\s\-./]*(\d{6,7})[\s\-./]*([A-Z])[\s\-./]*(\d{2})"
)

_ID_RE_SECONDARY = re.compile(
    r"(\d{2})[\s\-./]*(\d{5})[\s\-./]*([A-Z])[\s\-./]*(\d{2})"
)

_ID_RE_FALLBACK = re.compile(
    r"(\d{2})\D+(\d{4,7})\D*([A-Z])\D*(\d{2})"
)

_DATE_RE = re.compile(
    r"(\d{1,2})[\s./\-](\d{1,2})[\s./\-](\d{2,4})"
)

_SEGMENT_RE = re.compile(
    r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)",
    re.DOTALL
)

_FIELDS = (
    "id_number",
    "surname",
    "first_name",
    "birth_date",
    "gender"
)

# =========================================================
# 噪声词
# =========================================================

_STOPWORDS = {
    "ZIMBABWE",
    "NATIONAL",
    "REGISTRATION",
    "NUMBER",
    "REPUBLIC",
    "IDENTITY",
    "CARD",
    "CIT",
    "CITIZEN",
    "NONE",
}

_PLACE_NAMES = {
    "BEITBRIDGE",
    "BULAWAYO",
    "MBERENGWA",
    "ZVISHAVANE",
    "BIKITA",
    "HARARE",
    "MUTARE",
    "MASVINGO",
    "GWERU",
    "KWEKWE",
    "CHITUNGWIZA",
    "EPWORTH",
    "MARONDERA",
    "BINDURA",
    "RUSAPE",
    "KADOMA",
    "SHURUGWI",
    "CHIREDZI",
    "KAROI",
    "VICTORIA FALLS",
    "HWANGE",
    "CHINHOYI",
    "MAFANEDZA",
    "SIKOBOKOBO",
    "MUSHAYI",
    "CHAKOTERA",
    "KENTARIBA",
    "RIVAHSE",
    "NDUNA",
    "SARAI",
    "RUVENGO",
    "TOMANI",
    "GREEN SPUN",
    "HO EO CO",
    "HO ED CO",
    "NO ED CO",
}

_ALL_NOISE = _STOPWORDS | _PLACE_NAMES

# =========================================================
# 图像处理
# =========================================================

def _prepare_image(image_path: Path) -> tuple[bytes, str]:

    img = Image.open(image_path)

    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()

    img.save(
        buf,
        format="JPEG",
        quality=95
    )

    return buf.getvalue(), "image/jpeg"

# =========================================================
# OCR段解析
# =========================================================

def _extract_segments(raw_text: str) -> list[dict]:

    segments = []

    for match in _SEGMENT_RE.finditer(raw_text):

        text, x1, y1, x2, y2 = match.groups()

        t = text.strip()

        if not t:
            continue

        segments.append({
            "text": t,
            "cy": (int(y1) + int(y2)) // 2,
            "cx": (int(x1) + int(x2)) // 2,
        })

    return sorted(
        segments,
        key=lambda x: (x["cy"], x["cx"])
    )

# =========================================================
# OCR摘要
# =========================================================

def _build_ocr_summary(
        raw_text: str,
        max_lines: int = 30
) -> str:

    segments = _extract_segments(raw_text)

    lines = []

    for i, s in enumerate(segments, 1):

        t = s["text"]

        if not t:
            continue

        lines.append(f"{i}. {t}")

        if len(lines) >= max_lines:
            break

    return "\n".join(lines)

# =========================================================
# 文本清洗
# =========================================================

def _clean_name(text: str) -> str:

    text = text.upper()

    text = re.sub(r"[^A-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = []

    for w in text.split():

        if w in _ALL_NOISE:
            continue

        if len(w) <= 1:
            continue

        words.append(w)

    return " ".join(words)

# =========================================================
# ID清洗
# =========================================================

def _normalize_id(raw_id: str) -> str:

    raw_id = raw_id.upper()

    raw_id = raw_id.replace("CIT", "")

    raw_id = re.sub(r"[^A-Z0-9]", "", raw_id)

    return raw_id

# =========================================================
# 日期清洗
# =========================================================

def _normalize_date(text: str) -> str:

    m = _DATE_RE.search(text)

    if not m:
        return ""

    try:

        d = int(m.group(1))
        mo = int(m.group(2))
        y = int(m.group(3))

        if not (1 <= d <= 31):
            return ""

        if not (1 <= mo <= 12):
            return ""

        if len(m.group(3)) == 2:
            y = 1900 + y if y > 30 else 2000 + y

        if not (1920 <= y <= 2025):
            return ""

        return f"{str(d).zfill(2)}.{str(mo).zfill(2)}.{y}"

    except Exception:
        return ""

# =========================================================
# ID提取
# =========================================================

def _extract_id_from_ocr(
        segments: list[dict]
) -> tuple[str, int | None]:

    candidates = []

    for idx, s in enumerate(segments):

        text = s["text"].upper().strip()

        if text in _STOPWORDS:
            continue

        for pat in (
                _ID_RE_PRIMARY,
                _ID_RE_SECONDARY,
                _ID_RE_FALLBACK
        ):

            m = pat.search(text)

            if not m:
                continue

            raw_id = (
                f"{m.group(1)}"
                f"{m.group(2)}"
                f"{m.group(3)}"
                f"{m.group(4)}"
            )

            clean = _normalize_id(raw_id)

            if re.fullmatch(r"\d{8}[A-Z]\d{2}", clean):

                candidates.append((
                    s["cy"],
                    clean,
                    idx
                ))

                break

    if not candidates:
        return "", None

    candidates.sort()

    return candidates[0][1], candidates[0][2]

# =========================================================
# 性别提取
# =========================================================

def _extract_gender_from_ocr(
        segments: list[dict],
        id_idx: int | None
) -> str:

    if id_idx is not None:

        for offset in range(-2, 3):

            idx = id_idx + offset

            if 0 <= idx < len(segments):

                t = segments[idx]["text"].upper()

                if re.search(r"\bCIT\s*M\b", t):
                    return "male"

                if re.search(r"\bCIT\s*F\b", t):
                    return "female"

                if t.strip() == "M":
                    return "male"

                if t.strip() == "F":
                    return "female"

    for s in segments:

        t = s["text"].upper()

        if "CIT M" in t:
            return "male"

        if "CIT F" in t:
            return "female"

    return ""

# =========================================================
# 日期提取
# =========================================================

def _extract_date_from_ocr(
        segments: list[dict]
) -> str:

    dates = []

    for s in segments:

        dt = _normalize_date(s["text"])

        if dt:
            dates.append((s["cy"], dt))

    if not dates:
        return ""

    dates.sort()

    return dates[0][1]

# =========================================================
# 姓名提取
# =========================================================

def _extract_names_from_ocr(
        segments: list[dict],
        id_y: int | None
) -> tuple[str, str]:

    candidates = []

    for s in segments:

        text = s["text"]

        upper = text.upper().strip()

        if upper in _ALL_NOISE:
            continue

        if any(sw in upper for sw in _STOPWORDS):
            continue

        if _DATE_RE.search(upper):
            continue

        if _ID_RE_PRIMARY.search(upper):
            continue

        if _ID_RE_SECONDARY.search(upper):
            continue

        cleaned = _clean_name(text)

        if not cleaned:
            continue

        if len(cleaned) < 2:
            continue

        candidates.append((
            s["cy"],
            cleaned
        ))

    # 重点：
    # 津巴布韦ID里姓名通常在ID号码下面

    if id_y is not None:

        below = []

        for cy, text in candidates:

            if cy > id_y + 10:
                below.append((cy, text))

        if below:
            candidates = below

    # 去重

    deduped = []

    seen = set()

    for cy, text in candidates:

        key = text.upper()

        if key in seen:
            continue

        seen.add(key)

        deduped.append((cy, text))

    # 第一行为surname
    # 第二行为first_name

    if len(deduped) >= 2:

        surname = deduped[0][1]

        first_name = deduped[1][1]

        return surname, first_name

    elif len(deduped) == 1:

        return deduped[0][1], ""

    return "", ""

# =========================================================
# 结果质量判断
# =========================================================

def _needs_structured_fallback(
        result: dict
) -> bool:

    if not result.get("id_number"):
        return True

    if not result.get("surname"):
        return True

    if not result.get("first_name"):
        return True

    if not result.get("birth_date"):
        return True

    if result.get("gender") not in ("male", "female"):
        return True

    return False

# =========================================================
# 结构化API调用
# =========================================================

def _call_structured_api(
        image_path: Path,
        raw_ocr: str
) -> str:

    img_bytes, mime = _prepare_image(image_path)

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    ocr_summary = _build_ocr_summary(raw_ocr)

    prompt = STRUCTURED_PROMPT.format(
        ocr_summary=ocr_summary
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    for attempt in range(MAX_RETRIES + 1):

        try:

            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=TIMEOUT
            )

            resp.raise_for_status()

            return resp.json()["choices"][0]["message"]["content"].strip()

        except Exception:

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise

# =========================================================
# JSON解析
# =========================================================

def _extract_json_payload(
        text: str
) -> dict | None:

    cleaned = text.strip()

    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines)

    try:

        data = json.loads(cleaned)

    except Exception:

        m = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not m:
            return None

        try:
            data = json.loads(m.group(0))
        except Exception:
            return None

    if not isinstance(data, dict):
        return None

    return {
        "id_number": _normalize_id(
            str(data.get("id_number", ""))
        ),

        "surname": _clean_name(
            str(data.get("surname", ""))
        ),

        "first_name": _clean_name(
            str(data.get("first_name", ""))
        ),

        "birth_date": _normalize_date(
            str(data.get("birth_date", ""))
        ),

        "gender": str(
            data.get("gender", "")
        ).strip().lower(),
    }

# =========================================================
# 结果融合
# =========================================================

def _merge_results(
        heuristic: dict,
        structured: dict | None
) -> dict:

    if structured is None:
        return heuristic

    result = {}

    for field in _FIELDS:

        h_val = heuristic.get(field, "")
        s_val = structured.get(field, "")

        # 规则优先
        # 模型兜底

        if field == "gender":

            if h_val in ("male", "female"):
                result[field] = h_val
            else:
                result[field] = s_val

        elif field == "id_number":

            if re.fullmatch(r"\d{8}[A-Z]\d{2}", h_val):
                result[field] = h_val
            else:
                result[field] = s_val

        else:

            result[field] = h_val if h_val else s_val

    return result

# =========================================================
# 单图处理
# =========================================================

def process_single_image(
        image_path: Path
) -> dict:

    ocr_file = OCR_DIR / f"{image_path.stem}.txt"

    if not ocr_file.exists():
        raise FileNotFoundError(f"OCR缺失: {ocr_file}")

    raw_ocr = ocr_file.read_text(
        encoding="utf-8"
    ).strip()

    segments = _extract_segments(raw_ocr)

    # =====================================================
    # 第一阶段：规则提取
    # =====================================================

    id_number, id_idx = _extract_id_from_ocr(
        segments
    )

    id_y = None

    if id_idx is not None:
        id_y = segments[id_idx]["cy"]

    gender = _extract_gender_from_ocr(
        segments,
        id_idx
    )

    birth_date = _extract_date_from_ocr(
        segments
    )

    surname, first_name = _extract_names_from_ocr(
        segments,
        id_y
    )

    result = {
        "id_number": id_number,
        "surname": surname,
        "first_name": first_name,
        "birth_date": birth_date,
        "gender": gender,
    }

    # =====================================================
    # 第二阶段：LLM结构化回退
    # =====================================================

    if _needs_structured_fallback(result):

        try:

            structured_text = _call_structured_api(
                image_path,
                raw_ocr
            )

            structured = _extract_json_payload(
                structured_text
            )

            result = _merge_results(
                result,
                structured
            )

        except Exception:
            pass

    return result

# =========================================================
# 单任务
# =========================================================

def _process_one(
        img_p: Path
) -> tuple[str, bool, str]:

    try:

        result = process_single_image(img_p)

        out_path = DEAL_DIR / f"{img_p.stem}.txt"

        out_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        return img_p.name, True, ""

    except Exception as e:

        return img_p.name, False, str(e)

# =========================================================
# 主程序
# =========================================================

def main():

    DEAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png"
        }
    )

    if MAX_COUNT > 0:
        images = images[:MAX_COUNT]

    total = len(images)

    pending = []

    for p in images:

        out_file = DEAL_DIR / f"{p.stem}.txt"

        if not out_file.exists():
            pending.append(p)

    skipped = total - len(pending)

    if skipped:
        print(f"跳过已处理: {skipped}")

    if not pending:
        print("全部完成")
        return

    print(f"待处理: {len(pending)}")
    print(f"并发数: {WORKERS}")
    print()

    done = skipped
    fail = 0

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:

        futures = {
            pool.submit(_process_one, p): p
            for p in pending
        }

        for future in as_completed(futures):

            name, ok, err = future.result()

            done += 1

            if ok:

                elapsed = time.time() - t0

                rate = (
                    (done - skipped) / elapsed * 60
                    if elapsed > 0 else 0
                )

                print(
                    f"[{done}/{total}] "
                    f"OK  {name}  "
                    f"({rate:.0f}张/分)"
                )

            else:

                fail += 1

                print(
                    f"[{done}/{total}] "
                    f"FAIL {name}: {err}"
                )

    elapsed = time.time() - t0

    print()
    print(
        f"完成 "
        f"成功:{done-skipped-fail} "
        f"失败:{fail} "
        f"耗时:{elapsed:.0f}秒"
    )

# =========================================================
# 入口
# =========================================================

if __name__ == "__main__":
    main()

