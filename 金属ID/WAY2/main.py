# main.py是单发，速度较慢，main-1之后都是可以并发的，并发数是workers设置
import base64
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps

# API 地址与鉴权
API_BASE_URL = "http://10.10.185.18:30067/v1"
MODEL_NAME   = "HunyuanOCR"
API_KEY      = "sk-placeholder-key"

# 待识别图片目录
IMAGE_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\金属ID")
# 识别结果输出目录（不存在会自动创建）
OUTPUT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-OUT")
# 结构化结果输出目录
DEAL_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-DEAL-OUT")

# 最多处理张数（0 = 处理全部）
MAX_COUNT   = 0
# 单次请求超时（秒）
TIMEOUT     = 120
# 失败自动重试次数（0 = 不重试）
MAX_RETRIES = 2
# 每次重试前等待时间（秒）
RETRY_DELAY = 3

# OCR 提示词
PROMPT = """检测并识别图片中的文字，将文本坐标格式化输出。
"""

_SEGMENT_RE = re.compile(r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)", re.DOTALL)
_ID_RE = re.compile(r"(?:\d[\s-]*){8}[A-Z](?:[\s-]*\d){2}", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}(?:[./\-\s])\d{1,2}(?:[./\-\s])\d{2,4}\b")
_LINE_TOLERANCE = 25
_FIELD_ORDER = ("id_number", "surname", "first_name", "birth_date", "gender")
_NAME_STOPWORDS = {
    "ZIMBABWE",
    "NATIONAL",
    "REGISTRATION",
    "NUMBER",
    "CIT",
    "HO",
    "ED",
    "EO",
    "CO",
    # 常见地名/辖区词，避免被当成姓名
    "BEITBRIDGE",
    "BIKITA",
    "MBERENGWA",
}

STRUCTURED_PROMPT_TEMPLATE = """You are extracting fields from a Zimbabwe ID card.

Return ONLY one JSON object with exactly these keys in this order:
id_number, surname, first_name, birth_date, gender

Rules:
- id_number: 8 digits + 1 uppercase letter + 2 digits. Remove spaces, hyphens, and CIT.
- surname: the family name on the card. Ignore header/noise words like ZIMBABWE, NATIONAL, REGISTRATION, NATIONAL NUMBER, HO EO CO, BIKITA, BEITBRIDGE.
- first_name: the given name(s) on the card. Keep spaces inside compound names.
- birth_date: normalize to DD.MM.YYYY. If the source has 2-digit year, expand it to 19xx or 20xx.
- gender: output male or female. If the card shows M or F (often near `CIT`), map them accordingly.
- Do NOT guess. If you cannot see a field in the image/OCR lines, return an empty string.

OCR line summary:
{ocr_summary}
"""

# ══════════════════════════════════════════════════════════════


def _prepare_image(image_path: Path) -> tuple[bytes, str]:
    """
    打开图片，修正 EXIF 旋转，转为 JPEG/PNG 字节流。
    返回 (图片字节, MIME类型)。
    """
    suffix = image_path.suffix.lower()
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)  # 修正手机拍摄的旋转信息

    fmt  = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"

    # JPEG 不支持透明通道，统一转 RGB
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


def _call_api(image_path: Path) -> str:
    """调用 HunyuanOCR API，返回识别文本（已 strip）。"""
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    resp = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"API 返回格式异常: {e}，原始响应: {data}") from e


def ocr_with_retry(image_path: Path) -> str:
    """带重试的 OCR 调用，全部失败则抛出最后一次异常。"""
    last_exc = None
    attempts = MAX_RETRIES + 1
    for attempt in range(1, attempts + 1):
        try:
            return _call_api(image_path)
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(RETRY_DELAY)
    raise last_exc


def _strip_code_fences(text: str) -> str:
    """去掉模型偶发返回的 markdown 代码块。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_segments(raw_text: str) -> list[dict[str, int | str]]:
    """把 HunyuanOCR 的坐标串拆成按版面排序的文本片段。"""
    segments: list[dict[str, int | str]] = []
    for match in _SEGMENT_RE.finditer(raw_text):
        text = match.group(1).strip()
        if not text:
            continue
        x1, y1, x2, y2 = (int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        segments.append({
            "text": text,
            "x": x1,
            "y": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) // 2,
            "cy": (y1 + y2) // 2,
        })
    segments.sort(key=lambda item: (int(item["y"]), int(item["x"])))
    return segments


def _infer_page_size(segments: list[dict[str, int | str]]) -> tuple[int, int]:
    """从 OCR 坐标推断页面宽高（用于相对位置打分）。"""
    if not segments:
        return 0, 0
    max_x = 0
    max_y = 0
    for segment in segments:
        try:
            max_x = max(max_x, int(segment.get("x2", 0)))
            max_y = max(max_y, int(segment.get("y2", 0)))
        except Exception:
            continue
    return max_x, max_y


def _cluster_lines(segments: list[dict[str, int | str]]) -> list[dict[str, object]]:
    """按 y 坐标把片段聚成行。"""
    lines: list[dict[str, object]] = []
    for segment in segments:
        y = int(segment["y"])
        y2 = int(segment.get("y2", y))
        x1 = int(segment.get("x", 0))
        x2 = int(segment.get("x2", x1))
        if not lines or y - int(lines[-1]["y_max"]) > _LINE_TOLERANCE:
            lines.append({
                "segments": [segment],
                "y_min": y,
                # y_max 用 top-y 聚行，避免高框把后续行吞掉
                "y_max": y,
                # y2_max 仅用于几何中心/范围
                "y2_max": y2,
                "x_min": x1,
                "x_max": x2,
            })
            continue
        line_segments = lines[-1]["segments"]  # type: ignore[assignment]
        line_segments.append(segment)
        lines[-1]["y_min"] = min(int(lines[-1]["y_min"]), y)
        lines[-1]["y_max"] = max(int(lines[-1]["y_max"]), y)
        lines[-1]["y2_max"] = max(int(lines[-1].get("y2_max", lines[-1]["y_max"])), y2)
        lines[-1]["x_min"] = min(int(lines[-1]["x_min"]), x1)
        lines[-1]["x_max"] = max(int(lines[-1]["x_max"]), x2)

    for line in lines:
        line_segments = sorted(line["segments"], key=lambda item: int(item["x"]))  # type: ignore[index]
        line["segments"] = line_segments
        line["text"] = " ".join(str(item["text"]).strip() for item in line_segments if str(item["text"]).strip())
        try:
            line["cx"] = (int(line["x_min"]) + int(line["x_max"])) // 2
            bottom = int(line.get("y2_max", line["y_max"]))
            line["cy"] = (int(line["y_min"]) + bottom) // 2
        except Exception:
            line["cx"] = 0
            line["cy"] = 0
    return lines


def _score_name_candidate(cleaned_name: str, raw_line_text: str, line: dict[str, object], page_w: int, page_h: int) -> int:
    tokens = cleaned_name.split()
    joined = cleaned_name.replace(" ", "")
    if not joined:
        return -10_000

    score = len(joined)

    # token 数越少越像姓名（地名/描述性文本通常更长更碎）
    if len(tokens) == 1:
        score += 30
    elif len(tokens) == 2:
        score += 18
    elif len(tokens) == 3:
        score += 8
    else:
        score -= 10

    # 太短的 token 往往是噪声（HO/ED/CO 等）
    if any(len(t) <= 3 for t in tokens):
        score -= 20

    # 若行里混入明显抬头词，强扣分
    upper_raw = raw_line_text.upper()
    if any(re.search(rf"\b{re.escape(word)}\b", upper_raw) for word in _NAME_STOPWORDS):
        score -= 25

    # 位置：姓名通常在版面中间区域（用相对坐标弱约束）
    try:
        cx = int(line.get("cx", 0))
        cy = int(line.get("cy", 0))
    except Exception:
        cx, cy = 0, 0

    if page_w > 0:
        if int(page_w * 0.20) <= cx <= int(page_w * 0.80):
            score += 12
        else:
            score -= 6
    if page_h > 0:
        if int(page_h * 0.22) <= cy <= int(page_h * 0.85):
            score += 8

    return score


def _pick_names_by_scoring(
    lines: list[dict[str, object]],
    start_index: int,
    end_index: int,
    page_w: int,
    page_h: int,
) -> tuple[str, str]:
    candidates: list[tuple[int, int, str]] = []
    for idx in range(start_index, max(start_index, end_index)):
        line = lines[idx]
        raw_text = str(line.get("text", "")).strip()
        if not raw_text:
            continue
        if _DATE_RE.search(raw_text):
            continue
        if re.search(r"\d", raw_text):
            # 含大量数字的行一般不是姓名（但 OCR 噪声也会夹数字，这里做轻过滤）
            if len(re.findall(r"\d", raw_text)) >= 2:
                continue

        cleaned = _clean_name_line(raw_text)
        if not cleaned:
            continue
        score = _score_name_candidate(cleaned, raw_text, line, page_w, page_h)
        candidates.append((score, idx, cleaned))

    if not candidates:
        return "", ""

    # 按分数挑前两名，确保不重复
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    picked: list[tuple[int, str]] = []  # (idx, cleaned)
    seen = set()
    for _, idx, cleaned in candidates:
        if cleaned in seen:
            continue
        seen.add(cleaned)
        picked.append((idx, cleaned))
        if len(picked) >= 2:
            break

    if not picked:
        return "", ""
    if len(picked) == 1:
        return picked[0][1], ""

    picked.sort(key=lambda item: item[0])
    surname = picked[0][1]
    first_name = picked[1][1]
    return surname, first_name


def _pick_first_token_name(cleaned_name: str) -> str:
    """把候选姓名短语缩成更像 given name 的单词（常见：生日后面跟 given name + 地名）。"""
    tokens = [t for t in cleaned_name.split() if t]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    # 多 token 时优先取第一个（通常是 given name），避免把地名一起带上
    return tokens[0]


def _clean_alpha_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", token).upper().strip()
    if not cleaned:
        return ""
    if cleaned in _NAME_STOPWORDS:
        return ""
    return cleaned


def _token_score(token: str, cx: int, cy: int, page_w: int, page_h: int) -> int:
    if not token:
        return -10_000
    if len(token) < 3:
        return -10_000

    score = len(token) * 4

    # 3~4 字母短码（如 HCVE）在标注里经常作为 surname
    if 3 <= len(token) <= 4:
        score += 20
    elif len(token) >= 7:
        score += 10

    # 位置弱约束：多数姓名块位于偏左/中部
    if page_w > 0:
        if cx <= int(page_w * 0.70):
            score += 10
        else:
            score -= 6
        if cx <= int(page_w * 0.55):
            score += 6
    if page_h > 0:
        if int(page_h * 0.20) <= cy <= int(page_h * 0.90):
            score += 6

    return score


def _pick_code_name_override(segments: list[dict[str, int | str]], page_w: int, page_h: int) -> tuple[str, str]:
    """窄范围兜底：当出现“3~4 字母短码 + 邻近的长名字”时，按标注习惯覆盖 surname/first_name。

    这类卡片里短码经常就是标注的 surname（例如 HCVE），长词是 first_name。
    """
    short: list[dict[str, int | str]] = []
    long: list[dict[str, int | str]] = []

    for seg in segments:
        raw = str(seg.get("text", "")).strip()
        if not raw:
            continue
        if _DATE_RE.search(raw) or re.search(r"\d", raw):
            continue
        for part in re.split(r"\s+", raw):
            t = _clean_alpha_token(part)
            if not t:
                continue
            cx = int(seg.get("cx", seg.get("x", 0)))
            cy = int(seg.get("cy", seg.get("y", 0)))

            # 常见地名不作为姓名覆盖来源
            if t in {"BEITBRIDGE", "BIKITA", "MBERENGWA"}:
                continue

            if 3 <= len(t) <= 4:
                short.append({"t": t, "cx": cx, "cy": cy})
            elif len(t) >= 6:
                long.append({"t": t, "cx": cx, "cy": cy})

    if not short or not long:
        return "", ""

    best = None  # (score, short_token, long_token)
    for s in short:
        for l in long:
            dx = abs(int(s["cx"]) - int(l["cx"]))
            dy = abs(int(s["cy"]) - int(l["cy"]))
            # 必须足够近，避免跨区域误配
            if dx > 220 or dy > 260:
                continue

            score = 0
            score += 30 - min(30, dx // 10)
            score += 30 - min(30, dy // 10)
            # 位置弱约束：偏左/中部优先
            if page_w > 0 and int(l["cx"]) <= int(page_w * 0.70) and int(s["cx"]) <= int(page_w * 0.70):
                score += 10
            if page_h > 0 and int(page_h * 0.20) <= int(l["cy"]) <= int(page_h * 0.90):
                score += 6

            if best is None or score > best[0]:
                best = (score, str(s["t"]), str(l["t"]))

    if best is None:
        return "", ""
    return best[1], best[2]


def _normalize_birth_date(date_text: str) -> str:
    """把日期统一成 DD.MM.YYYY。"""
    cleaned = date_text.strip().replace("/", ".").replace("-", ".")
    parts = [part for part in re.split(r"[.\s]+", cleaned) if part]
    if len(parts) != 3:
        return cleaned

    day, month, year = parts
    day = day.zfill(2)
    month = month.zfill(2)
    if len(year) == 2:
        year = ("19" if int(year) > 30 else "20") + year
    return f"{day}.{month}.{year}"


def _normalize_gender(gender_text: str) -> str:
    """统一性别为 male / female。"""
    gender = gender_text.strip().lower()
    if gender in {"m", "male"}:
        return "male"
    if gender in {"f", "female"}:
        return "female"
    return gender


def _extract_gender_from_id_area(lines: list[dict[str, object]], id_line_index: int) -> str:
    """从 ID 行附近提取性别，优先识别显式的 `CIT M/F`。"""
    candidate_texts: list[str] = []
    for offset in (-1, 0, 1):
        idx = id_line_index + offset
        if 0 <= idx < len(lines):
            candidate_texts.append(str(lines[idx].get("text", "")))

    joined = " ".join(text for text in candidate_texts if text)
    if not joined:
        return ""

    cit_match = re.search(r"\bCIT\b\s*([MF])\b", joined, re.IGNORECASE)
    if cit_match:
        return _normalize_gender(cit_match.group(1))

    lone_match = re.search(r"\b([MF])\b", joined, re.IGNORECASE)
    if lone_match:
        return _normalize_gender(lone_match.group(1))

    return ""


def _clean_name_line(line_text: str) -> str:
    """去掉版面头词后，保留姓名候选文本。"""
    tokens = []
    for token in re.split(r"\s+", line_text.strip()):
        cleaned = re.sub(r"[^A-Za-z]", "", token).upper()
        if not cleaned or cleaned in _NAME_STOPWORDS:
            continue
        tokens.append(cleaned)
    return " ".join(tokens).strip()


def _looks_like_id_piece(piece: str) -> bool:
    """判断清洗后的 11 位片段是否像 Zimbabwe ID 号。"""
    if len(piece) != 11:
        return False

    for index, ch in enumerate(piece):
        if index == 8:
            if not ch.isalpha():
                return False
            continue
        if ch.isdigit():
            continue
        if ch in {"O", "Q", "C", "D"}:
            continue
        return False
    return True


def _normalize_id_piece(piece: str) -> str:
    """把 OCR 容易混淆的字符归一化到 id_number。"""
    normalized: list[str] = []
    for index, ch in enumerate(piece):
        if index == 8:
            normalized.append(ch)
            continue
        if ch in {"O", "Q", "C", "D"}:
            normalized.append("0")
        else:
            normalized.append(ch)
    return "".join(normalized)


def _extract_id_number(line_text: str) -> str:
    """从 id 行中提取 8 数字 + 1 字母 + 2 数字的号码。"""
    compact = re.sub(r"[^A-Z0-9]", "", line_text.upper())
    for start in range(0, max(len(compact) - 10, 0)):
        piece = compact[start:start + 11]
        if _looks_like_id_piece(piece):
            return _normalize_id_piece(piece)
    return ""


def _parse_structured_fields(raw_text: str) -> dict[str, str]:
    """从原始 OCR 输出中提取五字段，兼容 JSON 和坐标文本两种形态。"""
    cleaned = _strip_code_fences(raw_text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {
                "id_number": re.sub(r"[\s-]+", "", str(data.get("id_number", ""))).upper(),
                "surname": str(data.get("surname", "")).strip().upper(),
                "first_name": str(data.get("first_name", "")).strip().upper(),
                "birth_date": _normalize_birth_date(str(data.get("birth_date", ""))),
                "gender": _normalize_gender(str(data.get("gender", ""))),
            }
    except Exception:
        pass

    segments = _extract_segments(cleaned)
    page_w, page_h = _infer_page_size(segments)
    lines = _cluster_lines(segments)

    result = {field: "" for field in _FIELD_ORDER}
    id_line_index = None

    for idx, line in enumerate(lines):
        line_text = str(line["text"]).strip()
        normalized_id = _extract_id_number(line_text)
        if not normalized_id:
            continue

        id_line_index = idx
        result["id_number"] = normalized_id

        extracted_gender = _extract_gender_from_id_area(lines, id_line_index)
        if extracted_gender:
            result["gender"] = extracted_gender
        break

    if id_line_index is None:
        return result

    # 先确定生日所在行（用于限定姓名候选范围）
    date_line_index = None
    date_tail_name = ""
    for idx in range(id_line_index + 1, len(lines)):
        line_text = str(lines[idx].get("text", "")).strip()
        if not line_text:
            continue
        date_match = _DATE_RE.search(line_text)
        if date_match:
            if not result["birth_date"]:
                result["birth_date"] = _normalize_birth_date(date_match.group(0))
            tail_text = line_text[date_match.end():].strip()
            tail_cleaned = _clean_name_line(tail_text)
            if tail_cleaned:
                date_tail_name = _pick_first_token_name(tail_cleaned)
            date_line_index = idx
            break

    # 姓名优先：在 id 行与生日行之间按坐标与文本特征打分选取
    name_start = id_line_index + 1
    name_end = date_line_index if date_line_index is not None else min(id_line_index + 10, len(lines))
    if not result["surname"] or not result["first_name"]:
        scored_surname, scored_first = _pick_names_by_scoring(lines, name_start, name_end, page_w, page_h)
        if scored_surname and not result["surname"]:
            result["surname"] = scored_surname
        if scored_first and not result["first_name"]:
            result["first_name"] = scored_first
        elif date_tail_name and not result["first_name"]:
            result["first_name"] = date_tail_name

    # 最后再尝试短码 surname 覆盖（只在靠得近时触发）
    code_surname, code_first = _pick_code_name_override(segments, page_w, page_h)
    if code_surname and code_first:
        result["surname"] = code_surname
        result["first_name"] = code_first

    before_date_names: list[str] = []
    after_date_names: list[str] = []
    seen_birth_date = False
    for line in lines[id_line_index + 1 :]:
        line_text = str(line["text"]).strip()
        if not line_text:
            continue

        date_match = _DATE_RE.search(line_text)
        if date_match:
            if not result["birth_date"]:
                result["birth_date"] = _normalize_birth_date(date_match.group(0))
            seen_birth_date = True
            tail_text = line_text[date_match.end():].strip()
            tail_name = _clean_name_line(tail_text)
            if tail_name:
                after_date_names.append(tail_name)
            continue

        alpha_text = _clean_name_line(line_text)
        if not alpha_text:
            continue

        if seen_birth_date:
            after_date_names.append(alpha_text)
        else:
            before_date_names.append(alpha_text)

    short_after_date = [name for name in after_date_names if len(name.replace(" ", "")) <= 4]

    # 若已找到姓名，就不再用旧规则覆盖
    if result["surname"] or result["first_name"]:
        return result

    if short_after_date:
        result["surname"] = min(short_after_date, key=lambda item: len(item.replace(" ", "")))
        if before_date_names:
            result["first_name"] = before_date_names[0]
        elif len(after_date_names) > 1:
            result["first_name"] = after_date_names[1]
        elif after_date_names:
            result["first_name"] = after_date_names[0]
    elif before_date_names:
        if len(before_date_names) == 1:
            result["surname"] = before_date_names[0]
            result["first_name"] = ""
        else:
            result["surname"] = before_date_names[0]
            result["first_name"] = before_date_names[1]
    elif after_date_names:
        if len(after_date_names) == 1:
            result["surname"] = after_date_names[0]
            result["first_name"] = ""
        else:
            result["surname"] = after_date_names[0]
            result["first_name"] = after_date_names[1]

    return result


def _build_json_output(fields: dict[str, str]) -> str:
    """把字段按固定顺序写成 JSON txt。"""
    payload = {field: fields.get(field, "") for field in _FIELD_ORDER}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_ocr_summary(raw_text: str, max_lines: int = 16) -> str:
    """把原始 OCR 输出压缩成更适合二次结构化抽取的行摘要。"""
    lines = _cluster_lines(_extract_segments(raw_text))
    summary_lines: list[str] = []
    for idx, line in enumerate(lines, 1):
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        summary_lines.append(f"{idx}. {text}")
        if len(summary_lines) >= max_lines:
            break
    return "\n".join(summary_lines)


def _call_structured_api(image_path: Path, raw_text: str) -> str:
    """基于原图和 OCR 行摘要，调用模型生成结构化 JSON。"""
    img_bytes, mime = _prepare_image(image_path)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    ocr_summary = _build_ocr_summary(raw_text)
    prompt = STRUCTURED_PROMPT_TEMPLATE.format(ocr_summary=ocr_summary)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    resp = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"结构化 API 返回格式异常: {e}，原始响应: {data}") from e


def _extract_json_payload(text: str) -> dict[str, str] | None:
    """从模型回复中提取 JSON 对象。"""
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None

    if not isinstance(data, dict):
        return None

    result = {field: "" for field in _FIELD_ORDER}
    for field in _FIELD_ORDER:
        value = data.get(field, "")
        if value is None:
            continue
        result[field] = str(value).strip()
    return result


def _looks_like_valid_id(value: str) -> bool:
    compact = re.sub(r"[\s-]+", "", value.upper())
    return bool(re.fullmatch(r"\d{8}[A-Z]\d{2}", compact))


def _looks_like_name(value: str) -> bool:
    cleaned = re.sub(r"[^A-Z ]+", "", value.upper()).strip()
    if not cleaned:
        return False
    return cleaned not in _NAME_STOPWORDS


def _merge_field_results(structured: dict[str, str], heuristic: dict[str, str]) -> dict[str, str]:
    """把结构化抽取结果和规则解析结果合并，优先保留更可信的字段。"""
    result = {field: "" for field in _FIELD_ORDER}

    structured_id = re.sub(r"[\s-]+", "", structured.get("id_number", "")).upper()
    heuristic_id = re.sub(r"[\s-]+", "", heuristic.get("id_number", "")).upper()
    if _looks_like_valid_id(structured_id):
        result["id_number"] = structured_id
    elif _looks_like_valid_id(heuristic_id):
        result["id_number"] = heuristic_id

    for field in ("surname", "first_name"):
        s_val = structured.get(field, "").strip().upper()
        h_val = heuristic.get(field, "").strip().upper()
        if _looks_like_name(s_val):
            result[field] = s_val
        elif _looks_like_name(h_val):
            result[field] = h_val
        else:
            result[field] = s_val or h_val

    structured_birth = structured.get("birth_date", "").strip()
    heuristic_birth = heuristic.get("birth_date", "").strip()
    result["birth_date"] = _normalize_birth_date(structured_birth) if structured_birth else _normalize_birth_date(heuristic_birth)

    structured_gender = _normalize_gender(structured.get("gender", ""))
    heuristic_gender = _normalize_gender(heuristic.get("gender", ""))
    # 性别优先采用规则通道的显式 M/F（例如 `CIT F`），结构化结果仅在规则为空时兜底。
    result["gender"] = heuristic_gender if heuristic_gender in {"male", "female"} else structured_gender

    return result


def _needs_structured_fallback(heuristic: dict[str, str]) -> bool:
    """判断是否需要调用结构化兜底。

    目标：OCR 已能识别时，尽量不让结构化输出“猜测”覆盖规则结果；
    仅在关键字段缺失/明显不完整时才调用结构化。
    """
    hid = re.sub(r"[\s-]+", "", heuristic.get("id_number", "").upper())
    if not _looks_like_valid_id(hid):
        return True
    if not heuristic.get("surname") or not heuristic.get("first_name"):
        return True
    if not heuristic.get("birth_date"):
        return True
    if _normalize_gender(heuristic.get("gender", "")) not in {"male", "female"}:
        return True
    return False


def main() -> None:
    # ── 前置检查 ──────────────────────────────────────────────
    if not IMAGE_DIR.exists():
        print(f"[错误] 图片目录不存在: {IMAGE_DIR}")
        return
    if not IMAGE_DIR.is_dir():
        print(f"[错误] 路径不是目录: {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 收集图片 ──────────────────────────────────────────────
    valid_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )

    if not images:
        print(f"[警告] 图片目录中未找到 jpg/jpeg/png 文件: {IMAGE_DIR}")
        return

    if MAX_COUNT > 0:
        images = images[:MAX_COUNT]

    total = len(images)
    width = len(str(total))  # 进度编号对齐宽度

    print(f"图片目录 : {IMAGE_DIR}")
    print(f"输出目录 : {OUTPUT_DIR}")
    print(f"结构目录 : {DEAL_DIR}")
    print(f"处理数量 : {total} 张  (MAX_COUNT={MAX_COUNT or '不限'})\n")

    success_count = 0
    skip_ocr_count = 0
    fail_count    = 0
    fail_list: list[tuple[str, str]] = []  # (文件名, 错误信息)

    # ── 逐张处理 ──────────────────────────────────────────────
    for idx, image_path in enumerate(images, 1):
        prefix = f"[{idx:{width}d}/{total}]"
        output_path = OUTPUT_DIR / (image_path.stem + ".txt")
        deal_path = DEAL_DIR / (image_path.stem + ".txt")

        # raw OCR 已存在时不再请求接口，但仍会重建结构化 deal 输出，方便修复后直接复跑。
        if output_path.exists():
            print(f"{prefix} 处理(跳过OCR): {image_path.name}", end="", flush=True)
            skip_ocr_count += 1
        else:
            print(f"{prefix} 处理: {image_path.name}", end="", flush=True)

        try:
            if output_path.exists():
                raw_result = output_path.read_text(encoding="utf-8")
            else:
                raw_result = ocr_with_retry(image_path)
                output_path.write_text(raw_result, encoding="utf-8")

            heuristic_fields = _parse_structured_fields(raw_result)
            structured_fields = None
            if _needs_structured_fallback(heuristic_fields):
                try:
                    structured_text = _call_structured_api(image_path, raw_result)
                    structured_fields = _extract_json_payload(structured_text)
                except Exception:
                    structured_fields = None

            if structured_fields is None:
                fields = heuristic_fields
            else:
                fields = _merge_field_results(structured_fields, heuristic_fields)

            deal_path.write_text(_build_json_output(fields), encoding="utf-8")
            print(f"  → 已保存: {output_path.name} / {deal_path.name}")
            success_count += 1
        except Exception as exc:
            err_msg = str(exc)
            print(f"  → 失败: {err_msg}")
            fail_count += 1
            fail_list.append((image_path.name, err_msg))

    # ── 汇总报告 ──────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"处理完成  成功: {success_count}  跳过OCR: {skip_ocr_count}  失败: {fail_count}  共: {total}")
    if fail_list:
        print(f"\n失败文件列表（共 {len(fail_list)} 个）：")
        for name, err in fail_list:
            print(f"  ✗ {name}")
            print(f"      {err}")


if __name__ == "__main__":
    main()
