"""
判断 金属ID-JSON（目标字段）中的内容能否从 金属ID-MAIN-OUT（原始 OCR 坐标文本）中提取得到。

输出：
  - 每个字段的可提取率
  - 每个文件的完整度（5 字段全可提取 vs 缺失哪些）
  - 不可提取的示例
"""

import json
import re
from collections import defaultdict
from pathlib import Path

JSON_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-JSON")
OCR_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-OUT")

# ── 正则 ──────────────────────────────────────────────────────
_SEGMENT_RE = re.compile(r"(.*?)\((\d+),(\d+)\),\((\d+),(\d+)\)", re.DOTALL)
_CIT_MF_RE  = re.compile(r'\b(CIT|GIT|C1T)\s*([MF])\b', re.IGNORECASE)
_DATE_RAW_RE = re.compile(r"(\d{1,2})[\s./\-](\d{1,2})[\s./\-](\d{2,4})")


def _extract_segment_texts(raw_text: str) -> list[str]:
    """从 OCR 坐标文本中提取所有纯文字段（不含坐标）"""
    texts = []
    for match in _SEGMENT_RE.finditer(raw_text):
        t = match.group(1).strip()
        if t:
            texts.append(t)
    return texts


def _check_id(texts: list[str], target_id: str) -> bool:
    """检查 id_number 是否出现在 OCR 段中（忽略空格/连字符）"""
    target_compact = re.sub(r'[^A-Z0-9]', '', target_id.upper())
    for t in texts:
        t_compact = re.sub(r'[^A-Z0-9]', '', t.upper())
        if target_compact in t_compact or t_compact in target_compact:
            return True
    return False


def _check_name(texts: list[str], target_name: str) -> bool:
    """检查姓名是否在 OCR 段中出现（允许片段匹配）"""
    if not target_name:
        return False
    target_upper = target_name.upper().strip()
    # 拆词逐一检查（支持多词名如 GOLDEN GEORGE）
    tokens = target_upper.split()
    for token in tokens:
        found = False
        for t in texts:
            t_clean = re.sub(r'[^A-Z]', '', t.upper())
            token_clean = re.sub(r'[^A-Z]', '', token)
            if token_clean and token_clean in t_clean:
                found = True
                break
        if not found:
            return False
    return True


def _check_date(texts: list[str], target_date: str) -> bool:
    """检查出生日期是否在 OCR 段中出现（匹配日/月/年数字）"""
    if not target_date:
        return False
    # 目标日期格式: DD.MM.YYYY
    parts = target_date.split(".")
    if len(parts) != 3:
        return False
    target_d = parts[0].zfill(2)
    target_m = parts[1].zfill(2)
    target_y = parts[2]
    # 统一转 4 位年
    if len(target_y) == 2:
        target_y4 = "19" + target_y if int(target_y) > 30 else "20" + target_y
    else:
        target_y4 = target_y

    for t in texts:
        m = _DATE_RAW_RE.search(t)
        if not m:
            continue
        try:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d_str = str(d).zfill(2)
            mo_str = str(mo).zfill(2)
            if len(str(y)) == 2:
                y = 1900 + y if y > 30 else 2000 + y
            y_str = str(y)
            if d_str == target_d and mo_str == target_m and y_str == target_y4:
                return True
        except ValueError:
            continue
    return False


def _check_gender(texts: list[str], target_gender: str) -> bool:
    """检查 OCR 段中是否有 CIT/GIT + M/F 标记能推断出性别"""
    target = target_gender.lower().strip()
    if target not in ("male", "female"):
        return False

    expected_letter = "M" if target == "male" else "F"

    # 检查每个段
    for t in texts:
        m = _CIT_MF_RE.search(t)
        if m:
            if m.group(2).upper() == expected_letter:
                return True

    # 检查拼接文本（可能 CIT 和 M/F 在不同段）
    joined = " ".join(texts).upper()
    for m in _CIT_MF_RE.finditer(joined):
        if m.group(2).upper() == expected_letter:
            return True

    return False


def main():
    json_files = sorted(p for p in JSON_DIR.iterdir() if p.suffix.lower() == ".txt")
    total = len(json_files)
    print(f"目标 JSON 文件数: {total}")
    print(f"OCR 原始文件数 : {len(list(OCR_DIR.iterdir()))}\n")

    # 统计
    field_ok: dict[str, int] = defaultdict(int)       # 各字段可提取计数
    field_fail: dict[str, list[str]] = defaultdict(list)  # 各字段失败的文件名（最多存 10 个示例）
    file_completeness: dict[int, int] = defaultdict(int)  # 完整度分布
    all_ok = 0
    all_fail_files: list[str] = []

    for json_p in json_files:
        stem = json_p.stem
        ocr_p = OCR_DIR / f"{stem}.txt"

        with open(json_p, "r", encoding="utf-8") as f:
            target = json.load(f)

        if not ocr_p.exists():
            for field in ["id_number", "surname", "first_name", "birth_date", "gender"]:
                field_fail[field].append(f"{stem} (OCR文件缺失)")
            file_completeness[0] += 1
            all_fail_files.append(stem)
            continue

        raw_ocr = ocr_p.read_text(encoding="utf-8").strip()
        texts = _extract_segment_texts(raw_ocr)

        checks = {
            "id_number":  _check_id(texts, target.get("id_number", "")),
            "surname":    _check_name(texts, target.get("surname", "")),
            "first_name": _check_name(texts, target.get("first_name", "")),
            "birth_date": _check_date(texts, target.get("birth_date", "")),
            "gender":     _check_gender(texts, target.get("gender", "")),
        }

        ok_count = sum(1 for v in checks.values() if v)
        file_completeness[ok_count] += 1

        for field, ok in checks.items():
            if ok:
                field_ok[field] += 1
            elif len(field_fail[field]) < 10:
                field_fail[field].append(stem)

        if ok_count == 5:
            all_ok += 1
        else:
            all_fail_files.append(stem)

    # ── 输出报告 ──────────────────────────────────────────────
    print("=" * 65)
    print(f"{'字段':<14} {'可提取':>6} {'总数':>6} {'提取率':>8}")
    print("-" * 38)
    for field in ["id_number", "surname", "first_name", "birth_date", "gender"]:
        ok = field_ok[field]
        pct = ok / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 10) + "-" * (10 - int(pct / 10))
        print(f"{field:<14} {ok:>6} {total:>6} {pct:>7.1f}%  {bar}")
    print("=" * 65)

    print(f"\n文件完整度分布（5 字段 = 全部可提取）：")
    for k in sorted(file_completeness.keys(), reverse=True):
        n = file_completeness[k]
        pct = n / total * 100
        bar = "#" * int(pct / 5) if pct > 0 else ""
        print(f"  {k}/5 字段可提取: {n:>4} 个文件 ({pct:>5.1f}%)  {bar}")
    print(f"\n全部 5 字段可提取: {all_ok}/{total} ({all_ok/total*100:.1f}%)")

    print(f"\n{'=' * 65}")
    print("各字段不可提取的文件示例（最多 10 个）：")
    for field in ["id_number", "surname", "first_name", "birth_date", "gender"]:
        fails = field_fail[field]
        if fails:
            print(f"\n  [{field}] ({len(fails)} 个失败，仅展示前 {min(10, len(fails))} 个)")
            for f_name in fails[:10]:
                # 读目标值帮助分析
                with open(JSON_DIR / f"{f_name}.txt", "r", encoding="utf-8") as f:
                    gt = json.load(f)
                val = gt.get(field, "")
                print(f"    {f_name}: 目标值=\"{val}\"")


if __name__ == "__main__":
    main()
