"""evaluate-for-json.py

对比：
  - 预测：文件夹下的 .json 文件
  - 标签：文件夹下的 .txt 文件（内容是JSON）
输出：
  - 汇总报告 (.txt)
  - 全部错误样本明细 (.csv)
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ══════════════════════════════════════════════════════════════
# 配置（修改这里即可）
# ══════════════════════════════════════════════════════════════
PRED_DIR   = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-MAIN-6-DEAL-OUT")
LABEL_DIR  = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\金属ID-JSON")
REPORT_DIR = Path(r"D:\aaa_my_iwhalecloud\VScode_AI\HunyuanOCR\津巴布韦API开发测试\WAY2\report")

# 输出文件名
REPORT_TXT = "main-6-eval_pred_vs_label_json.txt"
ERROR_CSV  = "main-6-errors.csv"
# ══════════════════════════════════════════════════════════════

FIELDS = ("id_number", "surname", "first_name", "birth_date", "gender")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _try_parse_json(text: str) -> dict[str, Any] | None:
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_generic(value: Any) -> str:
    s = _as_str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


def _norm_id_number(value: Any) -> str:
    s = _as_str(value)
    s = re.sub(r"\s+", "", s)
    return s.upper()


def _norm_gender(value: Any) -> str:
    s = _as_str(value).strip()
    if not s:
        return ""
    upper = s.upper()
    if re.search(r"\bF\b", upper) or "FEMALE" in upper:
        return "FEMALE"
    if re.search(r"\bM\b", upper) or "MALE" in upper:
        return "MALE"
    return upper


@dataclass(frozen=True)
class _ParsedDate:
    day: int
    month: int
    year: int

    @property
    def year2(self) -> int:
        return self.year % 100


def _parse_birth_date(value: Any) -> _ParsedDate | None:
    s = _as_str(value)
    if not s:
        return None
    nums = re.findall(r"\d+", s)
    if len(nums) >= 3:
        try:
            day, month, year = int(nums[0]), int(nums[1]), int(nums[2])
            if 1 <= day <= 31 and 1 <= month <= 12 and year >= 0:
                return _ParsedDate(day=day, month=month, year=year)
        except Exception:
            pass
    digits = re.sub(r"\D", "", s)
    if len(digits) in (6, 8):
        try:
            day, month, year = int(digits[0:2]), int(digits[2:4]), int(digits[4:])
            if 1 <= day <= 31 and 1 <= month <= 12 and year >= 0:
                return _ParsedDate(day=day, month=month, year=year)
        except Exception:
            pass
    return None


def _birth_date_equal(pred: Any, label: Any) -> bool:
    p = _parse_birth_date(pred)
    l = _parse_birth_date(label)
    if p is None or l is None:
        return _norm_generic(pred) == _norm_generic(label)
    if p.day != l.day or p.month != l.month:
        return False
    return p.year == l.year or p.year2 == l.year2


def _field_equal(field: str, pred_val: Any, label_val: Any) -> bool:
    if field == "birth_date":
        return _birth_date_equal(pred_val, label_val)
    if field == "gender":
        return _norm_gender(pred_val) == _norm_gender(label_val)
    if field == "id_number":
        return _norm_id_number(pred_val) == _norm_id_number(label_val)
    return _norm_generic(pred_val) == _norm_generic(label_val)


@dataclass
class ErrorRecord:
    file_name: str
    field: str
    label_raw: str
    pred_raw: str
    label_norm: str
    pred_norm: str


def evaluate_json_dirs(
    pred_dir: Path,
    label_dir: Path,
    out_txt_path: Path,
    out_csv_path: Path,
) -> tuple[Path, Path]:
    if not label_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"预测目录不存在: {pred_dir}")

    # ====================== 修改点 1：遍历标签 .txt ======================
    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise RuntimeError(f"标签目录下没有 .txt: {label_dir}")

    correct: dict[str, int] = {f: 0 for f in FIELDS}
    total = 0
    missing_pred = 0
    parse_err_pred = 0
    parse_err_label = 0
    errors: list[ErrorRecord] = []

    for label_path in label_files:
        # 读取标签
        label_obj = _try_parse_json(_read_text(label_path))
        if label_obj is None:
            parse_err_label += 1
            continue

        total += 1

        # ====================== 修改点 2：预测文件改为 .json ======================
        pred_path = pred_dir / f"{label_path.stem}.json"

        pred_obj: dict[str, Any] | None
        if not pred_path.exists():
            missing_pred += 1
            pred_obj = None
        else:
            # 直接读取 json
            try:
                pred_obj = json.loads(_read_text(pred_path))
            except:
                pred_obj = None

            if pred_obj is None:
                parse_err_pred += 1

        # 字段对比
        for field in FIELDS:
            label_val = label_obj.get(field, "")
            pred_val = "" if pred_obj is None else pred_obj.get(field, "")
            if _field_equal(field, pred_val, label_val):
                correct[field] += 1
            else:
                if field == "birth_date":
                    label_n = _as_str(label_val)
                    pred_n = _as_str(pred_val)
                elif field == "gender":
                    label_n = _norm_gender(label_val)
                    pred_n = _norm_gender(pred_val)
                elif field == "id_number":
                    label_n = _norm_id_number(label_val)
                    pred_n = _norm_id_number(pred_val)
                else:
                    label_n = _norm_generic(label_val)
                    pred_n = _norm_generic(pred_val)

                errors.append(ErrorRecord(
                    file_name=label_path.stem,
                    field=field,
                    label_raw=_as_str(label_val),
                    pred_raw=_as_str(pred_val),
                    label_norm=label_n,
                    pred_norm=pred_n,
                ))

    if total == 0:
        raise RuntimeError("没有可评估的样本")

    acc = {f: correct[f] / total for f in FIELDS}
    overall = sum(correct.values()) / (total * len(FIELDS))

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"预测目录: {pred_dir} (读取 .json)",
        f"标签目录: {label_dir} (读取 .txt)",
        f"样本数(可解析标签): {total}",
        f"缺失预测文件数: {missing_pred}",
        f"预测 JSON 解析失败数: {parse_err_pred}",
        f"标签 JSON 解析失败数(未计入样本): {parse_err_label}",
        "",
        "五字段准确率:",
    ]
    for f in FIELDS:
        lines.append(f"- {f}: {acc[f]*100:.2f}% ({correct[f]}/{total})")
    lines.append("")
    lines.append(f"总体平均准确率(按字段微平均): {overall*100:.2f}% "
                 f"({sum(correct.values())}/{total * len(FIELDS)})")
    lines.append("")
    lines.append(f"错误总数: {len(errors)}")
    out_txt_path.write_text("\n".join(lines), encoding="utf-8")

    csv_headers = [
        "file_name", "field",
        "label_raw", "pred_raw",
        "label_norm", "pred_norm",
    ]
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for e in errors:
            writer.writerow([
                e.file_name, e.field,
                e.label_raw, e.pred_raw,
                e.label_norm, e.pred_norm,
            ])

    return out_txt_path, out_csv_path


def main() -> None:
    txt_path = REPORT_DIR / REPORT_TXT
    csv_path = REPORT_DIR / ERROR_CSV
    r1, r2 = evaluate_json_dirs(PRED_DIR, LABEL_DIR, txt_path, csv_path)
    print(f"已生成汇总报告: {r1}")
    print(f"已生成错误CSV : {r2}")
    print(f"错误条目数: {sum(1 for _ in open(r2, encoding='utf-8-sig')) - 1}")


if __name__ == "__main__":
    main()