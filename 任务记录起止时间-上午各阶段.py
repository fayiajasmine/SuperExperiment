# -*- coding: utf-8 -*-
"""
从 25 个“任务记录”Excel中提取红框所示内容：
  - “完成稳态1/2/3/4”
  - 若存在则额外提取 “维持25度室温”
兼容两种版式（图1、图2）。

输出：task_stable_windows.csv
    columns = [file, stage, task_text, start_time, end_time, duration_min]
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

# ========= 配置：修改为你的任务记录文件夹 =========
FOLDER = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录"   # TODO: 改成你的路径
OUT_CSV = Path(FOLDER) / "task_stable_windows.csv"
# =================================================

# ---------- 小工具 ----------
def norm(s: str) -> str:
    """列名/文本规范化：去空白、降小写、全角半角统一（简单处理）"""
    if s is None:
        return ""
    s = str(s)
    s = s.strip()
    # 全角数字/空格简单归一
    trans = str.maketrans("０１２３４５６７８９：；，．－（）　",
                          "0123456789:;,.-() ")
    s = s.translate(trans)
    return s.lower()

def find_header_row(df: pd.DataFrame) -> int | None:
    """
    在无固定表头行的情况下，向下扫描前 20 行，
    找到同时包含“任务”和“开始/完成”列名的行，作为表头。
    """
    max_scan = min(20, len(df))
    for i in range(max_scan):
        row_vals = [norm(x) for x in df.iloc[i].tolist()]
        row_text = " ".join(row_vals)
        # 需要至少出现开始/完成中的一个关键词
        has_task = any(k in row_text for k in ["任务", "task"])
        has_start = any(k in row_text for k in ["开始", "起始", "start"])
        has_end   = any(k in row_text for k in ["完成", "结束", "end", "完成时间"])
        # 允许有的表没有明确“完成状态”列
        if has_task and (has_start or has_end):
            return i
    return None

def pick_cols(columns: list[str]) -> dict:
    """
    从列名中挑出 任务列 / 开始时间列 / 结束时间列
    兼容：
      - 图1：列名为 [任务, 开始时间, 完成时间]
      - 图2：列名为 [预计时间, 任务, 完成状态, 开始时间, 完成时间]
    """
    task_col, start_col, end_col = None, None, None

    # 优先：通过中文关键词严格匹配
    for c in columns:
        n = norm(c)
        if task_col is None and ("任务" in n or n.startswith("task")):
            task_col = c
        if start_col is None and any(k in n for k in ["开始时间", "起始时间", "start_time", "开始"]):
            start_col = c
        if end_col is None and any(k in n for k in ["完成时间", "结束时间", "end_time", "完成"]):
            end_col = c

    # 兜底策略：若未匹配，按常见列序列猜测（第1列任务，第2列开始，第3列结束）
    if task_col is None and len(columns) >= 1:
        task_col = columns[0]
    if start_col is None and len(columns) >= 2:
        start_col = columns[1]
    if end_col is None and len(columns) >= 3:
        end_col = columns[2]

    # ✅ 新增：如果同时出现“预计时间”，则说明是图2样式，往后偏移两列
    for c in columns:
        if "预计时间" in norm(c):
            idx = columns.index(c)
            if len(columns) >= idx + 5:
                task_col = columns[idx + 1]
                start_col = columns[idx + 3]
                end_col = columns[idx + 4]
            break

    return {"task": task_col, "start": start_col, "end": end_col}


def parse_time_series(s: pd.Series) -> pd.Series:
    """
    宽容解析任务记录表里的时间列，返回带“日期占位”的 pandas datetime64。
    兼容：
      - "10:07" / "10:07:00"（含全角冒号）
      - 已是 datetime / time
      - Excel 小数时间（0~1 浮点表示一天中的分数）
    """
    # 1) 先尝试直接按 datetime 解析（可吃掉真正的 datetime/time 字段）
    parsed = pd.to_datetime(s, errors="coerce")

    # 2) 对解析失败的部分按“字符串清洗 + 格式化”再试一次
    mask_need_str = parsed.isna()
    if mask_need_str.any():
        s_str = s[mask_need_str].map(lambda x: "" if pd.isna(x) else str(x))
        # 规范化：去空白、全角转半角、小写
        def _norm_one(x: str) -> str:
            trans = str.maketrans("０１２３４５６７８９：；，．－（）　",
                                  "0123456789:;,.-() ")
            x = x.strip().translate(trans).lower()
            return x
        s_str = s_str.map(_norm_one)

        # 把明显的占位符变成 NA（避免 .str 报错；且不会触发 FutureWarning）
        bad_tokens = {"", "nan", "/", "\\", "？", "?"}
        s_str = s_str.where(~s_str.isin(bad_tokens), pd.NA).astype("string")

        # "HH:MM" → "HH:MM:00"
        s_str = s_str.str.replace(r"^(\d{1,2})[:：](\d{2})$", r"\1:\2:00", regex=True)

        parsed2 = pd.to_datetime(s_str, format="%H:%M:%S", errors="coerce")
        parsed.loc[mask_need_str] = parsed2

    # 3) 仍未解析的，尝试 Excel 小数时间（0~1 浮点）
    mask_float = parsed.isna() & s.apply(lambda x: isinstance(x, (int, float)))
    if mask_float.any():
        vals = pd.to_numeric(s[mask_float], errors="coerce")
        # 合法：0 <= v < 1
        ok = vals.notna() & (vals >= 0) & (vals < 1)
        if ok.any():
            base = pd.Timestamp("2000-01-01")  # 任意占位日期
            td = pd.to_timedelta(vals[ok], unit="D")
            parsed.loc[mask_float[mask_float].index[ok]] = base + td

    return parsed


def detect_stage(task_text: str) -> tuple[str | None, bool]:
    """
    从任务文本里识别：
        - 完成稳态1/2/3/4  -> 返回 ("steady1/2/3/4", True)
        - 维持25度室温     -> 返回 ("hold25", True)
        - 其他             -> (None, False)
    """
    t = norm(task_text)
    # 完成稳态1~4
    m = re.search(r"完成\s*稳态\s*([1-4])", t)
    if m:
        return (f"steady{m.group(1)}", True)
    # 维持25度室温
    if ("维持" in t and "25" in t and ("室温" in t or "度室温" in t)) or ("保持25度" in t):
        return ("hold25", True)
    return (None, False)

def extract_from_excel(path: Path) -> list[dict]:
    """
    从单个 Excel 文件（取第一个 sheet）提取所需行。
    """
    out_rows = []
    try:
        # 读原始，不设表头先
        raw = pd.read_excel(path, header=None, dtype=object)
    except Exception as e:
        print(f"[SKIP] 无法读取 {path.name} | {e}")
        return out_rows

    if raw.empty:
        print(f"[SKIP] 空文件 {path.name}")
        return out_rows

    # 找表头行
    hdr_row = find_header_row(raw)
    if hdr_row is None:
        # 兜底：就假设第 0 行
        hdr_row = 0

    df = pd.read_excel(path, header=hdr_row, dtype=object)
    # 去掉全空列
    df = df.dropna(axis=1, how="all")
    if df.empty:
        print(f"[SKIP] {path.name} 表数据为空")
        return out_rows

    colmap = pick_cols(list(df.columns))
    task_c, start_c, end_c = colmap["task"], colmap["start"], colmap["end"]

    # 列存在性检查
    for need, c in [("任务列", task_c), ("开始时间列", start_c), ("结束时间列", end_c)]:
        if c not in df.columns:
            print(f"[WARN] {path.name} 未找到 {need}，列名猜测结果={colmap}。尝试继续（可能提取不到）。")

    # 只保留这三列，容忍缺失
    keep = [x for x in [task_c, start_c, end_c] if x in df.columns]
    df = df[keep].copy()

    # 统一列名
    rename_map = {}
    if task_c in df.columns:  rename_map[task_c]  = "task"
    if start_c in df.columns: rename_map[start_c] = "start_time"
    if end_c in df.columns:   rename_map[end_c]   = "end_time"
    df = df.rename(columns=rename_map)

    # 文本列
    if "task" not in df.columns:
        print(f"[SKIP] {path.name} 无任务列，无法识别‘完成稳态*’等文本。")
        return out_rows

    # 解析时间
    if "start_time" in df.columns:
        df["start_time_parsed"] = parse_time_series(df["start_time"])
    else:
        df["start_time_parsed"] = pd.NaT

    if "end_time" in df.columns:
        df["end_time_parsed"] = parse_time_series(df["end_time"])
    else:
        df["end_time_parsed"] = pd.NaT

    # 逐行识别
    for _, r in df.iterrows():
        task_text = str(r.get("task", ""))
        stage, ok = detect_stage(task_text)
        if not ok:
            continue  # 只要红框内容

        st = r.get("start_time_parsed", pd.NaT)
        et = r.get("end_time_parsed", pd.NaT)

        # 计算分钟（若两者都非空）
        if pd.notna(st) and pd.notna(et):
            dur_min = (et - st).total_seconds() / 60.0
        else:
            dur_min = np.nan

        out_rows.append({
            "file": path.name,
            "stage": stage,               # steady1~4 / hold25
            "task_text": task_text,       # 原文
            "start_time": st.strftime("%H:%M:%S") if pd.notna(st) else "",
            "end_time":   et.strftime("%H:%M:%S") if pd.notna(et) else "",
            "duration_min": round(dur_min, 1) if isinstance(dur_min, (int, float)) and not np.isnan(dur_min) else ""
        })

    return out_rows


def main():
    files = []
    for ext in ("*.xlsx", "*.xls"):
        files.extend(Path(FOLDER).glob(ext))
    files = sorted(files, key=lambda p: p.name)

    all_rows = []
    for f in files:
        rows = extract_from_excel(f)
        if rows:
            all_rows.extend(rows)
        else:
            print(f"[INFO] {f.name} 未提取到红框目标行。")

    if not all_rows:
        print("[DONE] 没有提取到任何结果。请检查文件路径与示例格式。")
        return

    out = pd.DataFrame(all_rows)
    # 同一文件优先按稳态阶段序排序：steady1→steady2→steady3→steady4→hold25
    stage_order = {"steady1":1, "steady2":2, "steady3":3, "steady4":4, "hold25":5}
    out["__order__"] = out["stage"].map(stage_order).fillna(99)
    out = out.sort_values(["file", "__order__", "start_time"]).drop(columns="__order__")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[DONE] 已输出：{OUT_CSV}  | 共 {len(out)} 行")

if __name__ == "__main__":
    main()
