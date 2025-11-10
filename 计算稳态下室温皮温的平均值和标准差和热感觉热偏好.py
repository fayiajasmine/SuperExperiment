# -*- coding: utf-8 -*-
"""
从 task_stable_windows_mapped.csv 的每个稳态时间段，在对应 Excel 中抽取：
  1) temp / back_temp℃ / hand_temp℃ / leg_temp℃ / neck_temp℃ / mTSK 的均值与标准差
  2) 主观投票：热感觉 / 热感觉_数值 / 热舒适 / 热舒适_数值 / 温度偏好 / 温度偏好_数值
规则：
  - 信号列：窗口内做均值/标准差
  - 投票列：若窗口内命中多条，复制稳态行多份；若未命中，投票为 NaN
输出：task_stable_windows_mapped_with_tempstats_plus_signals_votes.csv
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ===== 路径配置 =====
EXCEL_DIR = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing")
CSV_IN    = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25.csv")
CSV_OUT   = CSV_IN.with_name(CSV_IN.stem + "_with_tempstats_plus_signals_votes.csv")

# ===== 配置：Excel中的时间列 与 需要的列 =====
TIME_COL = "时间"  # Excel里时间的中文列名（精确匹配）

# 需要计算均值/标准差的信号列（Excel中需为这些精确列名；temp 需精确为 'temp'）
SIGNAL_COLS = ["temp", "back_temp℃", "hand_temp℃", "leg_temp℃", "neck_temp℃", "mTSK"]

# 需要抽取的主观投票列（Excel存在则取；缺失则自动补 NaN）
VOTE_COLS = ["热感觉", "热感觉_数值", "热舒适", "热舒适_数值", "温度偏好", "温度偏好_数值"]

# ===== 小工具 =====
def read_csv_guess(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")

def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()

def find_time_col(df: pd.DataFrame) -> str | None:
    # 允许“时间”或以time开头的英文
    for c in df.columns:
        n = norm(c)
        if ("时间" in n) or (n == "time") or n.startswith("time"):
            return c
    return None

def have_exact_col(df: pd.DataFrame, colname: str) -> bool:
    # 精确匹配（忽略大小写/空格）
    for c in df.columns:
        if norm(c) == norm(colname):
            return True
    return False

def get_exact_colname(df: pd.DataFrame, colname: str) -> str | None:
    for c in df.columns:
        if norm(c) == norm(colname):
            return c
    return None

def to_datetime_series(s: pd.Series) -> pd.Series:
    """尽量把文本/数字时间转为 pandas datetime"""
    ts = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if ts.notna().mean() >= 0.6:
        return ts
    # 尝试 Excel/Unix 数字
    sn = pd.to_numeric(s, errors="coerce")
    ts2 = pd.Series(pd.NaT, index=s.index)
    # Excel 天序列
    m_excel = sn.notna() & sn.between(20000, 80000)
    if m_excel.any():
        ts2.loc[m_excel] = pd.to_datetime(sn.loc[m_excel], unit="D", origin="1899-12-30", errors="coerce")
    # Unix 秒/毫秒
    m_unix = sn.notna() & ~m_excel
    if m_unix.any():
        cand = pd.to_datetime(sn.loc[m_unix], unit="s", origin="unix", errors="coerce")
        if cand.notna().mean() > 0.5:
            ts2.loc[m_unix] = cand
        else:
            ts2.loc[m_unix] = pd.to_datetime(sn.loc[m_unix], unit="ms", origin="unix", errors="coerce")
    return ts.where(ts.notna(), ts2)

def seconds_of_day(dt: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dt, errors="coerce")
    sec = (
        dt.dt.hour.fillna(0).astype(int) * 3600
        + dt.dt.minute.fillna(0).astype(int) * 60
        + dt.dt.second.fillna(0).astype(int)
    ).astype(float)
    if hasattr(dt.dt, "microsecond"):
        sec = sec + dt.dt.microsecond.fillna(0).astype(float) / 1e6
    return sec.values

def parse_hhmmss_to_sec(s: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    return seconds_of_day(dt)

def window_mask(t_sec: np.ndarray, s0: float | np.ndarray, e0: float | np.ndarray) -> np.ndarray:
    """生成窗口掩码，支持跨午夜（end < start）"""
    s_ok = np.isfinite(s0)
    e_ok = np.isfinite(e0)
    if s_ok and e_ok:
        s, e = float(s0), float(e0)
        if e >= s:
            return (t_sec >= s) & (t_sec <= e)
        else:
            return (t_sec >= s) | (t_sec <= e)
    elif s_ok:
        s = float(s0)
        return (t_sec >= s)
    elif e_ok:
        e = float(e0)
        return (t_sec <= e)
    else:
        return np.zeros_like(t_sec, dtype=bool)

# ===== 读取窗口 CSV =====
marks = read_csv_guess(CSV_IN)
marks.columns = [str(c).strip() for c in marks.columns]

# 必需列检查
need_cols = ["file", "start_time", "end_time"]
miss = [c for c in need_cols if c not in marks.columns]
if miss:
    raise ValueError(f"CSV 缺少列：{miss}")

# 预解析窗口时间为“当天秒数”
marks["start_sec"] = parse_hhmmss_to_sec(marks["start_time"])
marks["end_sec"]   = parse_hhmmss_to_sec(marks["end_time"])

# ===== 缓存每个 Excel 的数据（时间秒、信号、投票、原始时间） =====
cache: dict[str, pd.DataFrame] = {}

def load_excel_df(xfname: str) -> pd.DataFrame | None:
    """返回一个 DataFrame，至少包含列：t_sec, TIME_COL(原始时间)，以及存在的 SIGNAL_COLS/VOTE_COLS"""
    if xfname in cache:
        return cache[xfname]

    path = EXCEL_DIR / xfname
    if not path.exists():
        print(f"[WARN] 未找到 Excel：{xfname}")
        cache[xfname] = None
        return None

    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[WARN] 打不开 {xfname} | {e}")
        cache[xfname] = None
        return None

    for sh in xls.sheet_names:
        try:
            df = xls.parse(sh)
        except Exception:
            continue
        if df is None or df.empty:
            continue

        # 找时间列
        time_col = get_exact_colname(df, TIME_COL) or find_time_col(df)
        if not time_col:
            continue

        # 收集存在的列
        keep_cols = [time_col]
        # 信号列：精确名（允许大小写空格差异）
        sig_cols_avail = []
        for sc in SIGNAL_COLS:
            c_real = get_exact_colname(df, sc)
            if c_real:
                keep_cols.append(c_real)
                sig_cols_avail.append((sc, c_real))

        # 投票列
        vote_cols_avail = []
        for vc in VOTE_COLS:
            c_real = get_exact_colname(df, vc)
            if c_real:
                keep_cols.append(c_real)
                vote_cols_avail.append((vc, c_real))

        sub = df[keep_cols].copy()

        # 解析时间
        sub["_dt"] = to_datetime_series(sub[time_col])
        sub = sub.dropna(subset=["_dt"])
        if sub.empty:
            continue
        sub = sub.sort_values("_dt")

        # 数值化信号与“*_数值”的投票列
        for sc, real in sig_cols_avail:
            sub[real] = pd.to_numeric(sub[real], errors="coerce")
        for vc, real in vote_cols_avail:
            if vc.endswith("数值"):
                sub[real] = pd.to_numeric(sub[real], errors="coerce")
            else:
                sub[real] = sub[real].astype(str).replace({"nan": np.nan, "": np.nan})

        # 当天秒
        sub["t_sec"] = seconds_of_day(sub["_dt"])

        # 标准化列名：保留一份 TIME_COL 原名，信号/投票统一映射到标准名
        out = pd.DataFrame({"t_sec": sub["t_sec"].values, TIME_COL: sub["_dt"].values})

        for sc, real in sig_cols_avail:
            out[sc] = sub[real].values
        for vc, real in vote_cols_avail:
            out[vc] = sub[real].values

        cache[xfname] = out.reset_index(drop=True)
        return cache[xfname]

    print(f"[WARN] {xfname} 找不到包含‘{TIME_COL}’的有效工作表")
    cache[xfname] = None
    return None

# ===== 主循环：为每个窗口计算信号统计 + 复制行承载投票 =====
out_rows = []

for idx, r in marks.iterrows():
    fname = str(r["file"]).strip()
    df = load_excel_df(fname)

    # 先构造“基础行”（不含投票）
    base = r.to_dict()

    # 预填充所有信号的 mean/std
    for sc in SIGNAL_COLS:
        base[f"{sc}_mean"] = np.nan
        base[f"{sc}_std"]  = np.nan

    # 预填充投票列
    for vc in VOTE_COLS:
        base[vc] = np.nan
    base["vote_time"] = np.nan  # 可选：把命中的投票时间也带上

    if df is None or df.empty:
        # 没有这位被试的数据，直接输出空投票
        out_rows.append(base)
        continue

    # 窗口掩码
    m = window_mask(df["t_sec"].values, r.get("start_sec", np.nan), r.get("end_sec", np.nan))
    seg = df.loc[m]

    # 1) 信号均值/标准差（对存在的列做）
    if not seg.empty:
        for sc in SIGNAL_COLS:
            if sc in seg.columns:
                vals = pd.to_numeric(seg[sc], errors="coerce")
                if vals.notna().any():
                    base[f"{sc}_mean"] = float(vals.mean(skipna=True))
                    base[f"{sc}_std"]  = float(vals.std(skipna=True, ddof=0))

    # 2) 投票抽取（任意一列非空即视为命中）
    vote_avail = [vc for vc in VOTE_COLS if vc in seg.columns]
    if vote_avail:
        sub_votes = seg[vote_avail + [TIME_COL]].copy()
        if not sub_votes.empty:
            mask_any = sub_votes[vote_avail].notna().any(axis=1)
            sub_votes = sub_votes[mask_any]
        else:
            sub_votes = pd.DataFrame(columns=vote_avail + [TIME_COL])
    else:
        sub_votes = pd.DataFrame(columns=[TIME_COL])

    if sub_votes.empty:
        # 无投票：输出 1 行（均值/标准差已在 base 中）
        out_rows.append(base)
    else:
        # 有多条投票：复制输出
        for _, vr in sub_votes.iterrows():
            row = base.copy()
            for vc in VOTE_COLS:
                row[vc] = vr.get(vc, np.nan)
            row["vote_time"] = vr.get(TIME_COL, np.nan)
            out_rows.append(row)

# ===== 导出 =====
out_df = pd.DataFrame(out_rows)
out_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print(f"[DONE] 已写出：{CSV_OUT}  | 共 {len(out_df)} 行")
