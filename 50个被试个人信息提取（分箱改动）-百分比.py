# -*- coding: utf-8 -*-
"""
批量读取 50 个被试 Excel，提取人口学信息到一个 CSV，并按固定分箱绘制直方/柱状图：
- 年龄：自定义分箱 (18,23), (23,28), (28,33), (33,38), (38,∞)
- 身高：5 cm 一档（内部统一换算为 m；画图用 cm）
- 体重：5 kg 一档
- BMI ：中国成人标准分箱 <18.5（偏瘦）、18.5–24（正常）、24–28（超重）、≥28（肥胖）
- 广东居住年限：2 年一档
- 性别 / 教育经历 / 籍贯 ：分类条形图

输出：
  ./participants_summary/participants.csv
  ./participants_summary/plots/<变量>_hist.png
"""

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import Optional
from matplotlib.ticker import FuncFormatter

# ================== 路径（按需修改） ==================
DATA_DIR = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\问卷\问卷处理后")
OUT_DIR  = DATA_DIR / "participants_summary5"
# =====================================================

# -------- 显示中文（自动找可用字体） --------
for fam in ["Microsoft YaHei","SimHei","PingFang SC","Heiti SC","Noto Sans CJK SC","Source Han Sans SC"]:
    if fam in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [fam]; break
plt.rcParams["axes.unicode_minus"] = False

# -------- 列名模糊识别工具 --------
def norm(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    s = s.replace("（", "(").replace("）", ")")
    return s

def pick_col(df: pd.DataFrame, keys: tuple[str, ...]) -> str|None:
    for c in df.columns:
        nc = norm(c)
        if any(k in nc for k in keys):
            return c
    return None


def _percent_formatter():
    return FuncFormatter(lambda y, _: f"{y:.0f}%")


# -------- 数值解析与清洗 --------
def first_notna(series: pd.Series):
    """取第一个非 NaN 的值，否则返回 np.nan"""
    for v in series:
        if pd.notna(v) and str(v).strip() != "":
            return v
    return np.nan

def to_float_maybe(x) -> Optional[float]:
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == "": return np.nan
    s = s.replace("cm","").replace("m","").replace("kg","")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group())
    except:
        return np.nan

def parse_height_to_m(val) -> Optional[float]:
    if pd.isna(val) or str(val).strip()=="":
        return np.nan
    s = str(val).lower().strip()
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    v = float(m.group())
    if "cm" in s or v > 3:
        return v / 100.0
    else:
        return v

def parse_weight_kg(val) -> Optional[float]:
    v = to_float_maybe(val)
    return v

def parse_age(val) -> Optional[float]:
    v = to_float_maybe(val)
    return v

def parse_years(val) -> Optional[float]:
    v = to_float_maybe(val)
    return v

# -------- 读取一个 Excel 的信息 --------
def read_one_file(path: Path) -> dict|None:
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        print(f"[SKIP] 无法读取 {path.name} | {e}")
        return None

    df = None
    for sh in xls.sheet_names:
        try:
            t = xls.parse(sh)
        except:
            continue
        if not t.empty:
            df = t
            break
    if df is None:
        print(f"[SKIP] {path.name} 无有效工作表")
        return None

    col_sex   = pick_col(df, ("性别","gender"))
    col_age   = pick_col(df, ("年龄","age"))
    col_ht    = pick_col(df, ("身高","height"))
    col_wt    = pick_col(df, ("体重","weight"))
    col_phys  = pick_col(df, ("生理","生理状况","health","phys"))
    col_edu   = pick_col(df, ("教育","学历","education"))
    col_jiguan= pick_col(df, ("籍贯","户籍","籍"))
    col_gdyrs = pick_col(df, ("广东地区居住","广东居住","在广东居住","广东年","广东地区"))

    sex  = first_notna(df[col_sex]) if col_sex else np.nan
    age  = parse_age(first_notna(df[col_age])) if col_age else np.nan
    ht_m = parse_height_to_m(first_notna(df[col_ht])) if col_ht else np.nan
    wt_kg= parse_weight_kg(first_notna(df[col_wt])) if col_wt else np.nan
    phys = first_notna(df[col_phys]) if col_phys else np.nan
    edu  = first_notna(df[col_edu]) if col_edu else np.nan
    jg   = first_notna(df[col_jiguan]) if col_jiguan else np.nan
    gd_y = parse_years(first_notna(df[col_gdyrs])) if col_gdyrs else np.nan

    bmi = np.nan
    if pd.notna(ht_m) and ht_m>0 and pd.notna(wt_kg):
        bmi = wt_kg / (ht_m*ht_m)

    return {
        "file": path.name,
        "性别": sex,
        "年龄": age,
        "身高_m": ht_m,
        "身高_cm": ht_m*100 if pd.notna(ht_m) else np.nan,
        "体重_kg": wt_kg,
        "BMI": bmi,
        "生理状况": phys,
        "教育经历": edu,
        "籍贯": jg,
        "广东居住年_年": gd_y,
    }

# -------- 统一分箱工具 --------
def nice_bins(min_v: float, max_v: float, width: float):
    if np.isnan(min_v) or np.isnan(max_v):
        return None
    lo = math.floor(min_v / width) * width
    hi = math.ceil (max_v / width) * width
    if hi <= lo:
        hi = lo + width
    edges = np.arange(lo, hi + 0.5*width, width, dtype=float)
    return edges

def plot_hist_with_fixed_bins(series: pd.Series, bins, labels, xlabel: str, title: str, out_png: Path):
    """固定分箱的柱状图（Y=百分比）。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        print(f"[WARN] {title}: 无数据，跳过绘图")
        return
    total = len(s)
    cats = pd.cut(s, bins=bins, labels=labels, right=False, include_lowest=True)
    cnt = cats.value_counts().reindex(labels, fill_value=0)
    pct = cnt * 100.0 / total

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    ax.bar(range(len(pct)), pct.values, edgecolor="black")
    ax.set_xticks(range(len(pct)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("占比（%）")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(_percent_formatter())
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
    print(f"[OK] 保存图：{out_png.name}")


def plot_hist(series: pd.Series, bins, xlabel: str, title: str, out_png: Path, bin_type='numeric'):
    """
    直方/柱状图（Y=百分比）。
    bin_type: 'numeric'（数值直方图）, 'custom_age', 'custom_bmi'
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        print(f"[WARN] {title}: 无数据，跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)

    if bin_type == 'custom_age':
        age_bins   = [18, 26, 31, 41, 51, np.inf]
        age_labels = ['18-25', '26-30', '31-40', '41-50', '50以上']
        counts, _  = np.histogram(s, bins=age_bins)
        pct = counts * 100.0 / len(s)
        ax.bar(range(len(pct)), pct, edgecolor="black")
        ax.set_xticks(range(len(pct)))
        ax.set_xticklabels(age_labels, rotation=45, ha="right")

    elif bin_type == 'custom_bmi':
        bmi_bins   = [0, 18.5, 24, 28, np.inf]
        bmi_labels = ['偏瘦(<18.5)', '正常(18.5-24)', '超重(24-28)', '肥胖(≥28)']
        counts, _  = np.histogram(s, bins=bmi_bins)
        pct = counts * 100.0 / len(s)
        ax.bar(range(len(pct)), pct, edgecolor="black")
        ax.set_xticks(range(len(pct)))
        ax.set_xticklabels(bmi_labels, rotation=45, ha="right")

    else:
        # 数值直方图：直接用 weights 画百分比
        weights = np.ones_like(s) * 100.0 / len(s)
        ax.hist(s, bins=bins, weights=weights, edgecolor="black")
        if isinstance(bins, (list, np.ndarray)) and len(bins) <= 15:
            centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
            tick_labels = [f"{int(bins[i])}–{int(bins[i+1]-1)}" for i in range(len(bins)-1)]
            ax.set_xticks(centers, tick_labels, rotation=45, ha="right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("占比（%）")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(_percent_formatter())
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)
    print(f"[OK] 保存图：{out_png.name}")


# -------- 分类柱状图函数 --------
def plot_bar(series: pd.Series, title: str, out_png: Path, topn=None):
    """分类柱状图（Y=百分比）。"""
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "": np.nan}).dropna()
    if s.empty:
        print(f"[WARN] {title}: 无数据，跳过绘图")
        return

    total = len(s)
    vc = s.value_counts()
    if topn:
        vc = vc.head(topn)
    pct = vc * 100.0 / total

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    pct.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel("占比（%）")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(_percent_formatter())
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"[OK] 保存图：{out_png.name}")

# -------- 主函数 --------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.xls")))
    rows = []
    for p in files:
        info = read_one_file(p)
        if info:
            rows.append(info)

    if not rows:
        print("[DONE] 没读到任何被试文件。检查路径/表格。")
        return

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "participants.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] 已导出汇总：{csv_path}  | 共 {len(df)} 名被试")

    # ======== 连续变量的直方图 ========
    plot_dir = OUT_DIR / "plots"
    
    # 年龄：自定义分箱
    plot_hist(df["年龄"], bins=None, xlabel="年龄/岁", title="年龄分布（自定义分箱）",
              out_png=plot_dir / "age_hist.png", bin_type='custom_age')
    
    # 身高：5 cm 一档
    s_height = df["身高_cm"].dropna()
    if not s_height.empty:
        height_bins = nice_bins(s_height.min(), s_height.max(), 5)
        plot_hist(df["身高_cm"], bins=height_bins, xlabel="身高/cm", title="身高分布（每 5 cm）",
                  out_png=plot_dir / "height_hist.png", bin_type='numeric')
    
    # 体重：5 kg 一档
    s_weight = df["体重_kg"].dropna()
    if not s_weight.empty:
        weight_bins = nice_bins(s_weight.min(), s_weight.max(), 5)
        plot_hist(df["体重_kg"], bins=weight_bins, xlabel="体重/kg", title="体重分布（每 5 kg）",
                  out_png=plot_dir / "weight_hist.png", bin_type='numeric')
    
    # BMI：中国成人标准分箱
    plot_hist(df["BMI"], bins=None, xlabel="BMI (kg/m²)", title="BMI 分布（中国成人标准）",
              out_png=plot_dir / "bmi_hist.png", bin_type='custom_bmi')
    
    # 广东居住年限：固定 2 年一档，最后一档 ≥10 年
    gd_bins   = [0, 2, 4, 6, 8, 10, np.inf]   # 左闭右开：[0,2), [2,4), ..., [10,∞)
    gd_labels = ["0–1", "2–3", "4–5", "6–7", "8–9", "≥10"]
    plot_hist_with_fixed_bins(
        df["广东居住年_年"], bins=gd_bins, labels=gd_labels,
        xlabel="广东地区居住年限/年",
        title="广东地区居住年限分布（每 2 年，含 ≥10 年）",
        out_png=(OUT_DIR / "plots" / "gd_years_hist.png")
    )


    # ======== 分类变量的柱状图 ========
    plot_bar(df["性别"], "性别分布", plot_dir / "gender_bar.png")
    plot_bar(df["教育经历"], "教育经历分布（Top 10）", plot_dir / "education_bar.png", topn=10)
    plot_bar(df["籍贯"], "籍贯分布（Top 10）", plot_dir / "native_bar.png", topn=10)

    print("\n[DONE] 所有图表已生成并保存到：", plot_dir)

if __name__ == "__main__":
    main()