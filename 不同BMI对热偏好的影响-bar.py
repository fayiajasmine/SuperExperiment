# -*- coding: utf-8 -*-
"""
100% 堆叠条形图：不同稳态温度 × BMI 的温度偏好比例（-1/0/1）
- 输入：包含列 temp_mean, 温度偏好_数值, BMI 的CSV（或批量CSV）
- 温度分箱：20±0.8 / 23±0.8 / 26±0.8 / 29±0.8
- 每个温度一组，组内按 BMI 分箱（<18.5, 18.5-24, 24-28, >28）各一根 100% 堆叠柱
- 段内显示百分比，柱顶显示 n
- 输出 PNG/PDF
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ========= 路径（改成你的）=========
DATA_PATH = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv")
OUT_DIR   = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing\box plots\thermal_prefer_bmi")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG   = OUT_DIR / "stacked_pref_by_temp_bmi_100pct.png"
OUT_PDF   = OUT_DIR / "stacked_pref_by_temp_bmi_100pct.pdf"

# ========= 中文字体 =========
for fam in ["Microsoft YaHei","SimHei","PingFang SC","Heiti SC","Noto Sans CJK SC","Source Han Sans SC"]:
    if fam in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [fam]; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ========= 读取数据 =========
def read_csv_guess(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="gbk")

df = read_csv_guess(DATA_PATH)

need_cols = ["temp_mean", "温度偏好_数值", "BMI"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"缺少必要列：{missing}（需要 {need_cols}）")

df = df[need_cols].copy()
df["temp_mean"]       = pd.to_numeric(df["temp_mean"], errors="coerce")
df["温度偏好_数值"]       = pd.to_numeric(df["温度偏好_数值"], errors="coerce")
df["BMI"]             = pd.to_numeric(df["BMI"], errors="coerce")
df = df.dropna(subset=["temp_mean", "温度偏好_数值", "BMI"])

# ========= BMI 分箱 =========
def bmi_to_label(bmi: float):
    if bmi < 18.5:            return "<18.5"
    if 18.5 <= bmi < 24.0:    return "18.5-24"
    if 24.0 <= bmi < 28.0:    return "24-28"
    if bmi >= 28.0:           return ">28"
    return np.nan

bmi_bins = ["<18.5", "18.5-24", "24-28", ">28"]
df["BMI_bin"] = df["BMI"].apply(bmi_to_label)
df = df.dropna(subset=["BMI_bin"])
df["BMI_bin"] = pd.Categorical(df["BMI_bin"], categories=bmi_bins, ordered=True)

# ========= 温度分箱 =========
def temp_to_label(t: float):
    if 19.2 <= t < 20.8: return "20.2±0.6"
    if 22.2 <= t < 23.8: return "23.2±0.6"
    if 25.2 <= t < 26.8: return "26.1±0.5"
    if 28.2 <= t < 29.8: return "29.0±0.6"
    return np.nan

temp_bins = ["20.2±0.6", "23.2±0.6", "26.1±0.5", "29.0±0.6"]
df["temp_bin"] = df["temp_mean"].apply(temp_to_label)
df = df.dropna(subset=["temp_bin"])
df["temp_bin"] = pd.Categorical(df["temp_bin"], categories=temp_bins, ordered=True)

# ========= 温度偏好标签（固定为 -1/0/1）=========
pref_order  = [-1, 0, 1]
pref_labels = {-1: "更冷(-1)", 0: "不变(0)", 1: "更暖(1)"}
df = df[df["温度偏好_数值"].isin(pref_order)].copy()

# ========= 分组统计（百分比）=========
grp = (
    df.groupby(["temp_bin", "BMI_bin", "温度偏好_数值"])
      .size().rename("n").reset_index()
)
tot = grp.groupby(["temp_bin", "BMI_bin"])["n"].transform("sum")
grp["pct"] = grp["n"] / tot * 100

# 补全缺失类别（保证每个温度×BMI都有 -1/0/1）
full_index = pd.MultiIndex.from_product(
    [temp_bins, bmi_bins, pref_order],
    names=["temp_bin", "BMI_bin", "温度偏好_数值"]
)
grp = (grp.set_index(["temp_bin", "BMI_bin", "温度偏好_数值"])
          .reindex(full_index, fill_value=0)
          .reset_index())
grp["pref_label"] = grp["温度偏好_数值"].map(pref_labels)

# 每柱样本量 n
n_per_bar = (
    df.groupby(["temp_bin", "BMI_bin"])
      .size().rename("N")
      .reindex(pd.MultiIndex.from_product([temp_bins, bmi_bins], names=["temp_bin","BMI_bin"]))
      .reset_index()
)

# 宽表（每柱三段：更冷/不变/更暖）
wide_pct = (
    grp.pivot_table(index=["temp_bin", "BMI_bin"], columns="pref_label", values="pct", fill_value=0)
       .reindex(index=pd.MultiIndex.from_product([temp_bins, bmi_bins], names=["temp_bin","BMI_bin"]))
)

# ========= 绘图 =========
fig, ax = plt.subplots(figsize=(14, 7), dpi=200)

# 每个温度一个组，组内 4 根柱（按 BMI）
group_centers = np.arange(len(temp_bins)) * 2.6
offsets = np.linspace(-0.9, 0.9, len(bmi_bins))
bar_width = 0.38

colors = {
    "更冷(-1)": "#66c2a5",
    "不变(0)" : "#ffff99",
    "更暖(1)" : "#fdc086",
}

bars_positions, bars_heights, bars_n = [], [], []
for i, tbin in enumerate(temp_bins):
    for j, bbin in enumerate(bmi_bins):
        x = group_centers[i] + offsets[j]
        bars_positions.append(x)
        row = wide_pct.loc[(tbin, bbin)] if (tbin, bbin) in wide_pct.index else pd.Series({k:0 for k in colors})
        bars_heights.append([row.get("更冷(-1)",0), row.get("不变(0)",0), row.get("更暖(1)",0)])
        n_val = n_per_bar[(n_per_bar["temp_bin"]==tbin) & (n_per_bar["BMI_bin"]==bbin)]["N"]
        bars_n.append(int(n_val.iloc[0]) if len(n_val)>0 else 0)

# 堆叠绘制
bottom = np.zeros(len(bars_positions))
keys_order = ["更冷(-1)","不变(0)","更暖(1)"]
for k in keys_order:
    vals = [h[keys_order.index(k)] for h in bars_heights]
    ax.bar(bars_positions, vals, bar_width, bottom=bottom,
           color=colors[k], edgecolor="white", linewidth=0.8, label=k if k not in [l.get_label() for l in ax.containers] else "")
    for x, v, btm in zip(bars_positions, vals, bottom):
        if v >= 3:
            ax.text(x, btm + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=8)
    bottom += np.array(vals)

# 顶部样本数
for x, n in zip(bars_positions, bars_n):
    ax.text(x, 103, f"n={n}", ha="center", va="bottom", fontsize=8, color="#555")

# x 轴标签
ax.set_xticks(group_centers)
ax.set_xticklabels(temp_bins)
# 在组下方标出 BMI 子标签
for i, tbin in enumerate(temp_bins):
    for j, bbin in enumerate(bmi_bins):
        ax.text(group_centers[i] + offsets[j], -6, bbin, ha="center", va="top", fontsize=9)

ax.set_ylim(0, 110)
ax.set_ylabel("比例 (%)")
ax.set_xlabel("室内温度区间 (℃)", labelpad=20)
ax.set_title("不同稳态温度 × BMI 的温度偏好")
ax.legend(title="温度偏好", frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout(rect=[0.06, 0.06, 0.82, 0.96])
plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.3)
plt.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.3)
plt.show()

print(f"[DONE] 图已保存：\n- {OUT_PNG}\n- {OUT_PDF}")
