# -*- coding: utf-8 -*-
"""
100% 堆叠条形图：不同稳态温度 × 性别 的温度偏好比例（-1/0/1）
- 从文件夹批量读取 CSV（或单个 CSV）
- 温度分箱：20±0.8 / 23±0.8 / 26±0.8 / 29±0.8
- 每个温度一组，组内按“女、男”各一根 100% 堆叠柱
- 段内显示百分比，柱顶显示 n
- 输出 PNG/PDF
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# ========= 路径（改成你的文件夹路径或单个CSV路径均可）=========
DATA_DIR = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv")   # ← 改这里：可以是文件夹，也可以直接指向一个CSV
OUT_DIR  = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing\box plots\thermal_prefer_gender")
OUT_PNG  = OUT_DIR / "stacked_pref_by_temp_gender_100pct.png"
OUT_PDF  = OUT_DIR / "stacked_pref_by_temp_gender_100pct.pdf"

# ========= 中文字体（自动就近选择）=========
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

if DATA_DIR.is_dir():
    csv_files = [p for p in DATA_DIR.glob("*.csv")]
    if not csv_files:
        raise FileNotFoundError(f"目录中未找到 CSV：{DATA_DIR}")
    dfs = []
    for p in csv_files:
        df = read_csv_guess(p)
        df["__src__"] = p.name
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
else:
    if DATA_DIR.suffix.lower() != ".csv":
        raise ValueError("请提供 CSV 文件或包含 CSV 的文件夹路径。")
    data = read_csv_guess(DATA_DIR)
    data["__src__"] = Path(DATA_DIR).name

need_cols = ["temp_mean", "温度偏好_数值", "性别"]
missing = [c for c in need_cols if c not in data.columns]
if missing:
    raise ValueError(f"缺少必要列：{missing}（要求包含 {need_cols}）")

# 只保留需要列并清洗
df = data[need_cols].copy()
df["temp_mean"] = pd.to_numeric(df["temp_mean"], errors="coerce")
df["温度偏好_数值"] = pd.to_numeric(df["温度偏好_数值"], errors="coerce")
df["性别"] = df["性别"].astype(str).str.strip()
df = df.dropna(subset=["temp_mean", "温度偏好_数值", "性别"])

# ========= 温度分箱（稳态目标：20/23/26/29 ±0.8）=========
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

# ========= 规范“温度偏好_数值”的标签顺序 =========
# 你希望固定为 -1, 0, 1（分别代表：更冷/不变/更暖）
pref_order = [-1, 0, 1]
pref_labels = { -1: "更冷(-1)", 0: "不变(0)", 1: "更暖(1)" }
df = df[df["温度偏好_数值"].isin(pref_order)].copy()

# 规范“性别”的顺序（可按需要调整）
gender_order = ["女", "男"]
df["性别"] = pd.Categorical(df["性别"], categories=gender_order, ordered=True)

# ========= 计算按（温度×性别）的百分比 =========
# 分组计数 -> 计算各偏好类别比例
grp = (df
       .groupby(["temp_bin", "性别", "温度偏好_数值"])
       .size()
       .rename("n")
       .reset_index())

# 计算每个 (温度, 性别) 的总数
tot = grp.groupby(["temp_bin", "性别"])["n"].transform("sum")
grp["pct"] = grp["n"] / tot * 100.0

# 补全缺失类别（确保每组都有 -1,0,1）
full_index = pd.MultiIndex.from_product([temp_bins, gender_order, pref_order],
                                        names=["temp_bin","性别","温度偏好_数值"])
grp = grp.set_index(["temp_bin","性别","温度偏好_数值"]).reindex(full_index, fill_value=0).reset_index()

# 为绘图准备矩阵：每根柱子是一组(温度×性别)，列是三个偏好类别的百分比
grp["pref_label"] = grp["温度偏好_数值"].map(pref_labels)
# 计算每根柱子的 n（为标注使用）
n_per_bar = (df
             .groupby(["temp_bin","性别"])
             .size()
             .rename("N")
             .reindex(pd.MultiIndex.from_product([temp_bins, gender_order], names=["temp_bin","性别"]))
             .reset_index())

# 透视为宽表：行是 (温度, 性别)，列是三类偏好，值是 pct
wide_pct = (grp
            .pivot_table(index=["temp_bin","性别"], columns="pref_label", values="pct", fill_value=0)
            .reindex(index=pd.MultiIndex.from_product([temp_bins, gender_order], names=["temp_bin","性别"]))
            )

# ========= 绘图（100% 堆叠条形图）=========
fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

# x 轴位置：每个温度一组，组内两个柱（女、男）
group_centers = np.arange(len(temp_bins)) * 2.0
offset = 0.25
bar_width = 0.45

# 颜色（-1, 0, 1）
colors = {
    "更冷(-1)": "#66c2a5",
    "不变(0)" : "#ffff99",
    "更暖(1)" : "#fdc086",
}

bars_positions = []
bars_heights = []   # 每个柱子的堆叠高度表
bars_labels = []    # 每个柱子对应 (温度, 性别)
bars_n = []         # 每个柱子的样本数

for i, tbin in enumerate(temp_bins):
    for j, g in enumerate(gender_order):
        x = group_centers[i] + ( -offset if g == "女" else offset )
        bars_positions.append(x)
        bars_labels.append((tbin, g))
        # 取出该柱子三段的百分比
        row = wide_pct.loc[(tbin, g)] if (tbin, g) in wide_pct.index else pd.Series({k:0 for k in colors.keys()})
        bars_heights.append([row.get("更冷(-1)",0), row.get("不变(0)",0), row.get("更暖(1)",0)])
        # n
        n_val = n_per_bar[(n_per_bar["temp_bin"]==tbin) & (n_per_bar["性别"]==g)]["N"]
        bars_n.append(int(n_val.iloc[0]) if len(n_val)>0 and pd.notna(n_val.iloc[0]) else 0)

# 画堆叠
bottom = np.zeros(len(bars_positions))
keys_order = ["更冷(-1)","不变(0)","更暖(1)"]
for k in keys_order:
    vals = [h[keys_order.index(k)] for h in bars_heights]
    ax.bar(bars_positions, vals, bar_width, bottom=bottom, label=k, color=colors[k], edgecolor="white", linewidth=0.8)
    # 在各段中间标注百分比（>3%再标，避免太挤）
    for x, v, btm in zip(bars_positions, vals, bottom):
        if v >= 3:
            ax.text(x, btm + v/2, f"{v:.0f}%", ha="center", va="center", fontsize=9)
    bottom += np.array(vals)

# 顶部标注 n
for x, n in zip(bars_positions, bars_n):
    ax.text(x, 102, f"n={n}", ha="center", va="bottom", fontsize=9, color="#555")

# x 轴：温度组+组内男女标签
ax.set_xticks(group_centers)
ax.set_xticklabels(temp_bins)
# 组内小刻度显示“女/男”
for i, tbin in enumerate(temp_bins):
    ax.text(group_centers[i]-offset, -6, "女", ha="center", va="top", fontsize=10)
    ax.text(group_centers[i]+offset, -6, "男", ha="center", va="top", fontsize=10)

ax.set_ylim(0, 110)  # 顶部留位置给 n
ax.set_ylabel("比例 (%)")
ax.set_xlabel("室内温度区间 (℃)", labelpad=20)
ax.set_title("不同稳态温度 × 性别 的温度偏好")

# 图例移到外侧
leg = ax.legend(title="温度偏好", frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout(rect=[0.06, 0.06, 0.82, 0.96])
plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.3)
plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.3)
plt.show()

print(f"[DONE] 已保存图：\n- {OUT_PNG}\n- {OUT_PDF}")
