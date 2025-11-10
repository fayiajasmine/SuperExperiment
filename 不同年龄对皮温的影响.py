# -*- coding: utf-8 -*-
"""
不同年龄 × 温度区间 的皮肤温度箱型图（背/手/腿/颈/mTSK）
- 年龄分组：18–25, 26–30, 31–40, 41–50, >50
- 自动分温度区间 (20/23/26/29 ±0.8℃)
- 并排箱型图 + 均值 + n 标注
- 保存 PNG 和 PDF（600dpi）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager
import os

# ========= 路径 =========
CSV_PATH = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv"
OUT_DIR  = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing\box plots\skin_temp_by_age"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 中文字体 =========
for fam in ["Microsoft YaHei","SimHei","PingFang SC","Heiti SC","Noto Sans CJK SC","Source Han Sans SC"]:
    if fam in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [fam]; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ========= 读取 CSV =========
def read_csv_guess(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")

df = read_csv_guess(CSV_PATH)

# ========= 检查列 =========
need_cols = ["temp_mean", "年龄"]
skin_cols = ["back_temp℃_mean", "hand_temp℃_mean", "leg_temp℃_mean", "neck_temp℃_mean", "mTSK_mean"]
for c in need_cols + skin_cols:
    if c not in df.columns:
        print(f"[WARN] 缺少列：{c}")

df = df[need_cols + skin_cols].dropna(subset=["temp_mean", "年龄"])

# ========= 年龄分箱 =========
def age_to_label(a: float):
    if 18 <= a <= 25: return "18–25"
    if 26 <= a <= 30: return "26–30"
    if 31 <= a <= 40: return "31–40"
    if 41 <= a <= 50: return "41–50"
    if a > 50:        return ">50"
    return np.nan

age_labels = ["18–25", "26–30", "31–40", "41–50", ">50"]
df["age_bin"] = df["年龄"].apply(age_to_label)
df = df.dropna(subset=["age_bin"])
df["age_bin"] = pd.Categorical(df["age_bin"], categories=age_labels, ordered=True)

# ========= 温度分箱 =========
def temp_to_label(t: float):
    if 19.2 <= t < 20.8: return "20.2±0.6"
    if 22.2 <= t < 23.8: return "23.2±0.6"
    if 25.2 <= t < 26.8: return "26.1±0.5"
    if 28.2 <= t < 29.8: return "29.0±0.6"
    return np.nan

bin_labels = ["20.2±0.6", "23.2±0.6", "26.1±0.5", "29.0±0.6"]
df["temp_bin"] = df["temp_mean"].apply(temp_to_label)
df = df.dropna(subset=["temp_bin"])
df["temp_bin"] = pd.Categorical(df["temp_bin"], categories=bin_labels, ordered=True)

# ========= 配色（5 组年龄）=========
colors = {
    "18–25": "#f4a7a3",
    "26–30": "#f4c27a",
    "31–40": "#9cc7ff",
    "41–50": "#7bd1c7",
    ">50": "#c39ef3",
}
edges = {
    "18–25": "#c23b2b",
    "26–30": "#b97a00",
    "31–40": "#274b7a",
    "41–50": "#1f7b7b",
    ">50": "#6d47af",
}

# ========= 绘图函数 =========
def plot_box_for_column(colname, ylabel, outprefix):
    sub = df.dropna(subset=[colname])
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200, constrained_layout=False)
    group_order = age_labels

    group_x = np.arange(len(bin_labels)) * 2.2
    k = len(group_order)
    total_width = 1.4
    step = total_width / k
    start = -(total_width / 2) + step / 2
    pos_offsets = [start + i * step for i in range(k)]
    width = step * 0.85

    positions, boxdata, which_grp = [], [], []

    for i, tbin in enumerate(bin_labels):
        for j, grp in enumerate(group_order):
            seg = sub[(sub["temp_bin"] == tbin) & (sub["age_bin"] == grp)]
            if seg.empty:
                continue
            positions.append(group_x[i] + pos_offsets[j])
            boxdata.append(seg[colname].values)
            which_grp.append(grp)

    if not boxdata:
        print(f"[WARN] {colname} 无数据可绘图。")
        return

    bp = ax.boxplot(
        boxdata,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        meanline=False,
        whis=1.5,
        manage_ticks=False
    )

    # 美化样式
    for i, patch in enumerate(bp["boxes"]):
        g = which_grp[i]
        patch.set(facecolor=colors[g], edgecolor=edges[g], alpha=0.45, linewidth=1.4)
        bp["medians"][i].set(color=edges[g], linewidth=2.0)
        w1, w2 = 2*i, 2*i+1
        bp["whiskers"][w1].set(color=edges[g], linewidth=1.2)
        bp["whiskers"][w2].set(color=edges[g], linewidth=1.2)
        bp["caps"][w1].set(color=edges[g], linewidth=1.2)
        bp["caps"][w2].set(color=edges[g], linewidth=1.2)
        bp["fliers"][i].set(marker="o", markersize=3.0, markerfacecolor="none",
                            markeredgecolor=edges[g], alpha=0.8)
        bp["means"][i].set(marker="D", markersize=4, markerfacecolor="black",
                           markeredgecolor="black", linestyle="none")

    # n 标注
    for pos, vals, g in zip(positions, boxdata, which_grp):
        ax.text(pos, np.nanmax(vals) + 0.2, f"n={len(vals)}",
                ha="center", va="bottom", color=edges[g], fontsize=10)

    ax.set_xticks(group_x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("室内温度区间 (℃)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [Patch(facecolor=colors[g], edgecolor=edges[g], label=g, alpha=0.45) for g in group_order]
    ax.legend(handles=handles, title="年龄分组", frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

    plt.tight_layout(rect=[0.06, 0.08, 0.82, 0.96])

    out_png = os.path.join(OUT_DIR, f"{outprefix}.png")
    out_pdf = os.path.join(OUT_DIR, f"{outprefix}.pdf")
    plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.4)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"[OK] 保存：{out_png}")

# ========= 循环绘制每个皮肤部位 =========
for col in skin_cols:
    ylabel = col.replace("_mean", "").replace("℃", " (℃)")
    outprefix = f"box_{col.replace('℃','').replace('_mean','')}_by_age"
    plot_box_for_column(col, ylabel, outprefix)

print("\n[DONE] 所有 年龄 × 温度区间 皮肤温度箱型图已生成。")
