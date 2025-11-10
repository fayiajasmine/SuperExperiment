# -*- coding: utf-8 -*-
"""
不同性别 × 温度区间的皮肤温度箱型图（背/手/腿/颈/mTSK）
- 自动分温度区间 (20/23/26/29 ±0.8℃)
- 男女并排箱型图 + 均值 + n 标注
- 自动保存 PNG 和 PDF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager
import os

# ========= 路径 =========
CSV_PATH = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv"
OUT_DIR  = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing\box plots\skin_temp_by_gender"
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
need_cols = ["temp_mean", "性别"]
skin_cols = ["back_temp℃_mean", "hand_temp℃_mean", "leg_temp℃_mean", "neck_temp℃_mean", "mTSK_mean"]
for c in need_cols + skin_cols:
    if c not in df.columns:
        print(f"[WARN] 缺少列：{c}")

df = df[need_cols + skin_cols].dropna(subset=["temp_mean", "性别"])

# ========= 定义温度区间标签 =========
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

# ========= 性别配色 =========
colors = {"女": "#f4a7a3", "男": "#9cc7ff"}
edges  = {"女": "#d33",    "男": "#274b7a"}

# ========= 绘图函数 =========
def plot_box_for_column(colname, ylabel, outprefix):
    sub = df.dropna(subset=[colname])
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200, constrained_layout=False)
    genders_order = ["女", "男"]

    group_x = np.arange(len(bin_labels)) * 1.8
    offset = 0.22
    width = 0.34
    positions, boxdata, which_gender = [], [], []

    for i, b in enumerate(bin_labels):
        for g in genders_order:
            seg = sub[(sub["temp_bin"] == b) & (sub["性别"] == g)]
            if seg.empty:
                continue
            positions.append(group_x[i] + (-offset if g == "女" else offset))
            boxdata.append(seg[colname].values)
            which_gender.append(g)

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

    # 美化
    for i, patch in enumerate(bp["boxes"]):
        g = which_gender[i]
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

    # 标注 n
    for pos, vals, g in zip(positions, boxdata, which_gender):
        ax.text(pos, np.nanmax(vals) + 0.2, f"n={len(vals)}",
                ha="center", va="bottom", color=edges[g], fontsize=10)

    ax.set_xticks(group_x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("室内温度区间 (℃)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [Patch(facecolor=colors[g], edgecolor=edges[g], label=g, alpha=0.45) for g in genders_order]
    ax.legend(handles=handles, title="性别", frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    plt.tight_layout(rect=[0.06, 0.08, 0.82, 0.96])

    # 保存
    out_png = os.path.join(OUT_DIR, f"{outprefix}.png")
    out_pdf = os.path.join(OUT_DIR, f"{outprefix}.pdf")
    plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.4)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.4)
    plt.close()
    print(f"[OK] 保存：{out_png}")

# ========= 循环绘制每个皮肤部位 =========
for col in skin_cols:
    ylabel = col.replace("_mean", "").replace("℃", " (℃)")
    outprefix = f"box_{col.replace('℃','').replace('_mean','')}"
    plot_box_for_column(col, ylabel, outprefix)

print("\n[DONE] 所有皮肤温度箱型图已生成。")
