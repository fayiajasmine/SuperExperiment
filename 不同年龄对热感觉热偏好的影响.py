# -*- coding: utf-8 -*-
"""
TSV / 温度偏好 × 年龄 × 温度区间(20/23/26/29 ±0.8℃) —— 打印友好箱型图（双图 + 固定Y轴刻度）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager
import os

# ========= 路径 =========
CSV_PATH = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv"
OUT_DIR  = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing\box plots\age_out"
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

# ========= 公共清洗 =========
need_cols_base = ["temp_mean", "年龄"]
miss_base = [c for c in need_cols_base if c not in df.columns]
if miss_base:
    raise ValueError(f"CSV 缺少列：{miss_base}")

df["temp_mean"] = pd.to_numeric(df["temp_mean"], errors="coerce")
df["年龄"] = pd.to_numeric(df["年龄"], errors="coerce")

# 年龄分箱
def age_to_label(age: float):
    if 18 <= age <= 25: return "18-25"
    if 26 <= age <= 30: return "26-30"
    if 31 <= age <= 40: return "31-40"
    if 41 <= age <= 50: return "41-50"
    if age > 50:       return ">50"
    return np.nan

age_labels = ["18-25", "26-30", "31-40", "41-50", ">50"]
df["age_bin"] = df["年龄"].apply(age_to_label)
df = df.dropna(subset=["age_bin"])
df["age_bin"] = pd.Categorical(df["age_bin"], categories=age_labels, ordered=True)

# 温度分箱
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

# ========= 配色 =========
fill_colors = {
    "18-25": "#f4a7a3",
    "26-30": "#f4c27a",
    "31-40": "#9cc7ff",
    "41-50": "#7bd1c7",
    ">50"  : "#c39ef3",
}
edge_colors = {
    "18-25": "#c23b2b",
    "26-30": "#b97a00",
    "31-40": "#274b7a",
    "41-50": "#1f7b7b",
    ">50"  : "#6d47af",
}

# ========= 通用绘图函数 =========
def plot_by_age_tempbin(
    df_in: pd.DataFrame,
    value_col: str,
    y_label: str,
    title_prefix: str,
    out_png: str,
    out_pdf: str,
    y_ticks=None,
    y_lim=None
):
    if value_col not in df_in.columns:
        print(f"[WARN] 缺少列：{value_col}，跳过绘制。")
        return

    dfv = df_in[["temp_mean", "age_bin", "temp_bin", value_col]].copy()
    dfv[value_col] = pd.to_numeric(dfv[value_col], errors="coerce")
    dfv = dfv.dropna(subset=["temp_mean", "age_bin", "temp_bin", value_col])
    if dfv.empty:
        print(f"[WARN] {value_col} 无可绘制数据。")
        return

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=200, constrained_layout=False)

    # 组中心
    group_centers = np.arange(len(bin_labels)) * 2.2
    k = len(age_labels)
    total_width = 1.4
    step = total_width / k
    start = -(total_width/2) + step/2
    pos_offsets = [start + i*step for i in range(k)]
    width = step * 0.85

    positions, boxdata, which_age = [], [], []
    for i, tbin in enumerate(bin_labels):
        sub_t = dfv[dfv["temp_bin"] == tbin]
        for j, abin in enumerate(age_labels):
            sub = sub_t[sub_t["age_bin"] == abin]
            if sub.empty: continue
            positions.append(group_centers[i] + pos_offsets[j])
            boxdata.append(sub[value_col].values)
            which_age.append(abin)

    if not boxdata:
        print(f"[WARN] {value_col} 在分箱后无数据，跳过绘制。")
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

    for i, patch in enumerate(bp["boxes"]):
        g = which_age[i]
        patch.set(facecolor=fill_colors[g], edgecolor=edge_colors[g], alpha=0.45, linewidth=1.4)
        bp["medians"][i].set(color=edge_colors[g], linewidth=2.0)
        w1, w2 = 2*i, 2*i+1
        bp["whiskers"][w1].set(color=edge_colors[g], linewidth=1.2)
        bp["whiskers"][w2].set(color=edge_colors[g], linewidth=1.2)
        bp["caps"][w1].set(color=edge_colors[g], linewidth=1.2)
        bp["caps"][w2].set(color=edge_colors[g], linewidth=1.2)
        bp["fliers"][i].set(marker="o", markersize=3.0, markerfacecolor="none",
                            markeredgecolor=edge_colors[g], alpha=0.8)
        bp["means"][i].set(marker="D", markersize=4, markerfacecolor="black",
                           markeredgecolor="black", linestyle="none")

    # 固定y轴范围与刻度
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # 标注 n
    for pos, vals, abin in zip(positions, boxdata, which_age):
        ax.text(pos, np.nanmax(vals) + 0.2, f"n={len(vals)}",
                ha="center", va="bottom", color=edge_colors[abin], fontsize=10)

    # 轴与网格
    ax.set_xticks(group_centers)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("室内温度区间 (℃)")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [Patch(facecolor=fill_colors[g], edgecolor=edge_colors[g], label=g, alpha=0.45) for g in age_labels]
    ax.legend(handles=handles, title="年龄", frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    plt.tight_layout(rect=[0.06, 0.08, 0.82, 0.96])

    plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.4)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.4)
    plt.show()
    print(f"[DONE] 已保存：{out_png}（600dpi） 和 {out_pdf}（矢量）")

# ========= 绘制两张图 =========

# 1) 热感觉（固定 -3~3）
plot_by_age_tempbin(
    df_in=df,
    value_col="热感觉_数值",
    y_label="热感觉（-3~3）",
    title_prefix="热感觉",
    out_png=os.path.join(OUT_DIR, "plot_tsv_by_age_tempbin_fixed.png"),
    out_pdf=os.path.join(OUT_DIR, "plot_tsv_by_age_tempbin_fixed.pdf"),
    y_ticks=[-3, -2, -1, 0, 1, 2, 3],
    y_lim=[-3.5, 3.5]
)

# 2) 温度偏好（固定 -1~1）
plot_by_age_tempbin(
    df_in=df,
    value_col="温度偏好_数值",
    y_label="温度偏好（-1~1）",
    title_prefix="温度偏好",
    out_png=os.path.join(OUT_DIR, "plot_pref_by_age_tempbin_fixed.png"),
    out_pdf=os.path.join(OUT_DIR, "plot_pref_by_age_tempbin_fixed.pdf"),
    y_ticks=[-1, 0, 1],
    y_lim=[-1.3, 1.3]
)
