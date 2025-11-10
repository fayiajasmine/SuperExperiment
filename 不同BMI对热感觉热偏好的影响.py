# -*- coding: utf-8 -*-
"""
TSV / 温度偏好 × BMI × 温度区间(20/23/26/29 ±0.8℃) —— 打印友好箱型图（双图 + 固定Y轴刻度）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager
import os

# ========= 路径 =========
CSV_PATH = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv"
OUT_DIR  = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing\box plots\bmi_out"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TSV_PNG = os.path.join(OUT_DIR, "plot_tsv_by_bmi_tempbin_print.png")
OUT_TSV_PDF = os.path.join(OUT_DIR, "plot_tsv_by_bmi_tempbin_print.pdf")
OUT_PREF_PNG = os.path.join(OUT_DIR, "plot_pref_by_bmi_tempbin_print.png")
OUT_PREF_PDF = os.path.join(OUT_DIR, "plot_pref_by_bmi_tempbin_print.pdf")

# ========= 中文字体（自动就近选择）=========
for fam in ["Microsoft YaHei","SimHei","PingFang SC","Heiti SC","Noto Sans CJK SC","Source Han Sans SC"]:
    if fam in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [fam]; break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ========= 读取 CSV（自动尝试编码）=========
def read_csv_guess(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")

df = read_csv_guess(CSV_PATH)

# 只保留需要列并清洗（temp_mean, BMI + 票面两列）
need_cols = ["temp_mean", "BMI", "热感觉_数值", "温度偏好_数值"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV 缺少列：{missing}")

df = df[need_cols].copy()
df["temp_mean"] = pd.to_numeric(df["temp_mean"], errors="coerce")
df["BMI"]       = pd.to_numeric(df["BMI"],       errors="coerce")
df["热感觉_数值"]   = pd.to_numeric(df["热感觉_数值"], errors="coerce")
df["温度偏好_数值"] = pd.to_numeric(df["温度偏好_数值"], errors="coerce")
df = df.dropna(subset=["temp_mean", "BMI"])

# ========= BMI 分箱：＜18.5，18.5–24.0，24.0–28.0，≥28.0 =========
def bmi_to_label(bmi: float):
    if bmi < 18.5:        return "<18.5"
    if 18.5 <= bmi < 24:  return "18.5–24.0"
    if 24   <= bmi < 28:  return "24.0–28.0"
    if bmi >= 28:         return "≥28.0"
    return np.nan

bmi_labels = ["<18.5", "18.5–24.0", "24.0–28.0", "≥28.0"]
df["bmi_bin"] = df["BMI"].apply(bmi_to_label)
df = df.dropna(subset=["bmi_bin"])
df["bmi_bin"] = pd.Categorical(df["bmi_bin"], categories=bmi_labels, ordered=True)

# ========= 温度分箱：20/23/26/29 ±0.8 ℃（用你的四个均值±std标签）=========
def temp_to_label(t: float):
    if 19.2 <= t < 20.8: return "20.2±0.6"
    if 22.2 <= t < 23.8: return "23.2±0.6"
    if 25.2 <= t < 26.8: return "26.1±0.5"
    if 28.2 <= t < 29.8: return "29.0±0.6"
    return np.nan

temp_labels = ["20.2±0.6", "23.2±0.6", "26.1±0.5", "29.0±0.6"]
df["temp_bin"] = df["temp_mean"].apply(temp_to_label)
df = df.dropna(subset=["temp_bin"])
df["temp_bin"] = pd.Categorical(df["temp_bin"], categories=temp_labels, ordered=True)

# ========= 配色（4 组 BMI）=========
fill_colors = {
    "<18.5"    : "#9cc7ff",
    "18.5–24.0": "#7bd1c7",
    "24.0–28.0": "#f4c27a",
    "≥28.0"    : "#f4a7a3",
}
edge_colors  = {
    "<18.5"    : "#274b7a",
    "18.5–24.0": "#1f7b7b",
    "24.0–28.0": "#b97a00",
    "≥28.0"    : "#c23b2b",
}

# ========= 通用绘图函数 =========
def plot_by_bmi_tempbin(
    df_in: pd.DataFrame,
    value_col: str,
    y_label: str,
    out_png: str,
    out_pdf: str,
    fixed_y_ticks=None,
    fixed_y_lim=None
):
    if value_col not in df_in.columns:
        print(f"[WARN] 缺少列：{value_col}，跳过绘制。")
        return

    data = df_in[["temp_bin", "bmi_bin", value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=["temp_bin", "bmi_bin", value_col])
    if data.empty:
        print(f"[WARN] {value_col} 无可绘制数据。")
        return

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=200, constrained_layout=False)

    # x 轴组中心（每个温度分箱）
    group_centers = np.arange(len(temp_labels)) * 2.2
    # 在每个温度分箱里并排摆 4 个 BMI 分箱
    k = len(bmi_labels)
    total_width = 1.4
    step = total_width / k
    start = -(total_width/2) + step/2
    pos_offsets = [start + i*step for i in range(k)]
    width = step * 0.85

    positions, boxdata, which_bmi = [], [], []
    for i, tbin in enumerate(temp_labels):
        sub_t = data[data["temp_bin"] == tbin]
        if sub_t.empty:
            continue
        for j, bbin in enumerate(bmi_labels):
            sub = sub_t[sub_t["bmi_bin"] == bbin]
            if sub.empty:
                continue
            positions.append(group_centers[i] + pos_offsets[j])
            boxdata.append(sub[value_col].values)
            which_bmi.append(bbin)

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

    # 样式
    for i, patch in enumerate(bp["boxes"]):
        g = which_bmi[i]
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

    # 固定 y 轴范围/刻度（若提供）
    if fixed_y_lim is not None:
        ax.set_ylim(fixed_y_lim)
    else:
        # 自动：为 n 标注留空间
        all_vals = np.concatenate([v for v in boxdata if len(v) > 0])
        ymin = np.nanmin(all_vals) - 0.4
        ymax = np.nanmax(all_vals) + 1.0
        ax.set_ylim(ymin, ymax)
    if fixed_y_ticks is not None:
        ax.set_yticks(fixed_y_ticks)

    # 标注 n
    # 使用当前 ylim 顶部 5% 处作为标注基线，避免越界
    y_top = ax.get_ylim()[1]
    for pos, vals, bbin in zip(positions, boxdata, which_bmi):
        ax.text(pos, min(y_top, np.nanmax(vals) + 0.35), f"n={len(vals)}",
                ha="center", va="bottom", color=edge_colors[bbin], fontsize=11)

    # 轴与网格
    ax.set_xticks(group_centers)
    ax.set_xticklabels(temp_labels)
    ax.set_xlabel("室内温度区间 (℃)")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 图例移到画布外
    handles = [Patch(facecolor=fill_colors[g], edgecolor=edge_colors[g], label=g, alpha=0.45)
               for g in bmi_labels]
    ax.legend(handles=handles, title="BMI 分箱", frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

    plt.tight_layout(rect=[0.06, 0.08, 0.82, 0.96])

    # 保存
    plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.4)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.4)
    plt.show()
    print(f"[DONE] 已保存：{out_png}（600dpi） 和 {out_pdf}（矢量）")

# ========= 绘制两张图 =========

# 1) 热感觉（固定 -3~3）
plot_by_bmi_tempbin(
    df_in=df,
    value_col="热感觉_数值",
    y_label="热感觉（-3~3）",
    out_png=OUT_TSV_PNG,
    out_pdf=OUT_TSV_PDF,
    fixed_y_ticks=[-3, -2, -1, 0, 1, 2, 3],
    fixed_y_lim=[-3.5, 3.5]
)

# 2) 温度偏好（固定 -1~1）
plot_by_bmi_tempbin(
    df_in=df,
    value_col="温度偏好_数值",
    y_label="温度偏好（-1~1）",
    out_png=OUT_PREF_PNG,
    out_pdf=OUT_PREF_PDF,
    fixed_y_ticks=[-1, 0, 1],
    fixed_y_lim=[-1.3, 1.3]
)
