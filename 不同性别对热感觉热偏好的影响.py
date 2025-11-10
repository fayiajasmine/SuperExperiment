# -*- coding: utf-8 -*-
"""
TSV × 性别 × 温度区间(20/23/26/29 ±0.8℃) 箱型图 + 温度偏好箱型图 —— 打印友好布局
- 自动识别 CSV 编码
- 男女并排箱型图 + 均值(黑菱形) + 每组 n
- 固定 y 轴刻度：TSV 为 -3..3；温度偏好为 -1..1
- 放大画布、增大组间距、图例移到画布外
- 保存 600dpi PNG 与 PDF（矢量）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager

# ========= 路径 =========
CSV_PATH = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv"
OUT_DIR  = r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\个体合并数据\preprocessing\box plots\gender_out"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TSV_PNG  = os.path.join(OUT_DIR, "plot_tsv_by_gender_tempbin_print.png")
OUT_TSV_PDF  = os.path.join(OUT_DIR, "plot_tsv_by_gender_tempbin_print.pdf")
OUT_PREF_PNG = os.path.join(OUT_DIR, "plot_pref_by_gender_tempbin_print.png")
OUT_PREF_PDF = os.path.join(OUT_DIR, "plot_pref_by_gender_tempbin_print.pdf")

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

# ========= 温度分箱：20/23/26/29 ±0.8 ℃（使用你的均值±std标签）=========
def temp_to_label(t: float):
    if 19.2 <= t < 20.8: return "20.2±0.6"
    if 22.2 <= t < 23.8: return "23.2±0.6"
    if 25.2 <= t < 26.8: return "26.1±0.5"
    if 28.2 <= t < 29.8: return "29.0±0.6"
    return np.nan

bin_labels = ["20.2±0.6", "23.2±0.6", "26.1±0.5", "29.0±0.6"]

def prepare_common(df_in: pd.DataFrame) -> pd.DataFrame:
    need = ["temp_mean", "性别"]
    for c in need:
        if c not in df_in.columns:
            raise ValueError(f"CSV 缺少列：{c}")
    d = df_in.copy()
    d["temp_mean"] = pd.to_numeric(d["temp_mean"], errors="coerce")
    d = d.dropna(subset=["temp_mean", "性别"])
    d["temp_bin"] = d["temp_mean"].apply(temp_to_label)
    d = d.dropna(subset=["temp_bin"])
    d["temp_bin"] = pd.Categorical(d["temp_bin"], categories=bin_labels, ordered=True)
    return d

# ========= 通用绘图函数 =========
def plot_gender_box(
    df_plot: pd.DataFrame,
    value_col: str,
    y_label: str,
    out_png: str,
    out_pdf: str,
    fixed_yticks=None,
    fixed_ylim=None
):
    if value_col not in df_plot.columns:
        print(f"[WARN] 缺少列：{value_col}，跳过绘制。")
        return

    d = df_plot[["temp_bin", "性别", value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["temp_bin", "性别", value_col])
    if d.empty:
        print(f"[WARN] {value_col} 无可绘制数据。")
        return

    fig, ax = plt.subplots(figsize=(14, 8), dpi=200, constrained_layout=False)

    colors = {"女": "#f4a7a3", "男": "#9cc7ff"}   # 填充色
    edges  = {"女": "#d33",    "男": "#274b7a"}   # 描边/中位数
    genders_order = ["女", "男"]

    # 组位置：拉大间距
    group_x = np.arange(len(bin_labels)) * 1.8
    offset  = 0.22
    width   = 0.34

    # 收集箱型数据与位置
    positions, boxdata, which_gender = [], [], []
    for i, b in enumerate(bin_labels):
        for g in genders_order:
            sub = d[(d["temp_bin"] == b) & (d["性别"] == g)]
            if sub.empty:
                continue
            positions.append(group_x[i] + (-offset if g == "女" else offset))
            boxdata.append(sub[value_col].values)
            which_gender.append(g)

    if not boxdata:
        print(f"[WARN] {value_col} 分箱后没有数据。")
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
        g = which_gender[i]
        patch.set(facecolor=colors[g], edgecolor=edges[g], alpha=0.45, linewidth=1.4)
        bp["medians"][i].set(color=edges[g], linewidth=2.0)
        for j in range(2*i, 2*i+2):
            bp["whiskers"][j].set(color=edges[g], linewidth=1.2)
            bp["caps"][j].set(color=edges[g], linewidth=1.2)
        bp["fliers"][i].set(marker="o", markersize=3.0, markerfacecolor="none",
                            markeredgecolor=edges[g], alpha=0.8)
        bp["means"][i].set(marker="D", markersize=4, markerfacecolor="black",
                           markeredgecolor="black", linestyle="none")

    # y 轴：固定 or 自适应
    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)
    else:
        all_vals = np.concatenate([v for v in boxdata if len(v) > 0])
        ymin = np.nanmin(all_vals) - 0.4
        ymax = np.nanmax(all_vals) + 1.0
        ax.set_ylim(ymin, ymax)
    if fixed_yticks is not None:
        ax.set_yticks(fixed_yticks)

    # n 标注（用当前 ylim 顶部 5% 处，避免越界）
    y_top = ax.get_ylim()[1]
    for pos, vals, g in zip(positions, boxdata, which_gender):
        ax.text(pos, min(y_top, np.nanmax(vals) + 0.35), f"n={len(vals)}",
                ha="center", va="bottom", color=edges[g], fontsize=12)

    # 轴与网格
    ax.set_xticks(group_x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("室内温度区间 (℃)")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # 图例移到画布外
    handles = [Patch(facecolor=colors[g], edgecolor=edges[g], label=g, alpha=0.45)
               for g in ["女", "男"]]
    ax.legend(handles=handles, title="性别", frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

    plt.tight_layout(rect=[0.06, 0.08, 0.82, 0.96])

    # 保存
    plt.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.4)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.4)
    plt.show()
    print(f"[DONE] 已保存：{out_png}（600dpi） 和 {out_pdf}（矢量）")

# ========= 数据准备 =========
df_common = prepare_common(df)

# ========= 图1：热感觉（固定 -3..3 标签）=========
if "热感觉_数值" in df_common.columns:
    df_common["热感觉_数值"] = pd.to_numeric(df_common["热感觉_数值"], errors="coerce")
plot_gender_box(
    df_plot=df_common,
    value_col="热感觉_数值",
    y_label="热感觉（-3 ~ 3）",
    out_png=OUT_TSV_PNG,
    out_pdf=OUT_TSV_PDF,
    fixed_yticks=[-3, -2, -1, 0, 1, 2, 3],
    fixed_ylim=[-3.5, 3.5]
)

# ========= 图2：温度偏好（固定 -1..1 标签）=========
if "温度偏好_数值" not in df_common.columns:
    # 若原表列名不同，可在这里改成你的实际列名
    print("[WARN] 表中缺少列：温度偏好_数值（将跳过温度偏好图）。")
else:
    df_common["温度偏好_数值"] = pd.to_numeric(df_common["温度偏好_数值"], errors="coerce")
    plot_gender_box(
        df_plot=df_common,
        value_col="温度偏好_数值",
        y_label="温度偏好（-1 ~ 1）",
        out_png=OUT_PREF_PNG,
        out_pdf=OUT_PREF_PDF,
        fixed_yticks=[-1, 0, 1],
        fixed_ylim=[-1.3, 1.3]
    )
