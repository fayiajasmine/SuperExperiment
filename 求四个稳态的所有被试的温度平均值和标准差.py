# -*- coding: utf-8 -*-
"""
根据 temp_mean 列将稳态温度分为四类（20, 23, 26, 29），
对每一类计算所有被试的平均值、标准差与样本数。
输出：task_stable_windows_temp_mean_summary.csv
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ===== 路径配置 =====
CSV_PATH = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_allcolumns_with_demo.csv")
OUT_PATH = CSV_PATH.with_name(CSV_PATH.stem + "_temp_summary.csv")

# ===== 读取数据 =====
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

if "temp" not in df.columns:
    raise ValueError("未找到 temp 列！")

# ===== 设定温度分档 =====
# 将 temp_mean 四舍五入到最接近的 {20, 23, 26, 29}
bins_target = [20, 23, 26, 29]

# 找最近温度类别
def nearest_temp(x):
    if pd.isna(x):
        return np.nan
    return min(bins_target, key=lambda b: abs(b - x))

df["temp_bin"] = df["temp"].apply(nearest_temp)

# ===== 分组统计 =====
summary = (
    df.dropna(subset=["temp_bin"])
      .groupby("temp_bin")["temp"]
      .agg(temp_mean_avg="mean", temp_mean_std="std", n="count")
      .reset_index()
      .sort_values("temp_bin")
)

# ===== 导出 =====
summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"[DONE] 已保存：{OUT_PATH}")
print(summary)
