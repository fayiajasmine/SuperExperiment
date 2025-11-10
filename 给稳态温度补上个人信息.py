# -*- coding: utf-8 -*-
from pathlib import Path
import re
import pandas as pd

# ===== 路径（修改成你的） =====
MAIN_CSV = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes.csv")
PART_CSV = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\问卷\问卷处理后\participants_summary5\participants.csv")
OUT_CSV  = MAIN_CSV.with_name(MAIN_CSV.stem + "_with_demo.csv")

# ===== 自动判断编码读取 =====
def read_csv_guess(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="gbk")

main_df = read_csv_guess(MAIN_CSV)
part_df = read_csv_guess(PART_CSV)

# ===== 提取匹配键（例如 1-YTY-ALLDATA → 1-YTY）=====
def extract_key(s: str) -> str:
    s = str(s).strip()
    parts = s.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}".upper()
    else:
        return s.upper()

main_df["_key"] = main_df["file"].map(extract_key)
part_df["_key"] = part_df["file"].map(extract_key)

# ===== 修正笔误列名 =====
if "广东记住年_年" in part_df.columns and "广东居住年_年" not in part_df.columns:
    part_df = part_df.rename(columns={"广东记住年_年": "广东居住年_年"})

# ===== 保留所需列 =====
need_cols = ["_key", "性别", "年龄", "身高_cm", "体重_kg", "BMI", "广东居住年_年"]
missing = [c for c in need_cols if c not in part_df.columns]
if missing:
    raise ValueError(f"参与者表缺少列：{missing}")
part_df = part_df[need_cols]

# ===== 按 key 合并 =====
merged = main_df.merge(part_df, on="_key", how="left", validate="m:1")

# ===== 删除辅助列并导出 =====
merged = merged.drop(columns=["_key"])
merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"[DONE] 已根据 file 前两段（如 1-YTY）匹配并合并。\n输出文件：{OUT_CSV}")
