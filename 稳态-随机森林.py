# -*- coding: utf-8 -*-
"""
随机森林预测“热感觉_数值”
- 训练/测试 = 8:2
- 输出：metrics.txt（R²/ RMSE/ MAE），feature_importance.(png/pdf)，
       confusion_matrix_counts.(png/pdf)，confusion_matrix_norm.(png/pdf)，
       feature_importance.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report

# ========= 路径 =========
CSV_PATH = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv")
OUT_DIR  = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\模型输出\RF_thermal_sensation1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 中文字体 =========
for fam in ["Microsoft YaHei","SimHei","PingFang SC","Heiti SC","Noto Sans CJK SC","Source Han Sans SC"]:
    if fam in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [fam]; break
plt.rcParams["axes.unicode_minus"] = False

def read_csv_guess(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="gbk")

# ========= 读取与清洗 =========
df = read_csv_guess(CSV_PATH)

feature_cols = ["temp_mean", "back_temp℃_mean", "hand_temp℃_mean",
                "leg_temp℃_mean", "neck_temp℃_mean", "mTSK_mean"]
target_col = "热感觉_数值"

df = df[feature_cols + [target_col]].copy()
for c in feature_cols + [target_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

X = df[feature_cols]
y = df[target_col]

# ========= 划分数据 =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=180
)

# ========= 训练模型 =========
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=21,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ========= 预测与回归指标 =========
y_pred = rf.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

# 保存指标
with open(OUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
    f.write("随机森林回归 - 预测热感觉_数值\n")
    f.write(f"样本量：全量={len(df)}, 训练={len(X_train)}, 测试={len(X_test)}\n\n")
    f.write(f"R²   = {r2:.4f}\n")
    f.write(f"RMSE = {rmse:.4f}\n")
    f.write(f"MAE  = {mae:.4f}\n")

# ========= 特征重要性 =========
importances = rf.feature_importances_
fi = (pd.Series(importances, index=feature_cols)
      .sort_values(ascending=True))  # 方便水平条形图自下而上

fi.to_csv(OUT_DIR / "feature_importance.csv", encoding="utf-8-sig")

plt.figure(figsize=(7,5), dpi=150)
plt.barh(fi.index, fi.values, color="#5DADE2", edgecolor="black")
plt.xlabel("特征重要性")
plt.title("随机森林特征重要性")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=300)
plt.savefig(OUT_DIR / "feature_importance.pdf")
plt.close()

# ========= 将回归输出离散化为等级，绘制混淆矩阵 =========
# 等级范围固定到 [-3, -2, -1, 0, 1, 2, 3]
classes = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=int)

def clip_round_to_classes(arr):
    arr = np.rint(arr).astype(int)        # 四舍五入
    arr = np.clip(arr, classes.min(), classes.max())
    return arr

y_true_cls = clip_round_to_classes(y_test.values)
y_pred_cls = clip_round_to_classes(y_pred)

cm = confusion_matrix(y_true_cls, y_pred_cls, labels=classes)               # 计数
cm_norm = confusion_matrix(y_true_cls, y_pred_cls, labels=classes, normalize="true") * 100  # 行归一化%

# 保存分类报告（基于离散化后的标签）
report = classification_report(y_true_cls, y_pred_cls, labels=classes, digits=3, zero_division=0)
with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write("（回归输出四舍五入至等级后的）混淆矩阵分类报告\n")
    f.write(report)

# 画热力图函数
def plot_cm(mat, title, cmap="Blues", fmt="d", vmin=None, vmax=None, fname="cm.png"):
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("预测等级")
    ax.set_ylabel("真实等级")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    # 标注
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.0f}" if fmt == "d" else f"{val:.1f}%"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if (val < (vmax or mat.max()) * 0.7) else "white", fontsize=10)
    # colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if fmt != "d": cbar.set_label("行百分比(%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / (fname + ".png"), dpi=300)
    plt.savefig(OUT_DIR / (fname + ".pdf"))
    plt.close()

# 保存两张矩阵：计数 & 行归一化百分比
plot_cm(cm,      "随机森林 混淆矩阵", cmap="Blues", fmt="d",
        vmin=0, vmax=cm.max(), fname="confusion_matrix_counts")
plot_cm(cm_norm, "随机森林 混淆矩阵（行百分比）", cmap="Blues", fmt="p",
        vmin=0, vmax=100, fname="confusion_matrix_norm")

print(f"[DONE] 已输出到：{OUT_DIR}")
print(f"R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
