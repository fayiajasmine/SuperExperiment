# -*- coding: utf-8 -*-
"""
序数逻辑回归（Ordinal Logistic Regression, LogisticAT）预测“热感觉_数值”
- 训练/测试 = 8:2（按标签分层）
- 标准化 + LogisticAT 组合为 Pipeline
- 输出：
  metrics.txt（Accuracy/F1 + R²/RMSE/MAE）
  classification_report.txt
  confusion_matrix_counts.(png/pdf)
  confusion_matrix_norm.(png/pdf)
  coef_importance.(png/pdf)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error
)

# ---- 关键：序数逻辑回归（mord库）----
from mord import LogisticAT  # 累积logit（阈值）模型：P(y <= k) 用同一组系数+不同阈值

# ========= 路径（按需修改）=========
CSV_PATH = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv")
OUT_DIR  = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\模型输出\OrdinalLogit_thermal_sensation1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 中文字体（自动就近选择）=========
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
feature_cols = [
    "temp_mean",
    "back_temp℃_mean",
    "hand_temp℃_mean",
    "leg_temp℃_mean",
    "neck_temp℃_mean",
    "mTSK_mean",
]
target_col = "热感觉_数值"
valid_classes = [-3, -2, -1, 0, 1, 2, 3]   # 固定的有序等级

df = read_csv_guess(CSV_PATH)
df = df[feature_cols + [target_col]].copy()

# 数值化与缺失处理
for c in feature_cols + [target_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

# 只保留设定的标签等级
df = df[df[target_col].isin(valid_classes)].copy()

X = df[feature_cols].values
y = df[target_col].astype(int).values

# ========= 划分 8:2（分层）=========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========= 建模：标准化 + LogisticAT =========
# LogisticAT 是阈值模型：同一组系数、不同阈值 theta_，天然适合有序等级（-3..3）
# C（或 alpha 的倒数）可调；mord 的 LogisticAT 使用正则化强度 alpha（越大越强）
# 这里用默认 alpha=1.0；如需更强/更弱正则可调 alpha（如 0.1 / 10）
model = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    LogisticAT(alpha=5)   # 调参入口
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========= 分类指标 =========
acc  = accuracy_score(y_test, y_pred)
f1_m = f1_score(y_test, y_pred, average="macro")
f1_w = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred, labels=valid_classes, digits=3, zero_division=0)

# ========= “回归型”指标（把等级当连续量）=========
r2   = r2_score(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = mean_absolute_error(y_test, y_pred)

# ========= 保存指标 =========
with open(OUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
    f.write("序数逻辑回归（LogisticAT） - 预测 热感觉_数值\n")
    f.write(f"样本量：全量={len(df)}, 训练={len(y_train)}, 测试={len(y_test)}\n\n")
    f.write("—— 分类指标 ——\n")
    f.write(f"Accuracy   = {acc:.4f}\n")
    f.write(f"Macro-F1   = {f1_m:.4f}\n")
    f.write(f"Weighted-F1= {f1_w:.4f}\n\n")
    f.write("—— 伪回归型指标（将等级当作连续量）——\n")
    f.write(f"R²   = {r2:.4f}\n")
    f.write(f"RMSE = {rmse:.4f}\n")
    f.write(f"MAE  = {mae:.4f}\n")

with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Classification report（labels = [-3,-2,-1,0,1,2,3]）\n\n")
    f.write(report)

print("指标已写入:", OUT_DIR / "metrics.txt")

# ========= 混淆矩阵（计数 & 行归一化%）=========
cm = confusion_matrix(y_test, y_pred, labels=valid_classes)
cm_norm = confusion_matrix(y_test, y_pred, labels=valid_classes, normalize="true") * 100.0

def plot_cm(mat, labels, title, fmt="d", vmin=None, vmax=None, fname="cm.png", cmap="Blues"):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("预测值"); ax.set_ylabel("真实值")
    ax.set_title(title)
    # 数值标注
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.0f}" if fmt == "d" else f"{val:.1f}%"
            ax.text(j, i, txt, ha="center", va="center",
                    color=("black" if val<14 else "white"),
                    fontsize=9)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    if fmt != "d": cbar.set_label("行百分比(%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / (fname + ".png"), dpi=300)
    plt.savefig(OUT_DIR / (fname + ".pdf"))
    plt.close(fig)

plot_cm(cm,      valid_classes, "序数逻辑回归 混淆矩阵",    fmt="d", vmin=0, vmax=cm.max(),   fname="confusion_matrix_counts", cmap="Blues")
plot_cm(cm_norm, valid_classes, "序数逻辑回归 混淆矩阵（行百分比）", fmt="p", vmin=0, vmax=100,       fname="confusion_matrix_norm",   cmap="Blues")
print("已保存混淆矩阵图片。")

# ========= 特征系数可视化 =========
# 从 Pipeline 中取出标准化器与模型本体
scaler: StandardScaler = model.named_steps["standardscaler"]
ord_model: LogisticAT  = model.named_steps["logisticat"]

# LogisticAT：同一组 coef_（长度 = n_features），阈值为 theta_
coefs = ord_model.coef_.ravel()  # shape (n_features,)
theta = ord_model.theta_         # 阈值（长度 = 类别数-1）

# 为了更易读，按绝对值大小排序
order = np.argsort(np.abs(coefs))
coefs_sorted = coefs[order]
feat_sorted  = [feature_cols[i] for i in order]

plt.figure(figsize=(8, 5), dpi=150)
plt.barh(feat_sorted, coefs_sorted, color="#4C78A8", edgecolor="black")
plt.axvline(0, color="#333", lw=1)
plt.title("Ordinal Logit 系数（标准化后特征）")
plt.xlabel("系数（正：越大越偏向较高热感觉等级）")
plt.tight_layout()
plt.savefig(OUT_DIR / "coef_importance.png", dpi=300)
plt.savefig(OUT_DIR / "coef_importance.pdf")
plt.close()
print("已保存特征系数条形图。")

print(f"[DONE] 输出目录：{OUT_DIR}")
print(f"Accuracy={acc:.3f} | Macro-F1={f1_m:.3f} | R²={r2:.3f} | RMSE={rmse:.3f} | MAE={mae:.3f}")
