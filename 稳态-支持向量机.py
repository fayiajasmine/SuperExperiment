# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
)
from matplotlib.ticker import FuncFormatter


# ========= 路径 =========
CSV_PATH = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\任务记录\起止时间汇总\task_stable_windows_mapped-without25_with_tempstats_plus_signals_votes_with_demo.csv")
OUT_DIR  = Path(r"D:\00 读研\04 组会\12 小米可穿戴项目\00 实验数据\超级大实验\模型输出\SVR_thermal_sensation2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 中文字体 =========
for fam in ["Microsoft YaHei","SimHei","PingFang SC","Heiti SC","Noto Sans CJK SC","Source Han Sans SC"]:
    if fam in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [fam]; break
plt.rcParams["axes.unicode_minus"] = False

# ========= 读取数据 =========
def read_csv_guess(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="gbk")

feature_cols = [
    "temp_mean",
    "back_temp℃_mean",
    "hand_temp℃_mean",
    "leg_temp℃_mean",
    "neck_temp℃_mean",
    "mTSK_mean",
]
target_col = "热感觉_数值"
valid_classes = [-3,-2,-1,0,1,2,3]

df = read_csv_guess(CSV_PATH)
df = df[feature_cols + [target_col]].copy()
for c in feature_cols + [target_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()
df = df[df[target_col].isin(valid_classes)]

X = df[feature_cols].values
y = df[target_col].astype(float).values

# ========= 划分训练/测试 =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=180
)

# ========= SVR 模型 =========
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=5, gamma="scale"))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# ========= 计算指标 =========
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# 保存指标
metrics_path = OUT_DIR / "metrics.txt"
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"样本量: 全量={len(df)}, 训练={len(y_train)}, 测试={len(y_test)}\n\n")
    f.write(f"R² = {r2:.4f}\n")
    f.write(f"RMSE = {rmse:.4f}\n")
    f.write(f"MAE = {mae:.4f}\n")
print(f"[OK] 指标已写入 {metrics_path}")

# ========= 绘制真实 vs 预测 散点图 =========
fig, ax = plt.subplots(figsize=(6,6), dpi=160)
ax.scatter(y_test, y_pred, alpha=0.7, edgecolors="k")
ax.plot([-3,3],[-3,3], "r--", lw=1.5)
ax.set_xlim(-3.5,3.5); ax.set_ylim(-3.5,3.5)
ax.set_xlabel("真实热感觉")
ax.set_ylabel("预测热感觉 (连续)")
ax.set_title(f"SVR 回归预测结果\nR²={r2:.3f}, RMSE={rmse:.3f}")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_true_vs_pred.png", dpi=600)
plt.close(fig)

# ========= 四舍五入为离散等级并绘制混淆矩阵 =========
y_pred_rounded = np.clip(np.round(y_pred), -3, 3).astype(int)
labels = valid_classes
cm = confusion_matrix(y_test.astype(int), y_pred_rounded, labels=labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

def plot_confmat(mat, labels, title, normalize=False, fname="cm.png"):
    fig, ax = plt.subplots(figsize=(7,6), dpi=160)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=(1 if normalize else None))
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("预测等级"); ax.set_ylabel("真实等级")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i,j]
            txt = f"{val*100:.0f}%" if normalize else f"{int(val)}"
            ax.text(j,i,txt,ha="center",va="center",fontsize=9,
                    color=("black" if normalize or val<20 else "white"))
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    if normalize:
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x*100:.0f}")  
        )
        cbar.set_label("行百分比(%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close(fig)

plot_confmat(cm, labels, "支持向量机 混淆矩阵", normalize=False, fname="cm_counts.png")
plot_confmat(cm_norm, labels, "支持向量机 混淆矩阵（行百分比）", normalize=True, fname="cm_normalized.png")

print("[DONE] 所有结果已保存至：", OUT_DIR)
