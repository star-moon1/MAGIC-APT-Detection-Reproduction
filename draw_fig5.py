import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25

# X轴映射：0.001在1/8(0.125),0.01在1/4(0.25),0.1在1/2(0.5),1在最右端
def transform_x(fpr):
    fpr = np.asarray(fpr)
    canvas_x = np.zeros_like(fpr)
    mask1 = fpr <= 0.001
    canvas_x[mask1] = fpr[mask1] / 0.001 * 0.125
    mask2 = (fpr > 0.001) & (fpr <= 0.01)
    canvas_x[mask2] = 0.125 + (fpr[mask2] - 0.001) / (0.01 - 0.001) * 0.125
    mask3 = (fpr > 0.01) & (fpr <= 0.1)
    canvas_x[mask3] = 0.25 + (fpr[mask3] - 0.01) / (0.1 - 0.01) * 0.25
    mask4 = fpr > 0.1
    canvas_x[mask4] = 0.5 + (fpr[mask4] - 0.1) / (1 - 0.1) * 0.5
    return canvas_x

# Y轴TPR映射（论文分段变换）
def forward_y(y):
    y = np.asarray(y)
    return np.where(
        y < 0.9,  y * (0.50 / 0.9),
        np.where(
            y < 0.99, 0.50 + (y - 0.9) * (0.25 / 0.09),
            np.where(
                y < 0.999, 0.75 + (y - 0.99) * (0.15 / 0.009),
                0.90 + (y - 0.999) * (0.10 / 0.001)
            )
        )
    )

# 加载数据
data_ss = np.load("roc_result_streamspot.npz")
ytrue_ss, yscore_ss = data_ss["y_true"], data_ss["y_score"]
data_wg = np.load("roc_result_wget.npz")
ytrue_wg, yscore_wg = data_wg["y_true"], data_wg["y_score"]

# ========= 关键改动：手动高密度阈值采样，曲线变平滑 =========
def get_dense_roc(y_true, y_score):
    thresholds = np.linspace(y_score.min(), y_score.max(), num=2000)
    fpr_list = []
    tpr_list = []
    for th in thresholds:
        pred = y_score >= th
        tp = np.sum((pred == 1) & (y_true ==1))
        fn = np.sum((pred ==0) & (y_true ==1))
        fp = np.sum((pred ==1) & (y_true ==0))
        tn = np.sum((pred ==0) & (y_true ==0))
        tpr = tp/(tp+fn) if (tp+fn) >0 else 0
        fpr = fp/(fp+tn) if (fp+tn) >0 else 0
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    return np.array(fpr_list), np.array(tpr_list)

fpr_ss, tpr_ss = get_dense_roc(ytrue_ss, yscore_ss)
auc_ss = auc(fpr_ss, tpr_ss)
fpr_wg, tpr_wg = get_dense_roc(ytrue_wg, yscore_wg)
auc_wg = auc(fpr_wg, tpr_wg)

#坐标转换
x_ss = transform_x(fpr_ss)
y_ss = forward_y(tpr_ss)
x_wg = transform_x(fpr_wg)
y_wg = forward_y(tpr_wg)

fig, ax = plt.subplots(figsize=(6,6), dpi=300)
ax.plot(x_ss, y_ss, "-", color="black", linewidth=1.4, label=f"StreamSpot (AUC={auc_ss:.4f})")
ax.plot(x_wg, y_wg, "--", color="black", linewidth=1.4, label=f"Unicorn Wget (AUC={auc_wg:.4f})")

ax.set_xlim(0,1)
tick_positions_x = [0, 0.125, 0.25, 0.5, 1.0]
tick_labels_x = ["0", "0.001", "0.01", "0.1", "1"]
ax.set_xticks(tick_positions_x)
ax.set_xticklabels(tick_labels_x)
ax.set_xlabel("False Positive Rate, Log Transformed", fontsize=11)

canvas_positions_y = [0, 0.5, 0.75, 0.90, 1.0]
label_text_y = ["0", "0.9", "0.99", "0.999", "1"]
ax.set_yticks(canvas_positions_y)
ax.set_yticklabels(label_text_y)
ax.set_ylim(0,1)
ax.set_ylabel("True Positive Rate, Log Transformed", fontsize=11)

leg = ax.legend(loc="lower left", title="Dataset", frameon=True, fontsize=10)
leg.get_title().set_fontsize(10)
ax.grid(True, which="both", linestyle="-")

plt.tight_layout(pad=1.2)
plt.savefig("fig5_final.png", dpi=300)
plt.close()
print("✅高密度采样绘图代码，运行后曲线会平滑很多")
