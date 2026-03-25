import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1. 录入图片中的数据
# 每一行是一个样本，每一列代表 Class 0, Class 1, Class 2
y_true = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0]
])

y_score = np.array([
    [0.1, 0.2, 0.7],
    [0.1, 0.6, 0.3],
    [0.5, 0.2, 0.3],
    [0.1, 0.1, 0.8],
    [0.4, 0.2, 0.4],
    [0.6, 0.3, 0.1],
    [0.4, 0.2, 0.4],
    [0.4, 0.1, 0.5],
    [0.1, 0.1, 0.8],
    [0.1, 0.8, 0.1]
])

n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()

# 2. 分别计算每个类别的 ROC 和 AUC (One-vs-Rest 思想)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 3. 计算微平均 (Micro-average) ROC 和 AUC（第4条曲线）
# .ravel() 的作用就是把二维矩阵“拍扁”成一维数组
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# 4. 开始绘制图表
plt.figure(figsize=(8, 6))

# 画第4条平均曲线（加粗，使用点线）
plt.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
         color='deeppink', linestyle=':', linewidth=4)

# 画前3条独立类别的曲线
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

# 画对角线（随机猜测线）
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# 设置图表格式
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Multi-class ROC Curve (Assignment 2)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 保存图片并显示
plt.tight_layout()
plt.savefig('multiclass_roc.png')
plt.show()

# 打印结果供报告使用
print("=== AUC 计算结果 ===")
for i in range(n_classes):
    print(f"Class {i} AUC: {roc_auc[i]:.3f}")
print(f"Micro-average AUC: {roc_auc['micro']:.3f}")