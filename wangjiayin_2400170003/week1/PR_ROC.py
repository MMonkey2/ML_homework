import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

#1.数据准备
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
y_score = np.array([0.90, 0.42, 0.20, 0.60, 0.50, 0.41, 0.70, 0.40, 0.65, 0.35])

#2.按预测分数降序排序
sorted_indices = np.argsort(-y_score)
y_true_sorted = y_true[sorted_indices]
y_score_sorted = y_score[sorted_indices]

P = np.sum(y_true)  #正样本数(4)
N = len(y_true) - P #负样本数(6)

tpr_list, fpr_list = [0.0], [0.0]
precision_list, recall_list = [1.0], [0.0]#PR曲线起点设为(Recall=0,Precision=1)
tp, fp = 0, 0

print("=== 机器学习作业：ROC、PR与AUC计算 ===")
print(f"样本总数：{len(y_true)} (正样本: {P}个, 负样本: {N}个)\n")
print("--- 手动计算详细过程 ---")

for i in range(len(y_true_sorted)):
    if y_true_sorted[i] == 1:
        tp += 1
    else:
        fp += 1
        
    tpr = tp / P
    fpr = fp / N
    recall = tp / P
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)
    precision_list.append(precision)
    recall_list.append(recall)
    
    print(f"阈值 >= {y_score_sorted[i]:.2f}: TPR={tpr:.2f}, FPR={fpr:.2f} | Precision={precision:.2f}, Recall={recall:.2f}")

#使用梯形法计算AUC
auc_roc_manual = auc(fpr_list, tpr_list)

print(f"\n--- 最终结果 ---")
print(f"计算得出的 ROC AUC 值：{auc_roc_manual:.3f}")

#3.绘制ROC和PR曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#绘制ROC曲线
ax1.plot(fpr_list, tpr_list, color='darkorange', marker='o', lw=2, label=f'ROC curve (AUC = {auc_roc_manual:.3f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate (FPR)')
ax1.set_ylabel('True Positive Rate (TPR)')
ax1.set_title('ROC Curve')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

#绘制PR曲线
ax2.plot(recall_list, precision_list, color='blue', marker='s', lw=2, label='PR curve')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall (PR) Curve')
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()