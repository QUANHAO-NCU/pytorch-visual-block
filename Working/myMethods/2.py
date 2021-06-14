import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

label = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
scores = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=1)

print('FPR:', fpr)
print('TPR:', tpr)
print('thresholds:', thresholds)
print('precision:', metrics.precision_score(label, scores))
print('recall:', metrics.recall_score(label, scores))
print('ROC:', metrics.roc_auc_score(label, scores))
plt.plot(fpr, tpr)
plt.show()
