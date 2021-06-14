from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,precision_recall_curve

X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
score = clf.predict_proba(X)[:, 1]
a = roc_auc_score(y, score)
k = clf.decision_function(X)
b = roc_auc_score(y, k)
print(b)
