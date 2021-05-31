import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

sns.set()
from sklearn.datasets._samples_generator import make_circles
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV


# x, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.6)
# xfit = np.linspace(-1, 3.5)

# plt.scatter(x[:, 0], x[:, 1], c=y, s=120, cmap='autumn')
# for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.8, 0.2)]:
#     yfit = m * xfit + b
# plt.plot(xfit, yfit, '-k')
# plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
# plt.xlim(-1, 3.5)
# model = SVC(kernel='linear')
# model.fit(x, y)


def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax == None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=3, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_svm(N=10, ax=None):
    # x, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
    x, y = make_circles(n_samples=100, factor=0.5, noise=0.5)
    X = x[:N]
    Y = y[:N]
    # model = SVC(kernel='linear')
    model = SVC(kernel='rbf', C=1e5)
    model.fit(X, Y)
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=Y, s=N, cmap='autumn')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plot_svc_decision_function(model, ax)


# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# for axi, N in zip(ax, [60, 120]):
#     plot_svm(N, axi)
#     axi.set_title(f'N={N}')
# # plot_svc_decision_function(model)
# plt.show()


# x, y = make_circles(n_samples=100, factor=0.1, noise=0.1)
# r = np.exp(-(x ** 2).sum(1))
#
#
# def polt_3D(elev=30, azim=30, x=x, y=y):
#     ax = plt.subplot(projection='3d')
#     ax.scatter3D(x[:, 0], x[:, 1], r, c=y, s=100, cmap='autumn')
#     ax.view_init(elev=elev, azim=azim)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('r')
#     plt.show()
#
# polt_3D(elev=45, azim=45, x=x, y=y)

def detect_face():
    faces = fetch_lfw_people(min_faces_per_person=60)

    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    xTrain, xTest, yTrain, yTest = train_test_split(faces.data, faces.target, random_state=40)
    param_grid = {'svc__C': [1, 5, 10],
                  'svc__gamma': [0.0001, 0.0005, 0.001]}
    grid = GridSearchCV(model, param_grid)
    grid.fit(xTrain, yTrain)
    model = grid.best_estimator_
    yfit = model.predict(xTest)
    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(xTest[i].reshape(62, 47), cmap='bone')
        axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                       color='black' if yfit[i] == yTest[i] else 'red')
    fig.suptitle('predicted Names: Incorrect Labels in Red', size=14)
    plt.show()


if __name__ == '__main__':
    detect_face()
