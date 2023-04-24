# Neural Network
# 合ってるかどうかは分からないが，作ってみた．
# NN_2D() or NN_3D() を実行可能
#

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

def import_data(path="input.csv"):
    df = pd.read_csv(path)
    ev = df.drop(["target"], axis=1).values
    target = df['target'].values
    return ev, target

def forward(ev, weight):
    # Old version code
    # tmp = 0
    # for i in range(len(weight)):
    #     tmp += ev[:, i] * weight[i]
    tmp = np.dot(ev, weight)
    return tmp

def judgment(summation):
    return np.where(summation >= 0, 1, 0)

def update_weight(weight, ev, judged, target, param, alpha=1):
    param[0] += 1
    param[1] = alpha
    tmp = alpha * ((target - judged)[:, None]) * ev
    tmp = np.sum(tmp, axis=0)
    updated_weight = weight + tmp
    return updated_weight, param

def plot(x, y, target, judged, weight):
    tmp = target - judged
    accuracy = (len(tmp) - np.count_nonzero(tmp)) / len(tmp)

    plt.scatter(x, y, c=target, cmap=plt.cm.get_cmap('bwr'), s=100)

    y = (-weight[0] - weight[1]*x)/weight[2]
    plt.plot(x, y, color='#000000', linewidth=0.5)
    plt.title(f'Accuracy: {accuracy:.0%}', fontsize=30)
    plt.show()

def plot_3d(x, y, z, target, judged, weight, param):
    tmp = target - judged
    accuracy = (len(tmp) - np.count_nonzero(tmp)) / len(tmp)
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c=target, cmap=plt.cm.get_cmap('bwr'), s=50)

    x, y = np.mgrid[-1:2, -1:2]

    z = (-weight[0] - weight[1]*x - weight[2]*y)/weight[3]
    ax.plot_surface(x, y, z, alpha=0.3, color="black")
    plt.title(f'Accuracy: {accuracy:.0%}', fontsize=30)
    fig.text(
        0.05, 0.02, f'weight: {weight}, itr: {param[0]}, alpha: {param[1]}, accuracy: {accuracy:.0%}', fontsize=6)

    plt.savefig("PLOT_3D.jpg")
    plt.show()

def NN_3D(file_name, itr=30, weight=[-0.5, 0.7, 0.7, 0.1], param=[0, 0], alpha=0.01):
    # フィルの読み込み
    ev, target = import_data(file_name)
    # 閾値をノードに追加
    ev = np.insert(ev, 0, 1, axis=1)

    for i in range(itr):
        # 前向き学習
        summation = forward(ev, weight)
        # 学習結果の判定
        judged = judgment(summation)
        # 重みの更新
        weight, param = update_weight(weight, ev, judged, target, param, alpha=alpha)
        # 図の作成と保存
        plot_3d(ev[:, 1], ev[:, 2], ev[:, 3], target, judged, weight, param)
    plot_3d(ev[:, 1], ev[:, 2], ev[:, 3], target, judged, weight, param)

def NN_2D(file_name, itr=30, weight=[-0.5, 0.7, 0.3], param=[0, 0], alpha=0.01):
    ev, target = import_data(file_name)

    ev = np.insert(ev, 0, 1, axis=1)
    for i in range(itr):
        summation = forward(ev, weight)

        judged = judgment(summation)

        weight, param = update_weight(weight, ev, judged, target, param, alpha=alpha)
        plot(ev[:, 1], ev[:, 2], target, judged, weight)

input_file_name = "https://raw.githubusercontent.com/yoriku/tmp_csv/main/input.csv"
kadai_file_name = "https://raw.githubusercontent.com/yoriku/tmp_csv/main/kadai.csv"
NN_3D(kadai_file_name)
