import numpy as np
import math
from matplotlib import pyplot as plt

# 绘制样例
# SVM，SVDD
feature_space_range = [8, 8]

center_p = positive_samples_center = [4, 4]

center_n = negative_samples_center = [6, 7]
center_n2 = negative_samples_center2 = [1, 2]

points_num_positive = 10
samples_points_x_positive = np.random.uniform(center_p[0] - 0.5, center_p[0] + 0.5, size=points_num_positive)
samples_points_y_positive = np.random.uniform(center_p[1] - 0.5, center_p[1] + 0.5, size=points_num_positive)

points_num_negative = 8
samples_points_x_negative = np.random.uniform(center_n[0] - 1.25, center_n[0] + 1.25, size=points_num_negative)
samples_points_y_negative = np.random.uniform(center_n[1] - 1.25, center_n[1] + 1.25, size=points_num_negative)

points_num_negative2 = 7
samples_points_x_negative2 = np.random.uniform(center_n2[0] - 1.25, center_n2[0] + 1.25, size=points_num_negative2)
samples_points_y_negative2 = np.random.uniform(center_n2[1] - 1.25, center_n2[1] + 1.25, size=points_num_negative2)


def SVMBoundary(x):
    return -1 * x + center_p[0] - 1.25


def get_ellipse(e_x, e_y, a, b, e_angle):
    """
    https://blog.csdn.net/XMZhu1992/article/details/89972653
    输入参数依次为（椭圆中心x轴坐标，椭圆中心y轴坐标，半长轴长度，半短轴长度，椭圆长轴与x轴夹角（deg））
    """
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * math.cos(angles)
        or_y = b * math.sin(angles)
        length_or = math.sqrt(or_x * or_x + or_y * or_y)
        or_theta = math.atan2(or_y, or_x)
        new_theta = or_theta + e_angle / 180 * math.pi
        new_x = e_x + length_or * math.cos(new_theta)
        new_y = e_y + length_or * math.sin(new_theta)
        x.append(new_x)
        y.append(new_y)
    return x, y


SVMBoundary_line_x = np.linspace(0, center_p[0] + 0.5 * 6, 200)
SVMBoundary_line_y = np.linspace(0, center_p[0] + 0.5 * 6, 200) * (-1) + center_p[0] + 0.5 * 6

SVDDBoundary_x, SVDDBoundary_y = get_ellipse(center_p[0], center_p[1], 0.7, 0.8, 45)
# 预设值
plt.title('OC-SVM')
plt.xlabel('(a)')
plt.xlim((0, 8))
plt.ylim((0, 8))
plt.scatter(samples_points_x_positive, samples_points_y_positive, color='r')
plt.scatter(samples_points_x_negative, samples_points_y_negative, color='g')
plt.scatter(samples_points_x_negative2, samples_points_y_negative2, color='b', marker='x')
plt.plot(SVMBoundary_line_x, SVMBoundary_line_y, linestyle='--')
plt.savefig('SVM.png')
plt.show()

# 预设值
plt.title('SVDD')
plt.xlabel('(b)')
plt.xlim((0, 8))
plt.ylim((0, 8))
plt.scatter(samples_points_x_positive, samples_points_y_positive, color='r')
plt.scatter(samples_points_x_negative, samples_points_y_negative, color='g')
plt.scatter(samples_points_x_negative2, samples_points_y_negative2, color='b', marker='x')
plt.plot(SVDDBoundary_x, SVDDBoundary_y, linestyle='--')
plt.savefig('SVDD.png')
plt.show()
