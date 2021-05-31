# coding=utf-8
import random

from matplotlib import pyplot as plt
from numpy import *


def Euclidean_distance(coordinate1, coordinate2):
    return math.sqrt((coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2)


def anti_Euclidean_distance(coordinate1, coordinate2):
    distance = math.sqrt((coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2)
    if distance != 0:
        return 1 / distance
    return float('inf')


def HPPP_Rectangle(Lambda, width, height, ISD=0.0):
    M = Lambda
    xx = []
    yy = []
    while M > 0:
        x_new = random.uniform(0, width)
        y_new = random.uniform(0, height)
        add_new_point = True
        for x, y in zip(xx, yy):
            dist = Euclidean_distance([x_new, y_new], [x, y])
            if dist <= ISD:
                add_new_point = False
                break
        if add_new_point:
            M -= 1
            xx.append(x_new)
            yy.append(y_new)
    return xx, yy


def HPPP_Circle(Lambda, radius, center_coordinate, ISD=0.0, SDC=0.0):
    M = Lambda
    xx = []
    yy = []
    center_x, center_y = center_coordinate
    while M > 0:
        radius_new = random.uniform(SDC, radius)
        radians_new = random.uniform(0, 360)
        x_new = center_x + radius_new * math.cos(math.radians(radians_new))
        y_new = center_y + radius_new * math.sin(math.radians(radians_new))
        add_new_point = True
        for x, y in zip(xx, yy):
            dist = Euclidean_distance([x_new, y_new], [x, y])
            if dist <= ISD:
                add_new_point = False
                break
        if add_new_point:
            M -= 1
            xx.append(x_new)
            yy.append(y_new)
    return xx, yy


def generate(cluster_num=3, point_num=1000, width=500, height=500):
    cluster_center = []
    M = cluster_num
    while M > 0:
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        if [x, y] not in cluster_center:
            cluster_center.append([x, y])
            M -= 1
    cluster_average = point_num * 0.7 / cluster_num
    uniform_distribute = point_num * (1 - 0.7)
    radius = int(sqrt(width ** 2 + height ** 2) / (cluster_num + 1))
    xx = []
    yy = []
    for i in range(cluster_num):
        xx_i, yy_i = HPPP_Circle(cluster_average, radius, cluster_center[i])
        xx += xx_i
        yy += yy_i
    xx_a, yy_a = HPPP_Rectangle(uniform_distribute, width, height)
    xx += xx_a
    yy += yy_a
    return xx, yy


def write(xx, yy, file_name):
    with open(file_name, mode='w+') as f:
        for index, x_c in enumerate(xx):
            f.write(str(x_c) + '\t' + str(yy[index]))
            f.write('\n')


def K_Means(data, cluster_num=3, cluster_centers=None, iteration=5, distance_function=None):
    # 二维坐标下K-Means聚类
    # 不指定初始聚类坐标的情况下属于K-Means++
    # step 1 聚类中心的选取
    # step 2 迭代，修正聚类中心和聚类形式
    # step 3 返回聚类结果
    # data[[x_coordinate,y_coordinate],[...],...]
    if not distance_function:
        distance_function = Euclidean_distance
    # step 1 聚类中心的选取
    if not cluster_centers:
        cluster_centers = []
        # 随机选取一个点作为候选的聚类中心
        random_init = random.randint(0, len(data))
        cluster_centers.append(data[random_init])
        # 再从所有点中，取一个点，使其到所有候选中心距离和最大。这个点作为新的聚类中心，加入候选聚类中心
        for index in range(cluster_num - 1):
            max_distance = 0
            next_center = []
            for point in data:
                # 计算该点到所有候选中心的距离
                tmp_distance = 0
                for candidate_center in cluster_centers:
                    tmp_distance += distance_function(point, candidate_center)
                if tmp_distance > max_distance:
                    next_center = point
                    max_distance = tmp_distance
            cluster_centers.append(next_center)
    else:
        assert cluster_num == len(cluster_centers)
    # step 2 迭代，修正聚类中心和聚类形式
    while iteration > 0:
        # 初始化结果容器
        clusters = [[] for i in range(cluster_num)]
        # 每一个点加入与其最近的一个聚类中心
        for point in data:
            min_distance = distance_function(point, cluster_centers[0])
            cluster_center_index = 0
            for index, cluster_center_point in enumerate(cluster_centers):
                tmp_distance = distance_function(point, cluster_center_point)
                if tmp_distance < min_distance:
                    cluster_center_index = index
                    min_distance = tmp_distance
            clusters[cluster_center_index].append(point)
        # 重新计算聚类中心,更新聚类中心
        # 新的聚类中心坐标为：x为当前聚类所有点的x的均值（Means），y同理。
        for index, cluster in enumerate(clusters):
            x_sum = y_sum = 0
            point_num = len(cluster)
            for point in cluster:
                x_sum += point[0]
                y_sum += point[1]
            x_average = x_sum / point_num
            y_average = y_sum / point_num
            cluster_centers[index] = [x_average, y_average]
        # 完成一轮迭代
        iteration -= 1
    # step 3 返回聚类结果,包括分好的簇和聚类中心
    return clusters, cluster_centers


def draw_cluster(clusters, cluster_centers, color=None, ax=None):
    # 将就着画画吧
    if color:
        assert len(color) == len(cluster_centers)
    else:
        # color = ['aliceblue',
        #          'antiquewhite',
        #          'aqua',
        #          'aquamarine',
        #          'azure',
        #          'beige',
        #          'bisque',
        #          'black',
        #          'blanchedalmond',
        #          'blue',
        #          'blueviolet',
        #          'brown',
        #          'burlywood',
        #          'cadetblue',
        #          'chartreuse',
        #          'chocolate',
        #          'coral',
        #          'cornflowerblue',
        #          'cornsilk',
        #          'crimson',
        #          'cyan',
        #          'darkblue',
        #          'darkcyan',
        #          'darkgoldenrod',
        #          'darkgray',
        #          'darkgreen',
        #          'darkkhaki',
        #          'darkmagenta',
        #          'darkolivegreen',
        #          'darkorange',
        #          'darkorchid',
        #          'darkred',
        #          'darksalmon',
        #          'darkseagreen',
        #          'darkslateblue',
        #          'darkslategray',
        #          'darkturquoise',
        #          'darkviolet',
        #          'deeppink',
        #          'deepskyblue',
        #          'dimgray',
        #          'dodgerblue',
        #          'firebrick',
        #          'floralwhite',
        #          'forestgreen',
        #          'fuchsia',
        #          'gainsboro',
        #          'ghostwhite',
        #          'gold',
        #          'goldenrod',
        #          'gray',
        #          'green',
        #          'greenyellow',
        #          'honeydew',
        #          'hotpink',
        #          'indianred',
        #          'indigo',
        #          'ivory',
        #          'khaki',
        #          'lavender',
        #          'lavenderblush',
        #          'lawngreen',
        #          'lemonchiffon',
        #          'lightblue',
        #          'lightcoral',
        #          'lightcyan',
        #          'lightgoldenrodyellow',
        #          'lightgreen',
        #          'lightgray',
        #          'lightpink',
        #          'lightsalmon',
        #          'lightseagreen',
        #          'lightskyblue',
        #          'lightslategray',
        #          'lightsteelblue',
        #          'lightyellow',
        #          'lime',
        #          'limegreen',
        #          'linen',
        #          'magenta',
        #          'maroon',
        #          'mediumaquamarine',
        #          'mediumblue',
        #          'mediumorchid',
        #          'mediumpurple',
        #          'mediumseagreen',
        #          'mediumslateblue',
        #          'mediumspringgreen',
        #          'mediumturquoise',
        #          'mediumvioletred',
        #          'midnightblue',
        #          'mintcream',
        #          'mistyrose',
        #          'moccasin',
        #          'navajowhite',
        #          'navy',
        #          'oldlace',
        #          'olive',
        #          'olivedrab',
        #          'orange',
        #          'orangered',
        #          'orchid',
        #          'palegoldenrod',
        #          'palegreen',
        #          'paleturquoise',
        #          'palevioletred',
        #          'papayawhip',
        #          'peachpuff',
        #          'peru',
        #          'pink',
        #          'plum',
        #          'powderblue',
        #          'purple',
        #          'red',
        #          'rosybrown',
        #          'royalblue',
        #          'saddlebrown',
        #          'salmon',
        #          'sandybrown',
        #          'seagreen',
        #          'seashell',
        #          'sienna',
        #          'silver',
        #          'skyblue',
        #          'slateblue',
        #          'slategray',
        #          'snow',
        #          'springgreen',
        #          'steelblue',
        #          'tan',
        #          'teal',
        #          'thistle',
        #          'tomato',
        #          'turquoise',
        #          'violet',
        #          'wheat',
        #          'white',
        #          'whitesmoke',
        #          'yellow',
        #          'yellowgreen']
        # random.shuffle(color)
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    if not ax:
        ax = plt
    cluster_points_coordinates = [[[], []] for i in range(len(clusters))]
    for index, cluster in enumerate(clusters):
        for point in cluster:
            cluster_points_coordinates[index][0].append(point[0])
            cluster_points_coordinates[index][1].append(point[1])
    for index, points_coordinate in enumerate(cluster_points_coordinates):
        ax.scatter(points_coordinate[0], points_coordinate[1], color=color[index], s=5)
        cluster_center = cluster_centers[index]
        ax.scatter(cluster_center[0], cluster_center[1],
                   color=color[index + len(cluster_centers)], s=36)
    ax.show()


if __name__ == '__main__':
    data = []
    xx, yy = generate(cluster_num=3, point_num=1000)
    for index, x_coordinate in enumerate(xx):
        data.append([x_coordinate, yy[index]])
    print(f'all points {len(data)}')
    clusters, cluster_centers = K_Means(data, cluster_num=3)
    for i, cluster in enumerate(clusters):
        print(f'cluster{i},point num{len(cluster)}')
    print('K-Means done')
    draw_cluster(clusters, cluster_centers)
