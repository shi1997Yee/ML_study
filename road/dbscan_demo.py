import random
from shapely.geometry import Point
import math
import random
from sklearn.cluster import DBSCAN


def metric_func(points, i):
    # 度量函数
    min_dis = 2.0
    x1, y1 = points[i][0], points[i][1]
    neib = []
    for j in range(len(points)):
        x2, y2 = points[j][0], points[j][1]
        if math.sqrt(math.pow(x2-x1, 2) + math.pow(y2-y1, 2)) <= min_dis:
            neib.append(j)
    return neib


if __name__ == '__main__':
    neib_list = []
    ker_list = []
    min_pts = 2

    points = [(0, 0), (0, 1), (1, 1), (0, 5)]
    cnts = len(points)
    theta = set([i for i in range(cnts)])  # 未被访问的
    cluser = [-1 for i in range(cnts)]  # 类别
    # 1、找核心点
    for i in range(cnts):
        neibs = metric_func(points, i)
        neib_list.append(neibs)
        if len(neibs) >= min_pts:
            ker_list.append(i)
    ker_list = set(ker_list)
    print("ker_list", ker_list)
    # 2、遍历核心点，聚合
    k = -1  # 类
    while len(ker_list) > 0:
        random_j = random.choice(list(ker_list))
        q_list = [random_j]
        theta.remove(random_j)
        find = set()
        k += 1
        while len(q_list) > 0:
            q = q_list[0]
            q_list.remove(q)
            ker_list.remove(q)
            find.add(q)
            q_neibs = set(neib_list[q]) & theta
            for item in q_neibs:
                print("item", item)
                q_list.append(item)
                theta.remove(item)
        for f in find:
            cluser[f] = k

    print(cluser)