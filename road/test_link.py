import numpy as np
import math
from shapely.geometry import Point, LineString, GeometryCollection
from shapely.wkt import loads
from scipy import signal


def angle_func1(point, angle, length):
    """
    给定一个点和角度，得到指定长度的线段
    :param point: Point类型
    :param angle: 0-360
    example: angle_func1(Point(1,1), 45, 2)
    """
    x1 = point.x
    y1 = point.y
    x2 = x1 + length * math.cos(math.radians(angle))
    y2 = y1 + length * math.sin(math.radians(angle))
    return LineString([[x1, y1], [x2, y2]]).wkt


def angle_func2(point1, point2):
    """
    获取两点所构成的有向边的角度(以正东为基准的逆时针角度)
    example: angle_func2(Point(1,1), Point(2,2))
    """
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))  #[-180, 180]
    angle = (360 + angle) % 360  #[0, 360]
    return angle


def angle_func3(line1, line2):
    """
    返回line2相对于line1顺时针角度[0, 360]
    line1 = LineString([[0, 0], [1, 1]])
    line2 = LineString([[0, 1], [0, 2]])
    angle_func3(line1, line2)
    """
    x1, x2 = line1.coords[0][0], line1.coords[-1][0]
    y1, y2 = line1.coords[0][1], line1.coords[-1][1]

    x3, x4 = line2.coords[0][0], line2.coords[-1][0]
    y3, y4 = line2.coords[0][1], line2.coords[-1][1]

    angle1 = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle2 = math.degrees(math.atan2(y4 - y3, x4 - x3))

    angle = (360 + angle1 - angle2) % 360
    return angle


def frechet_distance(P, Q):
    """
    弗雷彻距离(Frechet distance)/狗绳距离
    link1 = loads('LINESTRING (0 0, 1 1, 2 2)')
    link2 = loads('LINESTRING (0 1, 1 3)')
    P = np.array(link1.coords)
    Q = np.array(link2.coords)
    frechet_distance(P, Q)
    """
    m, n = len(P), len(Q)
    memo = np.zeros((m, n))
    memo[0][0] = np.linalg.norm(P[0] - Q[0])

    for i in range(1, m):
        memo[i][0] = max(memo[i - 1][0], np.linalg.norm(P[i] - Q[0]))

    for j in range(1, n):
        memo[0][j] = max(memo[0][j - 1], np.linalg.norm(P[0] - Q[j]))

    for i in range(1, m):
        for j in range(1, n):
            memo[i][j] = max(min(memo[i - 1][j], memo[i][j - 1], memo[i - 1][j - 1]), np.linalg.norm(P[i] - Q[j]))
    return memo[m - 1][n - 1]


def smooth_linestring():
    # 使用低通滤波器平滑曲线
    link1 = LineString([[1, 1], [2, 2], [3, 5]])
    ori_ls = LineString(link1['coordinates'])
    x = np.array(ori_ls.coords)[:, 0]
    y = np.array(ori_ls.coords)[:, 1]
    b, a = signal.butter(2, 0.08, btype='lowpass')  # 低通滤波器
    smoothed_y = signal.filtfilt(b, a, y, padlen=len(y) - 1)  # 应用滤波器
    new_ls = LineString(list(zip(x, smoothed_y)))
    print(new_ls.wkt)
    GeometryCollection([ori_ls, new_ls])


if __name__ == '__main__':
    res = angle_func1(Point(1, 1), 45, 2)  # LINESTRING (1 1, 2.414213562373095 2.414213562373095)
    print(res)
    res = angle_func2(Point(1, 1), Point(0, 0))  # -135.0
    print(res)


