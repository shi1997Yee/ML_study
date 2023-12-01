import numpy as np
import math
from shapely.geometry import Point, LineString


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


if __name__ == '__main__':
    res = angle_func1(Point(1, 1), 45, 2) # LINESTRING (1 1, 2.414213562373095 2.414213562373095)
    print(res)
    res = angle_func2(Point(1, 1), Point(0, 0)) # -135.0
    print(res)
