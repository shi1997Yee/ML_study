import pandas as pd
import geopandas as gpd
import numpy as np
import math
from shapely.geometry import Point, LineString

"""
# 绑路（Map Matching）是将一串轨迹点与地图上的道路网络相匹配的过程。在使用隐马尔可夫模型（HMM）进行绑路时，轨迹点作为观测状态，而道路网络上的各个路段作为隐藏状态。HMM的核心在于通过计算观测概率（发射概率）和转移概率来确定最可能的道路序列。
# 为了简化问题，我们可以按以下步骤实现一个基本的HMM绑路过程：
# 初始化状态空间：确定地图上所有可能的道路状态。
# 观测概率（发射概率）：计算每个轨迹点与各个道路状态的距离，并根据这个距离计算发射概率。通常，距离越近，发射概率越高。
# 转移概率矩阵：计算每对相邻道路状态之间的转移概率。这通常基于道路间的连通性和距离。
# 初始概率分布：如果有先验信息，可以设置初始概率分布；否则，可以假设所有状态的初始概率相同。
# 应用维特比算法（Viterbi Algorithm）：使用维特比算法找到最可能的隐藏状态序列，即最匹配的道路序列。


# 维特比算法的实现是为了找到整个轨迹中最可能的隐藏状态序列，这个序列对应于最后一个轨迹点的最大概率路径。在每个时间步骤（即轨迹点），算法都会计算到达每个状态的最大概率，并记录下到达每个状态的最佳路径。
# 在最后一步，算法会遍历最后一个轨迹点对应的所有状态的概率，并选择概率最大的状态。然后，通过回溯之前记录的路径，得到从第一个轨迹点到最后一个轨迹点的最可能的状态序列，即最佳匹配路径。
# 维特比算法是一种动态规划方法，它不仅找到了最后一个状态的最大概率，而且还能确保整个序列的概率是最大的，这是通过在每个步骤选择最大概率的转移来实现的。这样，得到的路径是整个轨迹的最佳绑路结果，而不仅仅是最后一个点的结果。
#--------------------
隐马尔可夫模型（Hidden Markov Model，HMM）是一种统计模型，用于描述一个系统随时间进展的状态序列，其中每个状态不可直接观察（因此称为“隐”状态），但可以通过某些观察到的数据间接推断。HMM在语音识别、自然语言处理、生物信息学等领域有广泛应用。
HMM建立在一些关键假设的基础上：
马尔可夫性假设：系统的下一个状态只依赖于当前状态。这意味着状态转移的概率不受之前状态的影响，即遵循一阶马尔可夫性质。对于更高阶的HMM，一个状态可能依赖于前几个状态，但这在实际应用中较少见。
观察独立性假设：给定当前状态，一个观测（输出）与其他观测（输出）相互独立。换句话说，任何时间点的观测只依赖于当前时间点的状态，不依赖于之前或之后的状态或观测。
时间不变的假设：状态转移概率和观测概率是随时间不变的。这意味着无论在序列中的哪个时间点，状态转移概率和给定状态下的观测概率是固定的。
初始状态分布：模型有一个初始状态分布，它是一个概率分布，用于定义序列开始时每个状态的概率。
HMM包含以下几个基本元素：
状态集合：模型中所有可能的隐状态。
观测集合：所有可能的观测值。
状态转移概率矩阵：定义从一个状态转移到另一个状态的概率。
观测概率矩阵（发射概率）：给定隐状态下，观测到某个观测值的概率。
初始状态概率分布：序列开始时每个状态的初始概率。
HMM的这些假设简化了模型的复杂性，使得它能够有效地应用于各种序列数据分析任务。然而，这些假设也限制了HMM的表达能力，使得它可能无法准确地捕捉到现实世界中某些序列数据的复杂依赖关系。在这些情况下，可能需要使用更复杂的模型，如条件随机场（Conditional Random Fields，CRFs）或循环神经网络（Recurrent Neural Networks，RNNs）。
#--------------------
在使用隐马尔可夫模型（HMM）进行轨迹匹配时，观察独立性假设的限制可能会影响模型的性能。轨迹匹配通常涉及将观测到的移动对象轨迹（如车辆或人的GPS轨迹）与地图上的路网进行对齐，以确定移动对象在路网上的最可能路径。在这个过程中，隐状态可能表示移动对象可能所在的路段，而观测则是GPS坐标。
观察独立性假设在轨迹匹配中的限制主要体现在：
空间关联：在实际应用中，连续的GPS观测之间通常存在空间上的关联。例如，车辆的位置通常沿着道路平滑移动，而不是随机跳跃。由于HMM假设在给定当前状态的情况下，每个观测与其他观测相互独立，这可能导致HMM忽略了观测之间的这种空间关联性。
观测误差：GPS观测数据通常包含噪声和误差，这些误差可能依赖于多种因素，如城市峡谷效应、多路径效应或信号遮挡。这些误差并非完全独立，而可能与周围环境和先前的观测相关。HMM可能无法充分考虑这种误差的相关性。
环境变化：路网环境的变化，如临时道路封闭或交通拥堵，可能会影响轨迹的观测数据。这些变化可能导致观测数据与HMM模型的假设不匹配，因为模型假设观测概率是时间不变的。
尽管存在这些限制，HMM依然是轨迹匹配中常用的一种方法，因为它在许多情况下仍然能够提供良好的匹配结果。为了克服观察独立性假设的限制，研究者们提出了一些方法，例如：
数据预处理：通过去除噪声和平滑轨迹数据，减少观测误差对匹配结果的影响。
模型改进：引入更高阶的依赖，例如使用粒子滤波（Particle Filter）或条件随机场（CRFs）来建模观测之间的关联。
结合地图信息：利用地图信息和道路网络结构来引导轨迹匹配过程，从而减少观测独立性假设的影响。
通过这些方法，可以在一定程度上缓解HMM在轨迹匹配中由于观察独立性假设带来的限制，并提高匹配准确度。
"""


def get_two_point_angle(p1, p2):
    """
    计算p1到p2的角度
    """
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    degree = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle = (360 + degree) % 360
    return angle


def get_angle_on_link(ls, p):
    """
    给定线外任意一点，计算点在线上映射位置的角度
    """
    proj = ls.project(p)  # 映射位置
    sta_p, end_p = ls.interpolate(proj - 1e-5), ls.interpolate(proj + 1e-5)  # 映射点扩buffer，计算该buffer的角度
    angle = get_two_point_angle(sta_p, end_p)
    return angle


def calculate_angle_difference(angle1, angle2):
    """计算两个角度之间的差异，结果在[-180, 180]之间"""
    diff = (angle1 - angle2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def gaussian_probability(x, sigma):
    """计算高斯分布的概率密度"""
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * (x / sigma) ** 2)


def calculate_trans_prob(roads, neib_road):
    trans_prob = {}
    for link_id in roads:
        trans_prob[link_id] = {}
        sum_weight = 0
        cur = neib_road[neib_road['link_id'] == link_id].iloc[0]
        for trans_id in roads:
            trans = neib_road[neib_road['link_id'] == trans_id].iloc[0]
            if link_id == trans_id:
                weight = 0.5
            elif cur.enodeid == trans.snodeid:
                weight = 0.5
            else:
                weight = 0
            trans_prob[link_id][trans_id] = weight
            sum_weight += weight
        if sum_weight != 0:
            for roadid in trans_prob[link_id]:
                trans_prob[link_id][roadid] /= sum_weight
    return trans_prob


def calculate_emission_prob(road_gdf, point_gdf, sigma_distance, sigma_angle):
    emission_prob = {}
    for i in range(point_gdf.shape[0]):
        row_i = point_gdf.iloc[i]
        pointid = row_i['pid']
        emission_prob[pointid] = {}
        total_weight = 0
        for j in range(road_gdf.shape[0]):
            row_j = road_gdf.iloc[j]
            roadid = row_j['link_id']
            distance = row_j['geometry'].distance(row_i['geometry']) * 1e5
            link_angle = get_angle_on_link(row_j['geometry'], row_i['geometry'])
            p_angle = row_i['pt_angle']
            angle_diff = calculate_angle_difference(p_angle, link_angle)
            if angle_diff > 90:
                weight = 1e-10
            else:
                # 使用高斯分布计算基于距离的发射概率
                prob_distance = gaussian_probability(distance, sigma_distance)
                # 使用高斯分布计算基于角度差异的发射概率
                prob_angle = gaussian_probability(angle_diff, sigma_angle)
                weight = prob_distance * prob_angle
            emission_prob[pointid][roadid] = weight
            total_weight += weight
        # 归一化概率
        for roadid in emission_prob[pointid]:
            emission_prob[pointid][roadid] /= total_weight
    return emission_prob


# 维特比算法
def viterbi(trajectory, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    # 初始化初始状态的概率和路径
    for st in states:
        V[0][st] = start_p[st] * emit_p[trajectory[0]][st]
        path[st] = [st]
    # 对于轨迹中的每一个点，计算每个状态的最大概率和路径
    for t in range(1, len(trajectory)):
        V.append({})
        newpath = {}

        for st in states:
            (prob, state) = max((V[t - 1][st0] * trans_p[st0][st] * emit_p[trajectory[t]][st], st0) for st0 in states)
            V[t][st] = prob  # 轨迹的第t个点被路段st观测到的概率
            newpath[st] = path[state] + [st]

        # 不需要保留旧路径
        path = newpath

    # 返回最大概率的路径
    (prob, state) = max((V[t][st], st) for st in states)
    return (prob, path[state])


if __name__ == "__main__":
    road_df = gpd.read_file("road.geojson")
    img_df = gpd.read_file("img.geojson")
    img_df['pid'] = img_df['pid'].map(str)
    img_df['commandid'] = img_df['commandid'].map(str)
    test_img = img_df[img_df['commandid'] == '1111']
    sort_img = test_img.sort_values(by="time")  # 轨迹按照采集时间排序
    trail = LineString(list(sort_img['geometry']))
    road_sp = road_df.sindex
    neibs = road_sp.query(trail.buffer(30e-5), predicate="intersects")  # 轨迹附近的路网
    neib_road = road_df.iloc[neibs]
    roads = list(neib_road['link_id'])  # 道路状态
    points = list(sort_img['pid'])
    trans_prob = calculate_trans_prob(roads, neib_road)  # 状态转移概率
    emission_prob = calculate_emission_prob(neib_road, sort_img, 5, 90)  # 发射概率

    # 初始概率分布
    eva_prob = 1 / len(roads)
    initial_prob = {}
    for i in range(len(roads)):
        initial_prob[roads[i]] = eva_prob

    print(emission_prob)
    # 应用维特比算法得到最可能的路径
    prob, matched_path = viterbi(points, roads, initial_prob, trans_prob, emission_prob)
    print(f"Probability of the best path: {prob}")
    print(f"Best matched path: {matched_path}")

    sort_img['bind_roadid'] = matched_path  # 绑路结果
