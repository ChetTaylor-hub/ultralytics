import numpy as np

def calculate_nAP(gt_points, predicted_points, delta, k):
    binary_list = []
    for p_j in predicted_points:
        # 计算预测点与所有GT点之间的欧氏距离
        distances = np.linalg.norm(gt_points - p_j, axis=1)

        # 对距离进行排序并取前k个最近的距离
        sorted_distances = np.sort(distances)[:k]

        # 计算d_kNN(p_i)
        d_knn_pi = np.mean(sorted_distances)

        # 判断预测点是否为TP
        if np.min(distances) / d_knn_pi < delta:
            binary_list.append(1)  # TP
        else:
            binary_list.append(0)  # FP

    # 计算nAP
    nAP = np.mean(binary_list)

    return nAP

# 示例数据：GT点和预测点
gt_points = np.array([[2,3], [5,6], [8,9]])  # Ground Truth Points
predicted_points = np.array([[4,5], [6,7], [8,9]])  # Predicted Points

# 阈值δ和k的值
delta = 0.25
k = 3

# 计算nAP
nAP = calculate_nAP(gt_points, predicted_points, delta, k)

print(f"Normalized Accuracy Precision (nAP): {nAP}")
