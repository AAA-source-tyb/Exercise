import numpy as np

H = np.array([[0.0, -1.0], [-1.0, 0.0]])

# 对角化得到本征值和本征向量
e, c = np.linalg.eigh(H)

print("轨道能量 (单位: β):")
print(e)

print("\n分子轨道系数:")
print(c)

# 计算π电子总能量（乙烯有2个π电子占据最低轨道）
total_energy = 2 * e[0]
print(f"\nπ电子总能量: {total_energy} β")

# 分析轨道占据
print("\n轨道分析:")
for i, energy in enumerate(e):
    if i < 2:  # 前2个轨道被占据
        print(f"轨道 {i+1}: 能量 = {energy:.3f} β (占据)")
    else:
        print(f"轨道 {i+1}: 能量 = {energy:.3f} β (空)")