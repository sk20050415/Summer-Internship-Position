import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csv_file_path = 'trajectory_preci.csv'  
data = pd.read_csv(csv_file_path)

posx = data['X']
posy = data['Y']
posz = data['Z']

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制轨迹
ax.plot(posx, posy, posz, color='black', linestyle='-', linewidth=1.5)

# 设置轴标签
ax.set_xlabel('x/mm')
ax.set_ylabel('y/mm')
ax.set_zlabel('z/mm')

# 保存图像并显示
plt.savefig('trajectory1.png',dpi=300)
plt.show()
