from matplotlib import pyplot as plt
import numpy as np
# plt.style.use('seaborn')  # 使用ggplot样式

colors = ['skyblue', 'indianred', 'lightgreen', 'thistle', 'sienna']
# 使用预定义的样式
plt.style.use('seaborn-darkgrid')
# k-shot , sigma
# x: sigma
# y: acc
# legend: k-shot
# 用来优化每个柱状图的文本标签位置的函数
def autolabel(ax,rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
# 创建柱状图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
labels = ['5-shot', '10-shot', '20-shot']
x = np.arange(len(labels))  # 标签位置
width = 0.1  # 柱状图的宽度


# 设置数据
sigma_1 = [30.51, 35.48, 37.92]
sigma_5 = [30.46, 35.03, 37.93]
sigma_10 = [30.42, 35.02, 38.02]
sigma_20 = [30.42, 35.25, 38.54]
sigma_50 = [31.12, 35.25, 37.81]
# 设置柱状图的位置

ax1.bar(x - 2*width, sigma_1, width, label=r'$\sigma = 1$',color=colors[0], edgecolor='grey')
ax1.bar(x - width, sigma_5, width, label=r'$\sigma = 5$',color=colors[1], edgecolor='grey')
ax1.bar(x, sigma_10, width, label=r'$\sigma = 10$',color=colors[2], edgecolor='grey')
ax1.bar(x + width, sigma_20, width, label=r'$\sigma = 20$',color=colors[3], edgecolor='grey')
ax1.bar(x + 2*width, sigma_50, width, label=r'$\sigma = 50$',color=colors[4], edgecolor='grey')

# 添加一些文本标签
ax1.set_ylabel('Acc(%)')
ax1.set_title('Acc on ENZYMES')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
# ax1.legend()


# 调用上述函数
# autolabel(ax1,rects1)
# autolabel(ax1,rects2)
# autolabel(ax1,rects3)
# 设置数据
sigma_1 = [58.28, 58.40, 64.00]
sigma_5 = [63.82, 61.47, 64.81]
sigma_10 = [57.66, 57.42, 64.94]
sigma_20 = [61.07, 59.94, 64.35]
sigma_50 = [60.28, 58.39, 64.42]
# 设置柱状图的位置

ax2.bar(x - 2*width, sigma_1, width, label=r'$\sigma = 1$',color=colors[0], edgecolor='grey')
ax2.bar(x - width, sigma_5, width, label=r'$\sigma = 5$',color=colors[1], edgecolor='grey')
ax2.bar(x, sigma_10, width, label=r'$\sigma = 10$',color=colors[2], edgecolor='grey')
ax2.bar(x + width, sigma_20, width, label=r'$\sigma = 20$',color=colors[3], edgecolor='grey')
ax2.bar(x + 2*width, sigma_50, width, label=r'$\sigma = 50$',color=colors[4], edgecolor='grey')

# 添加一些文本标签
ax2.set_ylabel('Acc(%)')
ax2.set_title('Acc on PROTEINS')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
#
# plt.subplot(2,2,2)
# x = np.array([1,5,10,20,50])  # X轴数据点
# accuracies = {
#     '5-shot': np.array([58.28, 63.82, 57.66, 61.07, 60.28]),
#     '10-shot': np.array([58.40, 61.47, 57.42, 59.94, 58.39]),
#     '20-shot': np.array([64.00, 64.81, 64.94, 64.35, 64.42])
# }
#
# x_indices = np.arange(len(x))
# # 绘制每个算法的数据点和线
# for label, values in accuracies.items():
#     plt.plot(x_indices, values, marker='o', label=label)
#
# # 添加图例
# plt.legend()
#
# plt.xticks(x_indices, x)  # 将实际的x值作为刻度标签
# # 添加标题和坐标轴标签
# plt.title('Acc on PROTEINS')
# plt.xlabel('Shot k')
# plt.ylabel('Accuracy (%)')

# plt.tight_layout()  # 自动调整子图参数，以给定填充
plt.savefig('sigma.png')
plt.show()