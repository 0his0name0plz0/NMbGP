import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 设置数据
labels = ['Group 1', 'Group 2', 'Group 3']
values_a = [20, 35, 30]
values_b = [25, 32, 34]
values_c = [22, 27, 29]

# 设置柱状图的位置
x = np.arange(len(labels))  # 标签位置
width = 0.2  # 柱状图的宽度

# 创建柱状图
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, values_a, width, label='A')
rects2 = ax.bar(x, values_b, width, label='B')
rects3 = ax.bar(x + width, values_c, width, label='C')

# 添加一些文本标签
ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Values by group and category')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 用来优化每个柱状图的文本标签位置的函数
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# 调用上述函数
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 显示图形
plt.show()
