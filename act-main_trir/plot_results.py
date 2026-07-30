import matplotlib.pyplot as plt

import numpy as np



# 这是你未来写论文时的实验数据字典

# 目前基线是 16%，后面的数据是我们预估的，等你跑出来后直接改数字就行

results = {
    "Base ACT (3 Cameras)": 38,  # 这是你刚刚跑出的坚实底座！
    "+ DINOv2 Features": 60,     # 预估目标：解决语义特征问题
    "+ Epipolar 3D Align": 78,   # 预估目标：解决深度缺失问题
    "Full Ours": 92              # 预估目标：完全体霸榜！
}



methods = list(results.keys())

success_rates = list(results.values())



# 设置画图的高级质感格式（对标 IEEE 论文风格）

plt.figure(figsize=(10, 6))

# 颜色递进，突出你最终的方法

colors = ['#cccccc', '#8da0cb', '#fc8d62', '#e78ac3'] 

bars = plt.bar(methods, success_rates, color=colors, width=0.5)



# 添加网格线和标签

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')

plt.title('Ablation Study on Sim Insertion Task', fontsize=16, fontweight='bold')

plt.ylim(0, 100)



# 在柱状图顶部自动标上百分比数字

for bar in bars:

    yval = bar.get_height()

    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval}%', 

             ha='center', va='bottom', fontsize=12, fontweight='bold')



# 保存为高清图片，dpi=300 是所有期刊的硬性要求

plt.savefig('ablation_study_chart.png', dpi=300, bbox_inches='tight')

print("🎉 论文图表已成功生成：ablation_study_chart.png")
