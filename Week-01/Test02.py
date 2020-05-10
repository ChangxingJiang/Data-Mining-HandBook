"""
牛吃草问题(池塘管理员之禅)

Version: 0.1
Author: 长行
"""

import math

if __name__ == "__main__":
    cattle_num_1 = int(input("请输入第1种情况牛的数量: "))
    day_num_1 = int(input("请输入第1种情况吃的天数: "))
    cattle_num_2 = int(input("请输入第2种情况牛的数量: "))
    day_num_2 = int(input("请输入第2种情况吃的天数: "))
    cattle_num_3 = int(input("请输入您想计算的情况的牛的数量: "))

    grass_grow_speed = (cattle_num_1 * day_num_1 - cattle_num_2 * day_num_2) / (day_num_1 - day_num_2)  # 计算草的生长速度
    total_grass = (cattle_num_1 - grass_grow_speed) * day_num_1  # 计算原有草的总量
    day_num_3 = total_grass / (cattle_num_3 - grass_grow_speed)  # 计算当前牛的数量可以吃的天数

    if day_num_3 < 1:
        print("你的牛太多了，请为生态可持续发展尽一份绵薄之力")
    else:
        print("可以吃的天数:", math.floor(day_num_3))
