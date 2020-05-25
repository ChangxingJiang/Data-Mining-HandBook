# 登录问题
> **作者**：长行

## 实现要求
通过编写代码，回答如下问题：这片青草供给10头牛吃，可以吃22天，或者供给16头牛吃，可以吃10天，期间一直有草生长。如果供给25头牛吃，可以吃多少天？

## 参考资料
草的生长速度 = (对应的牛头数×吃的较多天数-相应的牛头数×吃的较少天数)÷(吃的较多天数－吃的较少天数)

原有草量 = 牛头数×吃的天数-草的生长速度×吃的天数

吃的天数 = 原有草量÷（牛头数-草的生长速度）

牛头数 = 原有草量÷吃的天数+草的生长速度

## 实现代码
```python
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
```