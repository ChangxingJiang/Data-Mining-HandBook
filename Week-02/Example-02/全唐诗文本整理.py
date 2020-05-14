"""
全唐诗文本整理

Version: 0.1
Author: 长行
"""
import re

if __name__ == "__main__":
    with open('全唐诗.txt', encoding="UTF-8") as file:
        lines = file.readlines()
    print("总行数:", len(lines))

    poem_list = list()

    book_num = 0  # 卷编号
    poem_num = 0  # 诗编号

    for line in lines:
        if re.search("卷[0-9]+_[0-9]+", line):  # 判断是否开始新的一首诗
            # 将上一首诗添加到列表
            poem_list.append({
                "卷编号": book_num,
                "诗编号": poem_num
            })

            # 数据清洗
            line = line.replace("\n", "")

            # 读取标题行包含的卷编号和诗编号
            if book_regex := re.search("(?<=卷)[0-9]+(?=_)", line):
                book_num = int(book_regex.group())  # 读取卷编号
            else:
                book_num = 0
            if poem_regex := re.search("(?<=_)[0-9]+", line):
                poem_num = int(poem_regex.group())  # 读取诗编号\
            else:
                poem_num = 0
            print(book_num, poem_num)

            # 判断标题括号是否匹配
            if "【" in line and "】" in line:
                pass
            else:
                print("括号不匹配:", line)
