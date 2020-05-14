"""
全唐诗文本整理

Version: 0.1
Author: 长行
"""

import re

if __name__ == "__main__":
    with open("全唐诗.txt", encoding="UTF-8") as file:
        lines = file.readlines()
    print("总行数:", len(lines))

    poem_list = list()

    reading = False  # 启动标志(解决第1次识别到标题行写入的问题)
    book_num = 0  # 卷编号
    poem_num = 0  # 诗编号
    title = None  # 标题
    author = "未知"  # 作者
    content = ""  # 诗文内容

    for line in lines:
        # 数据清洗
        line = line.replace("\n", "").replace("　", "").replace(" ", "")
        line = re.sub("卷[一二三四五六七八九十百]+", "", line)
        line = re.sub("第[一二三四五六七八九十百]+卷", "", line)

        # 跳过空行(包括数据清洗后为空行的行)
        if len(line) == 0:
            continue

        # 跳过无实际内容的行:
        if "知古斋主精校" in line or "版权所有" in line or "web@guoxue.com" in line:
            continue

        # 处理标题行的情况
        if re.search("卷[0-9]+_[0-9]+", line):
            # 将上一首诗添加到列表
            if reading:
                poem_list.append({
                    "卷编号": book_num,
                    "诗编号": poem_num,
                    "标题": title,
                    "作者": author,
                    "内容": content
                })

                print(book_num, poem_num, title, author, content)
            else:
                reading = True

            # 标题行读取:卷编号和诗编号
            if book_regex := re.search("(?<=卷)[0-9]+(?=_)", line):
                book_num = int(book_regex.group())  # 读取卷编号
            else:
                book_num = 0
                print("未识别卷编号")
            if poem_regex := re.search("(?<=_)[0-9]+", line):
                poem_num = int(poem_regex.group())  # 读取诗编号
            else:
                poem_num = 0
                print("未识别诗编号")

            # 标题行读取:标题
            if title_regex := re.search("(?<=【)[^】]+(?=】)", line):
                title = title_regex.group()  # 读取标题
            else:
                title = None
                print("未识别标题")

            # 标题行读取:作者
            line = re.sub("卷[0-9]+_[0-9]+", "", line)
            line = re.sub("【[^】]+】", "", line)
            if author_regex := re.search("[\u4e00-\u9fa5]+", line):
                author = author_regex.group()  # 如果作者名位于标题行，则为清除其他所有内容后剩余的中文
            else:
                author = "未知"

            # 初始化诗文内容
            content = ""

        # 处理普通诗文行情况
        else:
            # 普通诗文行数据清洗
            line = line.replace("¤", "。")
            line = re.sub("(?<=[），。])[知古斋主]$", "", line)  # 剔除校注者名称

            if not re.search("[，。？！]", line):
                if author == "未知" and content == "":
                    author_regex = re.search("[\u4e00-\u9fa5]+", line)
                    author = author_regex.group()
            else:
                content += line

    # 将清洗后的全唐诗输出到文件
    with open("全唐诗(清洗后).txt", "w+", encoding="UTF-8") as file:
        for poem_item in poem_list:
            file.write(" ".join([str(poem_item["卷编号"]), str(poem_item["诗编号"]), poem_item["标题"], poem_item["作者"],
                                 poem_item["内容"]]) + "\n")
