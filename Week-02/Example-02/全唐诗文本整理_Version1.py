"""
全唐诗文本整理

Version: 0.1
Author: 长行

每首诗可以提取的内容：卷编号、诗编号、标题、作者、内容

"""

import json
import re
import time

if __name__ == "__main__":
    with open("全唐诗.txt", encoding="UTF-8") as file:
        lines = file.readlines()
    print("总行数:", len(lines))

    start_time = time.time()

    book_num = 0  # 卷编号
    poem_num = 0  # 诗编号
    title = None  # 标题
    author = "未知"  # 作者
    content = ""  # 诗文内容

    poem_list = list()  # 诗文列表

    for line in lines:

        # 文本清洗
        line = line.replace(" ", "").replace("　", "").replace("\r", "").replace("\n", "").replace("\t", "")

        # 无效行过滤
        if len(line) == 0:
            continue
        if "知古斋主精校" in line or "版权所有" in line or "web@guoxue.com" in line or "返回《全唐诗》" in line:
            continue
        if re.search("第[一二三四五六七八九十百]+卷|卷[一二三四五六七八九十百]+", line):
            continue

        # 处理标题行的情况
        if re.search("卷[0-9]+[_-][0-9]+", line):

            # 将上一首诗存入临时变量
            if title is not None:
                content = re.sub("[\n]+$", "", content)
                poem_list.append({
                    "book_num": book_num,
                    "poem_num": poem_num,
                    "title": title,
                    "author": author,
                    "content": content
                })

            # 解析卷编号
            if book_num_regex := re.search("(?<=卷)[0-9]+(?=[_-])", line):
                book_num = int(book_num_regex.group())
            else:
                book_num = 0
                print("卷编号解析失败!")

            # 解析诗编号
            if poem_num_regex := re.search("(?<=[_-])[0-9]+", line):
                poem_num = int(poem_num_regex.group())
            else:
                poem_num = 0
                print("诗编号解析失败!")

            # 解析标题
            if title_regex := re.search("(?<=【)[^】]+(?=】)", line):
                title = title_regex.group()
            else:
                title = None
                print("标题解析失败!")

            # 解析作者(存在于标题行中的情况)
            line = re.sub("卷[0-9]+[_-][0-9]+【[^】]+】", "", line)  # 清除作者之外的其他内容
            if author_regex := re.search("[\u4e00-\u9fa5]+", line):
                author = author_regex.group()
            else:
                author = "未知"

            content = ""  # 诗文内容

        # 处理非标题行的情况
        else:
            # 清洗非标题行的数据
            line = line.replace("¤", "。")
            line = re.sub("(?<=[，。？！）】])[知古斋主]+$", "", line)
            line = re.sub("(?<=[，。？！）】])[知古斋主]+(?=-)", "", line)

            # 处理作者行的情况
            if author == "未知" and content == "" and not re.search("[，。？！]", line):
                author = line

            # 处理诗文行的情况
            else:
                content += line + "\n"

    # 将最后一首诗写入到临时变量
    content = re.sub("[\n]+$", "", content)
    poem_list.append({
        "book_num": book_num,
        "poem_num": poem_num,
        "title": title,
        "author": author,
        "content": content
    })

    print("共解析诗文数:", len(poem_list))

    print(time.time() - start_time)

    # 遍历检查
    # for poem_item in poem_list:
    #     print(poem_item["卷编号"], poem_item["诗编号"], poem_item["标题"], poem_item["作者"], poem_item["内容"])

    # 将临时变量中的数据存储到Json文件
    with open("全唐诗.json", "w+", encoding="UTF-8") as file:
        file.write(json.dumps({"data": poem_list}, ensure_ascii=False))
