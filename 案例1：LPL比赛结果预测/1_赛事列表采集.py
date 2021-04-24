"""
采集WanPlus中的赛事列表，并将结果存入到数据库中；
需要设置采集的年份及每年包含的赛事页面数。
"""

# 目标Url : http://www.wanplus.com/lol/event?t=3&year=2014&page=1

import re
import time

import crawlertool as tool
from bs4 import BeautifulSoup

# 数据库信息
MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE = tool.io.load_json("E:\\同步工作区\\mysql_info.json")["lol_game"]

# 赛事列表页面Url
EVENT_LIST_PAGE_URL = "http://www.wanplus.com/lol/event"

# 年份列表及每年包含赛事页数列表
YEAR_PAGE_NUM_LIST = [(2014, 3), (2015, 3), (2016, 2), (2017, 4), (2018, 3), (2019, 3), (2020, 3), (2021, 1)]


def spider():
    # 连接到MySQL数据库
    mysql = tool.db.MySQL(host=MYSQL_HOST, database=MYSQL_DATABASE, user=MYSQL_USER, password=MYSQL_PASSWORD)

    # 遍历所有年份
    for year, page_num in YEAR_PAGE_NUM_LIST:

        # 遍历年份的所有页面
        for page in range(1, page_num + 1):

            print("当前采集:", year, "-", page)

            data_list = []

            # 定义请求的Url参数
            url_params = {
                "t": 3,  # t=3 : 已经结束的赛事
                "year": year,
                "page": page
            }

            # 执行请求
            response = tool.do_request(EVENT_LIST_PAGE_URL, params=url_params)

            # 解析返回结果
            lxml = BeautifulSoup(response.text, "lxml")

            # 遍历所有赛事的外层标签
            for label in lxml.select("#info > div.left-slide > div.left-box > div.event-list > ul > li"):
                wanplus_event_id = re.search("[0-9]+", label.select_one("li > a")["href"]).group()  # 玩加电竞赛事ID
                event_name = label.select_one("li > a > span").text  # 赛事名称

                time_frame = label.select_one("li > a > p:nth-child(3)").text  # 时间范围
                start_date, end_date = time_frame.split(" — ")  # 赛事开始时间、赛事结束时间

                data_list.append({
                    "event_name": event_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "wanplus_event_id": wanplus_event_id
                })

            # 将数据写入到MySQL数据库
            mysql.insert("event", data_list)

            time.sleep(3)


if __name__ == "__main__":
    spider()
