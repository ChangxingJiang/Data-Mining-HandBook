"""
采集WanPlus中的赛事包含的比赛列表，并将结果存储到数据库；同时将各个赛段的信息写入数据库；
在具体实现中，首先采集赛事所包含的赛段列表，然后采集每个赛段所包含的比赛列表。
"""

# 采集赛事赛段列表
# 目标Url : http://www.wanplus.com/event/18.html

# 采集比赛赛段包含的比赛
# 目标Url : http://www.wanplus.com/ajax/event/shedule/detail

import re
import time

import crawlertool as tool
from bs4 import BeautifulSoup

# 数据库信息
MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE = tool.io.load_json("E:\\同步工作区\\mysql_info.json")["lol_game"]

# 赛事信息Url
EVENT_INFO_URL = "http://www.wanplus.com/event/{}.html"

# 赛段信息Url
STAGE_INFO_URL = "http://www.wanplus.com/ajax/event/shedule/detail"


def spider():
    # 连接到MySQL数据库
    mysql = tool.db.MySQL(host=MYSQL_HOST, database=MYSQL_DATABASE, user=MYSQL_USER, password=MYSQL_PASSWORD)

    # 从数据库中读取赛事列表
    event_list = mysql.select("event", columns=["event_id", "event_name", "wanplus_event_id"], where="event_id>=251")

    # 遍历所有赛事
    for event_id, event_name, wanplus_event_id in event_list:
        print("当前采集:", event_id, "-", event_name, "-", wanplus_event_id)

        # ----- 采集赛事的赛段列表 -----
        # 执行请求
        response = tool.do_request(EVENT_INFO_URL.format(wanplus_event_id))
        time.sleep(5)

        # 解析返回结果
        lxml = BeautifulSoup(response.text, "lxml")

        # 提取所有赛段编号
        stage_id_list = []
        stage_name_list = []
        for label in lxml.select("#event_stage > li"):
            stage_id_list.append(int(label["data-stageid"]))
            stage_name_list.append(label.text.replace("\n", "").lstrip())

        # 将赛段信息写入数据库
        data_list_1 = []
        for i in range(len(stage_id_list)):
            data_list_1.append({
                "event_id": event_id,
                "stage_name": stage_name_list[i],
                "wanplus_stage_id": stage_id_list[i]
            })
        mysql.insert("stage", data_list_1)

        # ----- 遍历各个赛段中包含的比赛 -----
        for stage_id in stage_id_list:
            print("当前采集赛段:", stage_id)

            data_list_2 = []

            # 定义请求的Url参数
            url_params = {
                "_gtk": 868258461,  # 可视作常量
                "eId": wanplus_event_id,
                "stageId": stage_id,
                "gameType": 2  # 可视作常量
            }

            # 执行请求
            response = tool.do_request(STAGE_INFO_URL, method="POST", data=url_params)
            time.sleep(3)

            # 解析返回结果
            lxml = BeautifulSoup(response.text, "lxml")

            # 提取赛段中比赛的编号
            for label in lxml.select("a"):
                wanplus_schedule_id = int(re.search("[0-9]+", label["href"]).group())

                data_list_2.append({
                    "event_id": event_id,
                    "stage_id": stage_id,
                    "wanplus_schedule_id": wanplus_schedule_id
                })

            # 将数据写入到MySQL数据库
            mysql.insert("schedule", data_list_2)


if __name__ == "__main__":
    spider()
