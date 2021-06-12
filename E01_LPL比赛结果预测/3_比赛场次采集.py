"""
采集WanPlus中的比赛详细信息及包含的场次列表，并将结果存储到数据库；同时将场次的详细信息存入本地文件中。

目标列表：schedule数据表
结果列表：schedule_detail数据表、match数据表
"""

# 比赛信息页面
# 目标Url : http://www.wanplus.com/schedule/66558.html

# 场次详情页面
# 目标Url : http://www.wanplus.com/ajax/matchdetail/71043?_gtk=868258461

import re
import time

import crawlertool as tool
from bs4 import BeautifulSoup

# 数据库信息
MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE = tool.io.load_json("E:\\同步工作区\\mysql_info.json")["lol_game"]

# 比赛信息Url及请求的headers
RACE_LIST_URL = "https://www.wanplus.com/schedule/{}.html"
RACE_LIST_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "zh-CN,zh;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "referer": "https://www.wanplus.com/lol/schedule",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
}


def spider():
    # 连接到MySQL数据库
    mysql = tool.db.MySQL(host=MYSQL_HOST, database=MYSQL_DATABASE, user=MYSQL_USER, password=MYSQL_PASSWORD)

    # 从数据库中读取赛事列表
    schedule_list = mysql.select("schedule", columns=["schedule_id", "event_id", "stage_id", "wanplus_schedule_id"])

    # 遍历所有比赛
    for schedule_id, event_id, stage_id, wanplus_schedule_id in schedule_list:
        if schedule_id <= 1933:
            continue

        print("当前采集:", schedule_id, "-", event_id, ",", stage_id, ",", wanplus_schedule_id)

        # ----- 请求比赛页面并检测返回结果是否有效 -----
        # 执行请求
        response = tool.do_request(RACE_LIST_URL.format(schedule_id), headers=RACE_LIST_HEADERS)
        time.sleep(5)

        # 判断比赛ID是否已经失效
        if response.status_code != 200:
            continue

        # 解析返回结果
        lxml = BeautifulSoup(response.content.decode(), "lxml")

        label = lxml.select_one("body")

        # 判断比赛是否为未进行的状态
        if label.has_attr("class") and "matchbf" in label["class"]:
            continue

        # 判断是否为页面未找到的状态
        if label.has_attr("class") and "mess_html" in label["class"]:
            continue

        # ----- 解析比赛详细信息 -----
        # 解析比赛时间：比赛日期+比赛时间
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li:nth-child(2) > span.time"
        label = lxml.select_one(selector)
        if " " in label.text:
            schedule_date, schedule_time = label.text.split(" ")[0:2]
        else:
            schedule_date, schedule_time = "", ""

        # 解析比赛双方队伍信息：队伍A的ID+队伍A的名称+队伍B的ID+队伍B的名称
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li.team-left > a"
        label = lxml.select_one(selector)
        team_a_wanplus_id = int(re.search("[0-9]+", label["href"]).group())  # 队伍A的ID
        team_a_name = label.text.replace("\n", "")  # 队伍A的名称

        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li.team-right.tr > a"
        label = lxml.select_one(selector)
        team_b_wanplus_id = int(re.search("[0-9]+", label["href"]).group())  # 队伍B的ID
        team_b_name = label.text.replace("\n", "")  # 队伍B的名称

        # 解析比赛比分和赛制：队伍A的小分+队伍B的小分+比赛赛制
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li:nth-child(2) > p"
        marks = lxml.select_one(selector).text.split(":")
        team_a_win, team_b_win = int(marks[0]), int(marks[1])  # 队伍A的小分、队伍B的小分
        schedule_bo_num = max(team_a_win, team_b_win)  # 比赛赛制

        # 将数据存储到数据库
        data_list_1 = [{
            "schedule_id": schedule_id,
            "schedule_date": schedule_date,
            "schedule_time": schedule_time,
            "schedule_bo_num": schedule_bo_num,
            "team_a_wanplus_id": team_a_wanplus_id,
            "team_a_name": team_a_name,
            "team_b_wanplus_id": team_b_wanplus_id,
            "team_b_name": team_b_name,
            "team_a_win": team_a_win,
            "team_b_win": team_b_win,
        }]
        mysql.insert("schedule_detail", data_list_1)

        # ----- 解析比赛包含场次 -----
        data_list_2 = []

        for label in lxml.select("body > div > div.content > div.left > div:nth-child(1) > div > a"):
            if label.has_attr("data-matchid"):
                # 解析场次ID
                match_id = int(label["data-matchid"])

                data_list_2.append({
                    "schedule_id": schedule_id,
                    "wanplus_match_id": match_id
                })

        # 将数据写入到MySQL数据库
        mysql.insert("match", data_list_2)


if __name__ == "__main__":
    spider()
