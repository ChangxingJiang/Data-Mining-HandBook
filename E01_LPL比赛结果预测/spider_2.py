import os
import re
import time

import crawlertool as tool
from bs4 import BeautifulSoup
from mysql import connector


class SpiderWanplusLolMatchList(tool.abc.SingleSpider):
    """WanPlus英雄联盟比赛包含场次列表爬虫"""

    _COLUMNS = ["match_id",
                "schedule_id", "schedule_date", "schedule_time", "schedule_bo_num", "event_id", "event_name",
                "team_a_id", "team_a_name", "team_b_id", "team_b_name", "team_a_win", "team_b_win"]

    # 比赛请求的url
    _RACE_LIST_URL = "https://www.wanplus.com/schedule/%s.html"

    # 比赛请求的Headers
    _RACE_LIST_HEADERS = {
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

    def running(self, schedule_id: int):
        result = []  # 初始化返回结果数据

        # ----- 计算请求参数并执行请求 -----
        response = tool.do_request(self._RACE_LIST_URL % str(schedule_id), headers=self._RACE_LIST_HEADERS)

        if response.status_code != 200:
            print("请求失败!")
            return result

        bs = BeautifulSoup(response.content.decode(), "lxml")

        # ----- 判断返回结果是否有效 -----
        label = bs.select_one("body")
        # 判断比赛是否为未进行的状态：如果是未进行的状态则跳出
        if label.has_attr("class") and "matchbf" in label["class"]:
            return
        # 判断是否为页面未找到的状态：如果是未找到的状态则跳出
        if label.has_attr("class") and "mess_html" in label["class"]:
            return

        # ----- 解析请求比赛基本信息 -----
        # 解析赛事ID和赛事名称
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > h1 > a"
        label = bs.select_one(selector)
        event_id = int(re.search("[0-9]+", label["href"]).group())  # 赛事ID
        event_name = label.text  # 赛事名称

        # 解析比赛时间
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li:nth-child(2) > span.time"
        label = bs.select_one(selector)
        if " " in label.text:
            schedule_date, schedule_time = label.text.split(" ")[0:2]  # 比赛日期、比赛时间
        else:
            schedule_date, schedule_time = "", ""

        # 解析比赛双方队伍信息
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li.team-left > a"
        label = bs.select_one(selector)
        team_a_id = int(re.search("[0-9]+", label["href"]).group())  # 队伍A的ID
        team_a_name = label.text.replace("\n", "")  # 队伍A的名称
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li.team-right.tr > a"
        label = bs.select_one(selector)
        team_b_id = int(re.search("[0-9]+", label["href"]).group())  # 队伍B的ID
        team_b_name = label.text.replace("\n", "")  # 队伍B的名称

        # 解析比赛比分、比赛赛制
        selector = "body > div.body-inner > div.content > div.left > div:nth-child(1) > ul > li:nth-child(2) > p"
        marks = bs.select_one(selector).text.split(":")
        team_a_win, team_b_win = int(marks[0]), int(marks[1])  # 队伍A的小分、队伍B的小分
        schedule_bo_num = max(team_a_win, team_b_win)  # 比赛赛制

        # ----- 解析请求比赛场次信息 -----
        game_labels = bs.select("body > div > div.content > div.left > div:nth-child(1) > div > a")
        for game_label in game_labels:
            if game_label.has_attr("data-matchid"):
                result.append({
                    "match_id": game_label["data-matchid"],  # 场次ID
                    "schedule_id": schedule_id,  # 比赛ID
                    "schedule_date": schedule_date,
                    "schedule_time": schedule_time,
                    "schedule_bo_num": schedule_bo_num,
                    "event_id": event_id,
                    "event_name": event_name,
                    "team_a_id": team_a_id,
                    "team_a_name": team_a_name,
                    "team_b_id": team_b_id,
                    "team_b_name": team_b_name,
                    "team_a_win": team_a_win,
                    "team_b_win": team_b_win
                })

        # 输出数据结果
        self.output(result)

    def output(self, data):
        pass


# ------------------- 爬虫运行 -------------------
HOST, USER, PASSWORD = os.getenv("DATA_MINING_MYSQL").split(";")
DATABASE = "lol_game_record"


class MySpider(SpiderWanplusLolMatchList):
    def __init__(self):
        self.connect = connector.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
        self.cursor = self.connect.cursor()

    def output(self, data):
        # 处理没有需要写入的数据的情况
        if len(data) == 0:
            return

        def format_val(column):
            # 整理变量格式：如果是字符串则添加引号，如果不是则转换为字符串
            if isinstance(column, str):
                return "'" + column + "'"
            else:
                return str(column)

        # 生成SQL语句
        sql = ("INSERT INTO match_info (" + ",".join(self._COLUMNS) + ") VALUES " +
               ",".join(["(" + ",".join([format_val(item[cell]) for cell in self._COLUMNS]) + ")" for item in data]) +
               " ON DUPLICATE KEY UPDATE add_time = NOW()")

        # 执行SQL语句
        try:
            self.cursor.execute(sql)
            self.connect.commit()
        except connector.errors.ProgrammingError:
            print("SQL语句执行异常:", sql)


START_NUM = 1447
END_NUM = 1732

if __name__ == "__main__":
    spider = MySpider()
    for sid in range(START_NUM, END_NUM + 1):
        print("当前抓取:", sid, "(", (sid - START_NUM + 1), "/", (END_NUM - START_NUM + 1), ")")
        spider.running(sid)
        time.sleep(5)
