## 1. 采集赛事信息

采集赛事信息数据，并将结果存入数据库。

数据库结构：

```mysql
CREATE TABLE `event` (
  `event_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '赛事ID',
  `event_name` char(40) DEFAULT NULL COMMENT '赛事名称',
  `start_date` date DEFAULT NULL COMMENT '赛事开始日期',
  `end_date` date DEFAULT NULL COMMENT '赛事结束日期',
  `wanplus_event_id` int(11) DEFAULT NULL COMMENT '玩加电竞ID',
  PRIMARY KEY (`event_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='赛事信息';
```

爬虫代码：

```python
# 目标Url : http://www.wanplus.com/lol/event?t=0&year=2014&page=1

import re
import time

import crawlertool as tool
from bs4 import BeautifulSoup

# 数据库信息
MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE = tool.io.load_json("E:\\同步工作区\\mysql_info.json")["lol_game"]

# 赛事列表页面Url
EVENT_LIST_PAGE_URL = "http://www.wanplus.com/lol/event"

# 年份列表及每年包含赛事页数列表
YEAR_PAGE_NUM_LIST = ((2014, 3), (2015, 3), (2016, 2), (2017, 4), (2018, 3), (2019, 3), (2020, 3))


def spider():
    # 遍历所有年份
    for year, page_num in YEAR_PAGE_NUM_LIST:

        # 遍历年份的所有页面
        for page in range(1, page_num + 1):

            print("当前采集:", year, "-", page)

            data_list = []

            # 定义请求的Url参数
            url_params = {
                "t": 0,  # t=0 : 所有赛事
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
            mysql = tool.db.MySQL(host=MYSQL_HOST, database=MYSQL_DATABASE, user=MYSQL_USER, password=MYSQL_PASSWORD)
            mysql.insert("event", data_list)

            time.sleep(3)


if __name__ == "__main__":
    spider()
```

## 2. 采集赛事包含比赛

采集赛事包含的比赛信息，并存储到数据库。

数据库结构：

```mysql
CREATE TABLE `schedule` (
  `schedule_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '比赛ID',
  `event_id` int(11) DEFAULT NULL COMMENT '赛事ID',
  `stage_id` int(11) DEFAULT NULL COMMENT '赛段ID',
  `wanplus_schedule_id` int(11) DEFAULT NULL COMMENT '玩加电竞比赛ID',
  PRIMARY KEY (`schedule_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='比赛信息';

CREATE TABLE `stage` (
  `stage_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '赛程ID',
  `event_id` int(11) DEFAULT NULL COMMENT '赛事ID',
  `stage_name` varchar(40) DEFAULT NULL COMMENT '赛程名称',
  `wanplus_stage_id` int(11) DEFAULT NULL COMMENT 'WanPlus的赛程ID',
  PRIMARY KEY (`stage_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='赛程信息';
```

爬虫代码：

```python
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
    event_list = mysql.select("event", columns=["event_id", "event_name", "wanplus_event_id"])

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
```

## 3. 采集比赛详细信息及比赛包含场次

数据库结构：

```mysql
CREATE TABLE `schedule_detail`
(
    `schedule_id` int(11) NOT NULL,
    `schedule_date` char(30) DEFAULT NULL COMMENT '比赛日期',
    `schedule_time` char(30) DEFAULT NULL COMMENT '比赛时间',
    `schedule_bo_num` int(11) DEFAULT NULL COMMENT '比赛赛事',
    `team_a_wanplus_id` int(11) DEFAULT NULL COMMENT '队伍A的Wanplus的ID',
    `team_a_name` char(10) DEFAULT NULL COMMENT '队伍A的名称',
    `team_b_wanplus_id` int(11) DEFAULT NULL COMMENT '队伍B的Wanplus的ID',
    `team_b_name` char(10) DEFAULT NULL COMMENT '队伍B的名称',
    `team_a_win` int(11) DEFAULT NULL COMMENT '队伍A的小分',
    `team_b_win` int(11) DEFAULT NULL COMMENT '队伍B的小分',
    PRIMARY KEY (`schedule_id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8 COMMENT ='比赛详细信息';
  
CREATE TABLE `match`
(
    `match_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '场次ID',
    `schedule_id` int(11) DEFAULT NULL COMMENT '比赛ID',
    `wanplus_match_id` int(11) DEFAULT NULL COMMENT '场次的Wanplus的ID',
    PRIMARY KEY (`match_id`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8 COMMENT ='场次信息';
```

爬虫代码：

```python
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
        if schedule_id <= 2:
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
```













