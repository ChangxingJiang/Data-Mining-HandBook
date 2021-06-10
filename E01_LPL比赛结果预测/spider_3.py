import json
import os
import time

import crawlertool as tool
from mysql import connector


class SpiderWanplusLolMatchInfo(tool.abc.SingleSpider):
    """WanPlus英雄联盟场次详细信息爬虫"""

    # 比赛页面的Url
    _SCHEDULE_URL = "https://www.wanplus.com/schedule/%s.html"

    # 场次Ajax的Url
    _MATCH_URL = "https://www.wanplus.com/ajax/matchdetail/%s?_gtk=345357323"

    # 场次Ajax的请求headers
    _MATCH_HEADERS = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-TW;q=0.6",
        "cache-control": "no-cache",
        "cookie": "wanplus_token=40854d66239062302b8443d29ba4ee29; "
                  "wanplus_storage=lf4m67eka3o; "
                  "wanplus_sid=435664e12e07bf6e8d6511f604789cb0; "
                  "isShown=1; "
                  "wanplus_csrf=_csrf_tk_446622052; "
                  "gameType=2",
        "pragma": "no-cache",
        "referer": "",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36",
        "x-csrf-token": "446622052",
        "x-requested-with": "XMLHttpRequest"
    }

    def running(self, schedule_id, match_id):
        # ----- 执行请求 -----
        response = tool.do_request(self._MATCH_URL % str(match_id), headers=self._MATCH_HEADERS)
        response_text = response.content.decode()

        # 如果请求结果为空，或请求结果不是Json格式数据，则直接返回
        if not response_text or response_text[0] == "<":
            return

        response_json = json.loads(response_text)

        # 如果请求失败（来历不明的请求），则直接返回
        if response_json["ret"] == -400:
            return

        # ----- 解析返回结果 -----
        return response_text


# ------------------- 运行爬虫 -------------------

HOST, USER, PASSWORD = os.getenv("DATA_MINING_MYSQL").split(";")
DATABASE = "lol_game_record"

PATH = r"E:\数据存储区\英雄联盟比赛数据"

if __name__ == "__main__":
    # ----- 加载数据库中未采集的场次 -----
    # 连接到数据库
    connect = connector.connect(host=HOST, user=USER, password=PASSWORD, database=DATABASE)
    cursor = connect.cursor()

    # 查询需要采集的场次列表
    aim_list = []
    try:
        cursor.execute("SELECT `schedule_id`, `match_id` "
                       "FROM match_info "
                       "WHERE `event_id` = 907 "
                       "  AND `is_fetch_info` = FALSE")
        aim_list = list(cursor.fetchall())
    except connector.errors.ProgrammingError:
        pass

    if aim_list:
        # ----- 执行采集 -----
        size = len(aim_list)
        print("共计需要采集的场次数:", size)

        for i in range(len(aim_list)):
            print("当前采集:", i + 1, "/", size)
            schedule_id, match_id = aim_list[i]

            # 运行爬虫
            match_info_text = SpiderWanplusLolMatchInfo().running(schedule_id=schedule_id, match_id=match_id)

            # 将结果存储到文件(Json格式)
            save_path = os.path.join(PATH, str(schedule_id) + "_" + str(match_id) + ".json")
            tool.io.write_json(save_path, match_info_text)

            # 将数据库中未采集的标记改为已采集的标记
            sql = ("UPDATE match_info "
                   "SET `is_fetch_info` = TRUE "
                   "WHERE `schedule_id` = " + str(schedule_id) +
                   "  AND `match_id` = " + str(match_id))
            try:
                cursor.execute(sql)
                connect.commit()
            except connector.errors.ProgrammingError:
                print("更新数据库中采集状态失败:", sql)

            time.sleep(5)
