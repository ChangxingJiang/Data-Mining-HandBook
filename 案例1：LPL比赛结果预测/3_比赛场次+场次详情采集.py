"""
采集WanPlus中的比赛包含的场次列表，并将结果存储到数据库；同时将场次的详细信息存入本地文件中。
"""

# 比赛信息页面
# 目标Url : http://www.wanplus.com/schedule/66558.html

# 场次详情页面
# 目标Url : http://www.wanplus.com/ajax/matchdetail/71043?_gtk=868258461

import crawlertool as tool

# 数据库信息
MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE = tool.io.load_json("E:\\同步工作区\\mysql_info.json")["lol_game"]

def spider():
    # 连接到MySQL数据库
    mysql = tool.db.MySQL(host=MYSQL_HOST, database=MYSQL_DATABASE, user=MYSQL_USER, password=MYSQL_PASSWORD)

    # 从数据库中读取赛事列表
    schedule_list = mysql.select("schedule", columns=["event_id", "stage_id", "wanplus_schedule_id"])


if __name__ == "__main__":
    spider()
