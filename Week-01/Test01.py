"""
解决登录问题

Version: 0.1
Author: 长行
"""

USER_LIST = {"AAA": "123456", "BBB": "123456", "CCC": "123456"}

if __name__ == "__main__":
    for i in range(3):
        username = input("请输入用户名:")
        password = input("请输入密码:")
        if username in USER_LIST and USER_LIST.get(username) == password:
            print("登录成功")
            break
        else:
            print("用户名或密码错误，若连续3次错误则自动锁死账号")
    else:
        print("因连续3次输入用户名或密码错误，账号已锁死")
