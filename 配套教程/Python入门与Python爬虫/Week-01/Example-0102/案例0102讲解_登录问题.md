# 登录问题
> **作者**：长行

## 实现要求
* 让用户输入用户名和密码，如果输入正确，则显示“登陆成功”；
若用户名或密码输入错误，则显示“用户名或密码错误，若连续3次错误则自动锁死账号”；
若用户名或密码输入错误次数达到3次，则显示“因连续3次输入用户名或密码错误，账号已锁死”。

## 实现代码
```python
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
```