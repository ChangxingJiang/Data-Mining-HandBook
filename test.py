def day(a1,b1,a2,b2):
    c=(a1*b1-a2*b2)/(b1-b2)
    M=(a1-c)*b1
    t=M/(a3-c)
    return t

a1 = float(input("请输入第一个牛的头数的条件："))
a2 = float(input("请输入第二个牛的头数的条件："))
b1 = float(input("请输入第一个天数的条件："))
b2 = float(input("请输入第二个天数的条件："))
a3 = float(input("请输入您想求的牛的头数的条件："))
print("需要的天数为：{}".format(day(a1,b1,a2,b2)))