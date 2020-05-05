'''
'1234'
'123'
'''

x = int(input())
is_prime = False
for k in range(2, x):
    if x % k == 0:
        break
else:
    is_prime = True
if is_prime:
    print("质数")
else:
    print("不是质数")
