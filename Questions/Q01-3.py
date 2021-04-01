# LeetCode 0412
# https://leetcode-cn.com/problems/fizz-buzz/


class Solution:
    def fizzBuzz(self, n):
        res = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                res.append('FizzBuzz')
            elif i % 5 == 0:
                res.append('Buzz')
            elif i % 3 == 0:
                res.append('Fizz')
            else:
                res.append(str(i))
            print(i, ":", res)
        return res


if __name__ == "__main__":
    print(Solution().fizzBuzz(15))
    # [
    #     "1",
    #     "2",
    #     "Fizz",
    #     "4",
    #     "Buzz",
    #     "Fizz",
    #     "7",
    #     "8",
    #     "Fizz",
    #     "Buzz",
    #     "11",
    #     "Fizz",
    #     "13",
    #     "14",
    #     "FizzBuzz"
    # ]
