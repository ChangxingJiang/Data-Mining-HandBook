# LeetCode 0326
# https://leetcode-cn.com/problems/power-of-three/


class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        a = 1
        while a <= n:
            if a == n:
                return True
            a *= 3
        return False


if __name__ == "__main__":
    print(Solution().isPowerOfThree(27))  # True
    print(Solution().isPowerOfThree(0))  # False
    print(Solution().isPowerOfThree(9))  # True
    print(Solution().isPowerOfThree(45))  # False
