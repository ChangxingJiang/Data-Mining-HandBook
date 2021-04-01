# LeetCode 0896
# https://leetcode-cn.com/problems/monotonic-array/


from typing import List


class Solution:
    def isMonotonic(self, A: List[int]) -> bool:
        pass


if __name__ == "__main__":
    print(Solution().isMonotonic([1, 2, 2, 3]))  # True
    print(Solution().isMonotonic([6, 5, 4, 4]))  # True
    print(Solution().isMonotonic([1, 3, 2]))  # False
    print(Solution().isMonotonic([1, 2, 4, 5]))  # True
    print(Solution().isMonotonic([1, 1, 1]))  # True
