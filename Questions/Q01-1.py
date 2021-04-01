# LeetCode 0836
# https://leetcode-cn.com/problems/rectangle-overlap/

from typing import List


class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        pass


if __name__ == "__main__":
    print(Solution().isRectangleOverlap(rec1=[0, 0, 2, 2], rec2=[1, 1, 3, 3]))  # True
    print(Solution().isRectangleOverlap(rec1=[0, 0, 1, 1], rec2=[1, 0, 2, 1]))  # False
