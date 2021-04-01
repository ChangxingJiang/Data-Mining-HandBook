# LeetCode LCP 01
# https://leetcode-cn.com/problems/guess-numbers/

from typing import List


class Solution:
    def game(self, guess: List[int], answer: List[int]) -> int:
        ans = 0
        for i in range(len(guess)):
            ans += 1 if guess[i] == answer[i] else 0
        return ans


if __name__ == "__main__":
    print(Solution().game(guess=[1, 2, 3], answer=[1, 2, 3]))  # 3
    print(Solution().game(guess=[2, 2, 3], answer=[3, 2, 1]))  # 1
