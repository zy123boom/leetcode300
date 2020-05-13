package leetcode300_hard;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * LeetCode前300道，困难题
 * 题目来源：力扣（LeetCode）
 * 链接：https://leetcode-cn.com/problemset/algorithms/?difficulty=%E5%9B%B0%E9%9A%BE
 *
 * @author boomzy
 * @date 2020/2/2 20:30
 */
public class Main {


    /**
     * LeetCode.10 正则表达式匹配
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        /*
            dp[i][j]:代表s中从第一个字符取到第i个字符和从p中的第一个字符取到第j个字符
            是不是相匹配的。
            例如：s="abcdc" p="abc"，则dp[3][3] = true, dp[3][2] = false
         */
        if (s == null || p == null) {
            return false;
        }
        boolean[][] match = new boolean[s.length() + 1][p.length() + 1];
        match[0][0] = true;

        // 看p字符串，i是数组下标，如果当前字符是*，取上一个字符
        for (int i = 1; i <= p.length(); i++) {
            if (p.charAt(i - 1) == '*') {
                match[0][i] = match[0][i - 2];
            }
        }

        for (int si = 1; si <= s.length(); si++) {
            for (int pi = 1; pi <= p.length(); pi++) {
                if (p.charAt(pi - 1) == '.' || p.charAt(pi - 1) == s.charAt(si - 1)) {
                    match[si][pi] = match[si - 1][pi - 1];
                } else if (p.charAt(pi - 1) == '*') {
                    if (p.charAt(pi - 2) == s.charAt(si - 1) || p.charAt(pi - 2) == '.') {
                        match[si][pi] = match[si][pi - 2] || match[si - 1][pi];
                    } else {
                        match[si][pi] = match[si][pi - 2];
                    }
                }
            }
        }
        return match[s.length()][p.length()];
    }

    /**
     * LeetCode.42 接雨水
     *
     * @param height
     * @return
     */
    public int trap(int[] height) {
        /*
            方法一：找出左右两边最高的值，相当于墙。要得到雨水的量，取两堵墙中较矮的然后减去当前位置的高度
            就是当前位置可以存下的水，全部的加起来就是结果。空间复杂度O(N)

            代码如下：
            if (height == null || height.length == 0) {
                return 0;
            }
            int n = height.length;
            int res = 0;
            int[] left = new int[n];
            int[] right = new int[n];
            left[0] = height[0];
            for (int i = 1; i < n; i++) {
                left[i] = Math.max(left[i - 1], height[i]);
            }
            right[n - 1] = height[n - 1];
            for (int i = n - 2; i >= 0; i--) {
                right[i] = Math.max(right[i + 1], height[i]);
            }
            for (int i = 0; i < n; i++) {
                res += Math.min(left[i], right[i]) - height[i];
            }
            return res;

            方法二：双指针，此处是方法二的代码
         */
        if (height == null || height.length == 0) {
            return 0;
        }
        int res = 0;
        int leftMax = 0;
        int rightMax = 0;
        int i = 0, j = height.length - 1;
        while (i < j) {
            leftMax = Math.max(leftMax, height[i]);
            rightMax = Math.max(rightMax, height[i]);
            if (leftMax < rightMax) {
                res += leftMax - height[i];
                i++;
            } else {
                res += rightMax - height[j];
                j--;
            }
        }
        return res;
    }

    /**
     * LeetCode.51 N皇后
     * n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
     * 给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
     * <p>
     * 每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
     * <p>
     * 示例:
     * 输入: 4
     * 输出: [
     * [".Q..",  // 解法 1
     * "...Q",
     * "Q...",
     * "..Q."],
     * <p>
     * ["..Q.",  // 解法 2
     * "Q...",
     * "...Q",
     * ".Q.."]
     * ]
     * 解释: 4 皇后问题存在两个不同的解法。
     *
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        char[][] board = new char[n][n];
        init(board);
        helper(res, board, 0);
        return res;
    }

    /**
     * 51题帮助函数
     * 初始化数组，变为.
     *
     * @param board
     */
    private void init(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            Arrays.fill(board[i], '.');
        }
    }

    /**
     * 51题帮助函数
     * 核心函数
     *
     * @param res
     * @param board
     * @param rowIndex
     */
    private void helper(List<List<String>> res, char[][] board, int rowIndex) {
        if (rowIndex == board.length) {
            res.add(generate(board));
        }

        // 列从0开始，看能不能摆放皇后
        for (int colIndex = 0; colIndex < board.length; colIndex++) {
            if (isValid(board, rowIndex, colIndex)) {
                board[rowIndex][colIndex] = 'Q';
                helper(res, board, rowIndex + 1);
                board[rowIndex][colIndex] = '.';
            }
        }
    }

    /**
     * 51题帮助函数
     * 判断当前坐标能否摆放皇后
     *
     * @param board
     * @param rowIndex
     * @param colIndex
     */
    private boolean isValid(char[][] board, int rowIndex, int colIndex) {
        // 1.判断之前的列有没有其他皇后
        for (int i = 0; i < rowIndex; i++) {
            if (board[i][colIndex] == 'Q') {
                return false;
            }
        }

        // 2.判断左上到右下的对角线，当前行列的左上角坐标是(row - 1, col - 1)
        for (int i = rowIndex - 1,  j = colIndex - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        // 3.判断右上到左下的对角线，当前行列的右上角的坐标是(row - 1, col + 1)
        for (int i = rowIndex - 1, j = colIndex + 1; i >= 0 && j < board.length; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }

        return true;
    }

    /**
     * 51帮助函数
     * 将字符二维数组转化为嵌套List
     *
     * @param board
     * @return
     */
    private List<String> generate(char[][] board) {
        List<String> res = new ArrayList<>();
        for (char[] row : board) {
            StringBuilder sb = new StringBuilder();
            for (char c : row) {
                sb.append(c);
            }
            res.add(sb.toString());
        }
        return res;
    }
}
