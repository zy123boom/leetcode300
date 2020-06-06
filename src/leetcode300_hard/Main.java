package leetcode300_hard;

import java.util.*;

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
     * LeetCode.25 K个一组翻转链表
     * <p>
     * 给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。
     * k 是一个正整数，它的值小于或等于链表的长度。
     * 如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
     * <p>
     * 示例：
     * 给你这个链表：1->2->3->4->5
     * 当 k = 2 时，应当返回: 2->1->4->3->5
     * 当 k = 3 时，应当返回: 3->2->1->4->5
     * <p>
     * 说明：
     * 你的算法只能使用常数的额外空间。
     * 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        /*
            方法一：使用栈，将每k个元素放入栈，然后弹出就是反向的。当走到最后元素为null时栈的长度不足k，
            说明不需要后面的入栈k个操作。该方法不满足的要求，时间复杂度O(N)。

            首先设置一个dummy.next = head, curr指针指向dummy，next指针指向dummy.next即head。
            然后入栈。然后判断是否需要后面入栈的要求，看是不是入了k个元素。然后不断的出栈，最后让curr.next=next
            返回dummy.next.代码如下。

            if (head == null) {
                return null;
            }
            Stack<ListNode> stack = new Stack<>();
            ListNode dummy = new ListNode(0);
            dummy.next = head;
            ListNode curr = dummy;
            ListNode next = dummy.next;
            while (next != null) {
                for (int i = 0; i < k && next != null; i++) {
                    stack.push(next);
                    next = next.next;
                }
                if (stack.size() != k) {
                    return dummy.next;
                }
                while (!stack.isEmpty()) {
                    curr.next = stack.pop();
                    curr = curr.next;
                }
                curr.next = next;
            }
            return dummy.next;

            方法二：两个指针，一个prev指向dummy，一个last在后面，要翻转的是prev和last之间的链表
         */
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        while (prev != null) {
            prev = reverse(prev, k);
        }
        return dummy.next;
    }

    /**
     * 25题帮助函数
     *
     * @param prev
     * @param k
     * @return
     */
    private ListNode reverse(ListNode prev, int k) {
        ListNode last = prev;
        for (int i = 0; i < k + 1; i++) {
            last = last.next;
            if (i != k && last == null) {
                return null;
            }
        }
        ListNode tail = prev.next;
        ListNode curr = prev.next.next;
        while (curr != last) {
            ListNode next = curr.next;
            curr.next = prev.next;
            prev.next = curr;
            tail.next = next;
            curr = next;
        }
        return tail;
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
     * LeetCode.45 跳跃游戏II
     * <p>
     * 给定一个非负整数数组，你最初位于数组的第一个位置。
     * 数组中的每个元素代表你在该位置可以跳跃的最大长度。
     * 你的目标是使用最少的跳跃次数到达数组的最后一个位置。
     * <p>
     * 示例:
     * 输入: [2,3,1,1,4]
     * 输出: 2
     * 解释: 跳到最后一个位置的最小跳跃数是 2。
     * 从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
     * <p>
     * 说明:
     * 假设你总是可以到达数组的最后一个位置。
     *
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        /*
            记录三个变量。
            step为步数，最终的结果。
            currMax为当前情况下最大能走到的值，是新的可以走到的最大值
            nextMax为当前情况下能走到的最大的地方。每一轮结束该值赋给currMax.
         */
        if (nums == null || nums.length <= 1) {
            return 0;
        }
        int step = 0, currMax = 0, nextMax = 0;
        int index = 0;
        while (index <= currMax) {
            while (index <= currMax) {
                nextMax = Math.max(nextMax, nums[index] + index);
                index++;
            }
            currMax = nextMax;
            step++;
            if (currMax >= nums.length - 1) {
                return step;
            }
        }
        return 0;
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
        for (int i = rowIndex - 1, j = colIndex - 1; i >= 0 && j >= 0; i--, j--) {
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

    /**
     * LeetCode.76 最小覆盖子串
     * <p>
     * 给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字符的最小子串。
     * <p>
     * 示例：
     * 输入: S = "ADOBECODEBANC", T = "ABC"
     * 输出: "BANC"
     * <p>
     * 说明：
     * 如果 S 中不存这样的子串，则返回空字符串 ""。
     * 如果 S 中存在这样的子串，我们保证它是唯一的答案。
     *
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        /*
            由题意可知，最终的结果的起始字符肯定是t中的某个字符，结束字符肯定
            也是t中的某个字符。
            1.先统计t中的每个字符出现的次数，采用HashMap或256长度的数组。
            同时生成一个对应的数组用来与数组对应，每个值初始都是0。
            2.从左向右遍历s，找到第一个t中含有的字符，然后从此处开始搜索，记录指针为left
            表示字串最左边的边界。再记录指针right。
            3.当找到一个字符时，给对应的频率数组下标值加1，然后向右移动right指针，移动到
            下一个包含t中字符的位置。然后此时再记录上一个找到的下标的值count，例如找到第一个
            字符，count就是1，又找到了count为2，以此类推，当count == t.length()时，表示
            所有的字符都找到了，则表示当前的子串是一个可能的解。
            注意！当某个字符出现的次数对应的数组下标值跟t中的数组一样时，表示够了，如果还有该
            字符则count不再增加。
            4.当找到一个可能的解时，移动左指针，移动到下一个t中包含的字符的位置。然后要删掉left之前
            指向的字符，然后减去s对应数组的下标值。此时注意，如果减去之前s对应数组的下标值大于t对应
            数组的下标值，则此时新的left到right仍然是一个有效解。count不变。否则count减少。如果
            此时count < t.length()则不是一个有效的值，则移动right指针到下一个t中出现的字符，然后重复
            上述过程。
            5.当right到s末尾时，得出答案。
         */
        if (s == null || t == null || s.length() == 0 || t.length() == 0) {
            return "";
        }
        // count
        int matchCount = 0;
        String res = "";
        // t对应的数组
        int[] tArr = new int[256];
        for (char c : t.toCharArray()) {
            tArr[c]++;
        }

        // s对应的数组
        int[] sArr = new int[256];
        int left = findNextStrIdx(0, s, tArr);
        if (left == s.length()) {
            return "";
        }
        // 初始化right指针，跟left一个位置
        int right = left;

        // 步骤5的条件约束
        while (right < s.length()) {
            // 步骤3
            char rightChar = s.charAt(right);
            if (sArr[rightChar] < tArr[rightChar]) {
                matchCount++;
            }
            sArr[rightChar]++;
            // 步骤3最后部分寻找最优解的说明
            while (left < s.length() && matchCount == t.length()) {
                // 要替换的情况：当res为空或者当前res的长度大于新的长度
                if (res.isEmpty() || res.length() > right - left + 1) {
                    res = s.substring(left, right + 1);
                }
                // 步骤4
                char leftChar = s.charAt(left);
                if (sArr[leftChar] <= tArr[leftChar]) {
                    matchCount--;
                }
                sArr[leftChar]--;
                left = findNextStrIdx(left + 1, s, tArr);
            }
            right = findNextStrIdx(right + 1, s, tArr);
        }
        return res;
    }

    /**
     * 76题帮助函数
     * 找到s中下一个包含t中字符的位置
     *
     * @param start
     * @param s
     * @param tArr
     * @return
     */
    private int findNextStrIdx(int start, String s, int[] tArr) {
        while (start < s.length()) {
            char c = s.charAt(start);
            if (tArr[c] != 0) {
                return start;
            }
            start++;
        }
        return start;
    }

    /**
     * LeetCode.72 编辑距离
     * <p>
     * 给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数。
     * 你可以对一个单词进行如下三种操作：
     * 插入一个字符
     * 删除一个字符
     * 替换一个字符
     *  
     * 示例 1：
     * 输入：word1 = "horse", word2 = "ros"
     * 输出：3
     * 解释：
     * horse -> rorse (将 'h' 替换为 'r')
     * rorse -> rose (删除 'r')
     * rose -> ros (删除 'e')
     * <p>
     * 示例 2：
     * 输入：word1 = "intention", word2 = "execution"
     * 输出：5
     * 解释：
     * intention -> inention (删除 't')
     * inention -> enention (将 'i' 替换为 'e')
     * enention -> exention (将 'n' 替换为 'x')
     * exention -> exection (将 'n' 替换为 'c')
     * exection -> execution (插入 'u')
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        /*
            两个字符串匹配的一类题目，通常使用动态规划。
            1.state: dp[i][j]: 在word1中取前i个字符，在word2中取前j个字符，要使两个匹配，
            最少需要的步数是多少(i和j从0开始，所以数组长度长宽都加1)
            2.init: 当i或j某一个为0，另外一个可以通过增删改若干次使得两个匹配，步数就是当
            前位置的下标。
            3.func：
            ①当word1[i] == word2[j]，例如"12b"和"ab"
            第一种匹配："12b"和"a"，需要加上1，因为第二个字符串要去掉'b'才能从"ab"变成"b"
            第二种匹配："12"和"ab"，需要加上1，因为"12"里少个'b'
            第三种匹配：'12'和'a'，都加个'b'就行，不用额外加1
            上述三种匹配是之前的三个方向的状态量，所以dp[i][j] = min{dp[i-1][j-1], dp[i-1][j]+1, dp[i][j-1]+1}
            ②当word1[i] != word2[j]，①中的第三种匹配情况也要加1.
            dp[i][j] = min{dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1}
            4.result: 右下角的值，dp[m][n]
         */
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 后面要取最小，先设置成最大值
                dp[i][j] = Integer.MAX_VALUE;
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j] + 1), dp[i][j - 1] + 1);
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1] + 1, dp[i - 1][j] + 1), dp[i][j - 1] + 1);
                }
            }
        }

        return dp[m][n];
    }

    /**
     * LeetCode.84 柱状图中最大的矩形
     * <p>
     * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1。
     * 求在该柱状图中，能够勾勒出来的矩形的最大面积。
     * <p>
     * 示例:
     * 输入: [2,1,5,6,2,3]
     * 输出: 10
     *
     * @param heights
     * @return
     */
    public int largestRectangleArea(int[] heights) {
        /*
             找到某个特定位置，看该位置之前所能组成的最大面积。
             过程：
             判断该位置跟下一个位置的高低关系，如果低于下一个，则移动指针到下一个，再继续下一次判断。
             如果高于下一个，说明以当前位置为最右边界，则以该位置为右边界所围成的面积可能是最大。
             则以该位置开始向左遍历来计算面积。面积是高度（遍历当前位置及之前的位置高度）
             * 宽度（遍历的值）
             然后取最大。
             计算结束后继续向右移，开始下一次计算。
             时间复杂度O(N²)
         */
        if (heights == null || heights.length == 0) {
            return 0;
        }
        int max = 0;
        for (int curr = 0; curr < heights.length; curr++) {
            // 当curr到最后或者前面的值比后面的值大，进行面积计算
            if (curr == heights.length - 1 || heights[curr] > heights[curr + 1]) {
                int minHeight = heights[curr];
                // 往左遍历，计算面积
                for (int idx = curr; idx >= 0; idx--) {
                    minHeight = Math.min(minHeight, heights[idx]);
                    max = Math.max(max, minHeight * (curr - idx + 1));
                }
            }
        }
        return max;
    }

    /**
     * LeetCode.85 最大矩形
     * <p>
     * 给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
     * <p>
     * 示例:
     * 输入:
     * [
     * ["1","0","1","0","0"],
     * ["1","0","1","1","1"],
     * ["1","1","1","1","1"],
     * ["1","0","0","1","0"]
     * ]
     * 输出: 6
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        /*
            以数组的每行，往上看。累加每列的值，绘制出一个高度的直方图。
            所以该题转化成了84题 柱状图的最大矩形。不断的找最大的面积，就是答案

            绘制直方图（某一行）：
            如果当前下标的值是0，高度就是0。如果下标的值不是0，就加上之前对应位置
            的值。
         */
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int row = matrix.length;
        int col = matrix[0].length;
        // 高度
        int[] height = new int[col];
        int res = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == '1') {
                    height[j]++;
                } else {
                    height[j] = 0;
                }
            }
            res = Math.max(res, largestRectangleArea(height));
        }
        return res;
    }

    /**
     * LeetCode.128 最长连续序列
     * <p>
     * 给定一个未排序的整数数组，找出最长连续序列的长度。
     * 要求算法的时间复杂度为 O(n)。
     * <p>
     * 示例:
     * 输入: [100, 4, 200, 1, 3, 2]
     * 输出: 4
     * 解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
     *
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        /*
            方法：用一个HashSet存所有数字，再考虑数组中每个数，例如当前数是n，
            先看一下n-1是否存在，如果不存在，那么从n开始往后的序列就是我们要的。
            否则跳过
         */
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }

        for (int i = 0; i < nums.length; i++) {
            int currNum = nums[i];
            // 判断n-1是否存在
            if (!set.contains(currNum - 1)) {
                int count = 0;
                while (set.contains(currNum)) {
                    count++;
                    currNum += 1;
                }
                res = Math.max(res, count);
            }
        }
        return res;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    public TreeNode(int val) {
        this.val = val;
    }
}

class ListNode {
    int val;
    ListNode next;

    public ListNode(int val) {
        this.val = val;
    }
}
