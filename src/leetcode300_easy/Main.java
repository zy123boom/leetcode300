package leetcode300_easy;

import java.util.*;

/**
 * Leetcode前300道，简单题
 *
 * @author boomzy
 * @date 2020/2/1 10:45
 */
public class Main {

    /**
     * LeetCode.1 两数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement) && map.get(complement) != i) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException();
    }

    /**
     * LeetCode.7 整数翻转
     *
     * @param x
     * @return
     */
    public int reverse(int x) {
        /*
            需要检查overflow，因为Integer范围翻转后可能超出范围
            原公式： 原来的数 * 10 + 尾数 = 新的数。
            变换得：（新的数 - 尾数） / 10 = 原来的数。

            如果溢出，则 新的数不等于原来的数
         */
        int rev = 0;
        while (x != 0) {
            int newrev = rev * 10 + x % 10;
            if ((newrev - x % 10) / 10 != rev) {
                // 溢出
                return 0;
            }
            rev = newrev;
            x /= 10;
        }
        return rev;
    }

    /**
     * LeetCode.9 回文数
     * 判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
     * <p>
     * 示例 1:
     * 输入: 121
     * 输出: true
     * <p>
     * 示例 2:
     * 输入: -121
     * 输出: false
     * 解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
     *
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if ((x < 0) || x % 10 == 0 && x != 0) {
            return false;
        }
        int result = 0;
        while (x > result) {
            result = result * 10 + (x % 10);
            x /= 10;
        }
        return x == result || x == result / 10;
    }

    /**
     * Leetcode.20 有效的括号
     * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
     * <p>
     * 有效字符串需满足：
     * <p>
     * 左括号必须用相同类型的右括号闭合。
     * 左括号必须以正确的顺序闭合。
     * 注意空字符串可被认为是有效字符串。
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        /*
            用Stack记录正向的括号，一个curr记录pop()出来的。每次遇到一个反括号，pop然后对比，
            是一对，就继续。否则直接false。最后检查栈是否为空
         */
        Stack<Character> mark = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{') {
                mark.push(s.charAt(i));
            } else if (s.charAt(i) == ')' || s.charAt(i) == ']' || s.charAt(i) == '}') {
                if (mark.isEmpty()) {
                    return false;
                }
                char cur = mark.pop();
                if (cur == '(' && s.charAt(i) != ')') {
                    return false;
                }
                if (cur == '[' && s.charAt(i) != ']') {
                    return false;
                }
                if (cur == '{' && s.charAt(i) != '}') {
                    return false;
                }
            }
        }
        return mark.isEmpty();
    }

    /**
     * LeetCode.21 合并两个有序链表
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        ListNode head = null;
        if (l1.val < l2.val) {
            head = l1;
            head.next = mergeTwoLists(l1.next, l2);
        } else {
            head = l2;
            head.next = mergeTwoLists(l1, l2.next);
        }

        return head;
    }

    /**
     * leetCode.38  外观数列
     * 「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。前五项如下：
     * <p>
     * 1.     1
     * 2.     11
     * 3.     21
     * 4.     1211
     * 5.     111221
     * 1 被读作  "one 1"  ("一个一") , 即 11。
     * 11 被读作 "two 1s" ("两个一"）, 即 21。
     * 21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。
     * <p>
     * 给定一个正整数 n（1 ≤ n ≤ 30），输出外观数列的第 n 项。
     * <p>
     * 注意：整数序列中的每一项将表示为一个字符串。
     * <p>
     * 示例 1:
     * 输入: 1
     * 输出: "1"
     * 解释：这是一个基本样例。
     * <p>
     * 示例 2:
     * 输入: 4
     * 输出: "1211"
     * 解释：当 n = 3 时，序列是 "21"，其中我们有 "2" 和 "1" 两组，"2" 可以读作 "12"，也就是出现
     * 频次 = 1 而值 = 2；类似 "1" 可以读作 "11"。所以答案是 "12" 和 "11" 组合在一起，也就是 "1211"。
     *
     * @param n
     * @return
     */
    public String countAndSay(int n) {
        /*
            方法：设置一个count一个prev，count来记录个数，初值为0。prev指当前字符的前一个字符，初始为'.'
            然后走一位count增加，当prev != count时，将字符串添加进去，即str = count + prev。然后变
            count为1，改变prev，以此类推。

            例如：1211.
            第一步：第一位为'1'，count为1。prev为'1'。count == prev，过
            第二步：第二位为'2'，此时count != prev，加字符串，str = count + prev = "11"，
                   然后将count变为1，改变prev为'2'。
            第三步：第三位为'1'，此时count != prev，加字符串，str = count + prev = "1112"
            以此类推。。。。。。
         */
        if (n <= 0) {
            return "";
        }
        String str = "1";
        for (int i = 1; i < n; i++) {
            int count = 0;
            char prev = '.';
            StringBuilder sb = new StringBuilder();
            for (int idx = 0; idx < str.length(); idx++) {
                if (str.charAt(idx) == prev || prev == '.') {
                    count++;
                } else {
                    sb.append(count + Character.toString(prev));
                    count = 1;
                }
                prev = str.charAt(idx);
            }
            // 加上最后一组的prev和count
            sb.append(count + Character.toString(prev));
            str = sb.toString();
        }
        return str;
    }

    /**
     * LeetCode.53 最大子序列的和
     *
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        /*
            maxToCurr=max{maxToCurr+nums[i], nums[i]}
            max=max{max, maxToCurr}
        */
        int maxToCurr = nums[0];
        int sum = nums[0];
        for (int i = 1; i < nums.length; i++) {
            maxToCurr = Math.max(maxToCurr + nums[i], nums[i]);
            sum = Math.max(sum, maxToCurr);
        }
        return sum;
    }

    /**
     * LeetCode.66 加一
     * 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
     * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
     * 你可以假设除了整数 0 之外，这个整数不会以零开头。
     * <p>
     * 示例 1:
     * 输入: [1,2,3]
     * 输出: [1,2,4]
     * 解释: 输入数组表示数字 123。
     * <p>
     * 示例 2:
     * 输入: [4,3,2,1]
     * 输出: [4,3,2,2]
     * 解释: 输入数组表示数字 4321。
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        //非9加1
        for (int i = n - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            }
            //逢9进0
            digits[i] = 0;
        }

        //全是9
        int[] result = new int[n + 1];
        result[0] = 1;
        return result;
    }

    /**
     * LeetCode.67 二进制求和
     *
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        // 都从最后一位开始，最后一位是低位
        int n1 = a.length() - 1;
        int n2 = b.length() - 1;
        StringBuilder sb = new StringBuilder();
        // 进位
        int carry = 0;
        while (n1 >= 0 && n2 >= 0) {
            // 某个字符串的数：str.charAt(ch) - '0'
            int sum = a.charAt(n1) - '0' + b.charAt(n2) - '0' + carry;
            carry = sum / 2;
            sum %= 2;
            sb.append(sum);
            n1--;
            n2--;
        }
        // 处理一个字符串已经走完另一个没走完的情况
        while (n1 >= 0) {
            int sum = a.charAt(n1) - '0' + carry;
            carry = sum / 2;
            sum %= 2;
            sb.append(sum);
            n1--;
        }
        while (n2 >= 0) {
            int sum = b.charAt(n2) - '0' + carry;
            carry = sum / 2;
            sum %= 2;
            sb.append(sum);
            n2--;
        }
        if (carry > 0) {
            // 有进位，进位加上
            sb.append(carry);
        }

        // 结果需要逆序
        return sb.reverse().toString();
    }

    /**
     * LeetCode.69 x的平方根
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        // 采用二分法，答案的区间在1~N(根号下Integer.MAX_VALUE)，取中点t1，看t1*t1 和 x 是否相等
        if (x <= 0) {
            return 0;
        }
        // 得到“N”
        int magicNum = (int) Math.sqrt(Integer.MAX_VALUE);
        // 二分
        int start = 1, end = magicNum;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (mid * mid == x) {
                return mid;
            }
            if (mid * mid > x) {
                end = mid;
            } else {
                start = mid;
            }
        }

        if (end * end <= x) {
            return end;
        } else {
            return start;
        }
    }

    /**
     * LeetCode.70 爬楼梯
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        /*
            方法：
            由题意，第i阶可以得到的方式有：①在第i-1阶后向上爬一阶 ②在第i-2阶后向上爬两阶
            所以，能达到的就是两个情况的和。
            dp[i]:能到达i阶的总数，则有dp[i] = dp[i - 1] + dp[i - 2]。最终结果为dp[n]。
            动态规划的思想进行转化得到斐波那契Fib(n) = Fib(n - 1) + Fib(n - 2)。此题也
            可以使用动态规划的方法。
         */

        // 以下为dp的方法
        /*if (n == 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];*/

        if (n == 1) {
            return n;
        }
        int pre2 = 1, pre1 = 2;
        for (int i = 3; i <= n; i++) {
            int cur = pre1 + pre2;
            pre2 = pre1;
            pre1 = cur;
        }
        return pre1;
    }

    /**
     * LeetCode.107 二叉树层次遍历II
     * <p>
     * 给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
     * <p>
     * 例如：
     * 给定二叉树 [3,9,20,null,null,15,7],
     * <p>
     * 3
     * / \
     * 9  20
     * /   \
     * 15    7
     * 返回其自底向上的层次遍历为：
     * <p>
     * [
     * [15,7],
     * [9,20],
     * [3]
     * ]
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        /*
         * 思路同层次遍历，唯一区别在于：遍历完每一层时，放入结果集时采用头插入addFirst
         * 这样输出结果集时，底层在上。
         */
        if (root == null) {
            return new LinkedList<>();
        }
        LinkedList<List<Integer>> res = new LinkedList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        // 首先将根结点root入队
        queue.offer(root);
        while (!queue.isEmpty()) {
            // 代表每一层
            List<Integer> list = new LinkedList<>();
            // 代表上一回合的长度
            int length = queue.size();
            while (length > 0) {
                // 出队
                TreeNode node = queue.poll();
                // 看有没有左右孩子
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
                list.add(node.val);
                length--;
            }
            res.addFirst(list);
        }
        return res;
    }

    /**
     * LeetCode.108 将有序数组转换为二叉搜索树
     * 将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
     * <p>
     * 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
     * <p>
     * 示例:
     * <p>
     * 给定有序数组: [-10,-3,0,5,9],
     * <p>
     * 一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：
     * <p>
     * 0
     * /  \
     * -3    9
     * /   /
     * -10  5
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        /*
            按照二分的思想。中点作为根节点。
            中点左边的数作为左子树，中点右边的数作为右子树。
         */
        if (nums == null || nums.length == 0) {
            return null;
        }
        return help(nums, 0, nums.length - 1);
    }

    private TreeNode help(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = help(nums, start, mid - 1);
        node.right = help(nums, mid + 1, end);
        return node;
    }

    /**
     * LeetCode.110 二叉平衡树
     * 给定一个二叉树，判断它是否是高度平衡的二叉树。
     * <p>
     * 本题中，一棵高度平衡二叉树定义为：
     * 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
     *
     * @param root
     * @return
     */
    public boolean isBalanced(TreeNode root) {
        /*
            做两个判断：第一，按照题目的二叉树定义。第二，保证每个节点的左右子树高度差绝对值
            都不可以大于1。
         */
        if (root == null) {
            return true;
        }
        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        // 判断平衡二叉树定义
        if (Math.abs(leftHeight - rightHeight) > 1) {
            return false;
        }
        // 判断每个节点
        return isBalanced(root.left) && isBalanced(root.right);
    }

    /**
     * 110题帮助函数，得到平衡二叉树高度
     * @param root
     * @return
     */
    private int getHeight(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        return Math.max(leftHeight, rightHeight) + 1;
    }

    /**
     * LeetCode.112 路径总和
     * <p>
     * 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和
     *
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        sum -= root.val;
        if ((root.left == null) && (root.right == null)) {
            return sum == 0;
        }
        return hasPathSum(root.left, sum) || hasPathSum(root.right, sum);
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