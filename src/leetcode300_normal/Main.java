package leetcode300_normal;


import com.sun.xml.internal.bind.v2.model.core.ID;

import java.util.*;

/**
 * leetcode前300道，中等题
 * 题目来源：力扣（LeetCode）
 * 链接：https://leetcode-cn.com/problemset/algorithms/?difficulty=%E4%B8%AD%E7%AD%89
 *
 * @author boomzy
 * @date 2020/2/1 10:48
 */
public class Main {

    /**
     * LeetCode.2 两数相加
     * 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，
     * 并且它们的每个节点只能存储 一位 数字。
     * <p>
     * 如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
     * <p>
     * 您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
     * <p>
     * 示例：
     * 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
     * 输出：7 -> 0 -> 8
     * 原因：342 + 465 = 807
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        /*
            设置一个dummy，让cur指向dummy，然后链表非空进行计算。
            和sum = l1.val + l2.val + 进位carry。
            当前节点值val = sum % 10，进位carry = sum / 10。
            做完一次后curr节点的指针指向下一个节点。最后的结果是dummy节点
            指向的链表即dummy.next
         */
        ListNode dummy = new ListNode(0);
        int sum = 0;
        ListNode cur = dummy;
        ListNode p1 = l1, p2 = l2;
        while (p1 != null || p2 != null) {
            if (p1 != null) {
                sum += p1.val;
                p1 = p1.next;
            }
            if (p2 != null) {
                sum += p2.val;
                p2 = p2.next;
            }
            cur.next = new ListNode(sum % 10);
            sum /= 10;
            cur = cur.next;
        }
        if (sum == 1) {
            cur.next = new ListNode(1);
        }
        return dummy.next;
    }

    /**
     * LeetCode.3 无重复字符的最长子串
     * 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
     * <p>
     * 示例 1:
     * 输入: "abcabcbb"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
     * <p>
     * 示例 2:
     * 输入: "bbbbb"
     * 输出: 1
     * 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
     * <p>
     * 示例 3:
     * 输入: "pwwkew"
     * 输出: 3
     * 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     *      请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        /*
            方法一: HashSet+双指针。
            使用双指针，快指针j，慢指针i。让j走，如果集合里不包含
            当前字符，则加到字符里。如果包含了，则走慢指针i，判断
            i指针对应的字符是否出现在集合里，出现则移除掉，同时指针
            右移。最后看set的size。即res = max{res, set.size()}
            代码如下：
            Set<Character> set = new HashSet<>();
            int res = 0;
            for (int i = 0, j = 0; j < s.length(); j++) {
            while (set.contains(s.charAt(j))) {
                set.remove(s.charAt(i));
                i++;
            }
            set.add(s.charAt(j));
            res = Math.max(res, set.size());
            }
            return res;

            方法二：HashMap+双指针(优化)。
            方法一需要让慢指针每次走一格，比较次数过多。
            使用双指针，快指针j，慢指针i。让j走，如果集合里不包含
            当前字符，则加到字符里。如果包含了，直接让慢指针走到下
            标为重复字符的下标加1。但是如果重复在慢指针指的之前出现
            了，则判断一下有没有必要回去了。即：
            i被移动到的新位置为max{i, 重复字符出现的位置+1}
            代码见正文
         */
        // map记录上次出现某字符的位置
        Map<Character, Integer> map = new HashMap<>();
        int res = 0;
        for (int i = 0, j = 0; j < s.length(); j++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(i, s.charAt(j) + 1);
            }
            map.put(s.charAt(j), j);
            res = Math.max(res, j - i + 1);
        }
        return res;
    }

    /**
     * LeetCode.5 最长回文子串
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        /*
            从中间开始，向两边看，如果两边一样继续扩散。如果一端到头，得出结果。
            奇数时看字符串，偶数时看两个字符的中间
         */
        if (s.length() == 0) {
            return s;
        }
        // 最大回文子串长度
        int max = 0;
        // 左边界右边界
        int ll = 0, rr = 0;
        for (int i = 0; i < s.length(); i++) {
            // 字符串为奇数长度的情况
            int l = i - 1;
            int r = i + 1;
            while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
                // 子串长度
                int len = r - l + 1;
                if (len > max) {
                    max = len;
                    ll = l;
                    rr = r;
                }
                r++;
                l--;
            }
            // 字符串为偶数长度的情况
            l = i;
            r = i + 1;
            while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
                int len = r - l + 1;
                if (len > max) {
                    max = len;
                    ll = l;
                    rr = r;
                }
                r++;
                l--;
            }
        }
        return s.substring(ll, rr + 1);
    }

    /**
     * LeetCode.6 Z字形变换
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        /*
            将变换前的字符串存储在若干StringBuilder里，numRows是多少就有多少
            的StringBuilder。然后每行放到一个StringBuilder里，最后的结果就是
            每行每行的拼起来。注意，numRows从1开始。
         */
        if (numRows <= 1) {
            return s;
        }
        StringBuilder[] sb = new StringBuilder[numRows];
        for (int i = 0; i < numRows; i++) {
            sb[i] = new StringBuilder();
        }
        // 遍历字符串
        int index = 0;
        while (index < s.length()) {
            // Z字形往下的过程
            for (int i = 0; i < numRows && index < s.length(); i++) {
                sb[i].append(s.charAt(index));
                index++;
            }
            // Z字形往斜上的过程，相当于只是中间行
            for (int i = numRows - 2; i > 0 && index < s.length(); i--) {
                sb[i].append(s.charAt(index));
                index++;
            }
        }
        for (int i = 1; i < numRows; i++) {
            sb[0].append(sb[i]);
        }
        return sb[0].toString();
    }

    /**
     * LeetCode.8 字符串转换整数
     * 请你来实现一个 atoi 函数，使其能将字符串转换成整数。
     * <p>
     * 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
     * 当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数
     * 字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续
     * 的数字字符组合起来，形成整数。
     * <p>
     * 该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们
     * 对于函数不应该造成影响。
     * <p>
     * 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串
     * 仅包含空白字符时，则你的函数不需要进行转换。
     * <p>
     * 在任何情况下，若函数不能进行有效的转换时，请返回 0。
     * <p>
     * 说明：
     * 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。
     * 如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
     *
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        /*
            方法：
            1.找到一个不是空白的字符，去空白
            2.如果第一个字符是正负号，记录下来
            3.查看每一位，如果是数字就记录下来，直到出现第一个不是数字的字符。
            注意：
            1.如果一上来不是'+'/'-'/数字，返回0
            2.如果数字超过32位，返回Integer最大值或最小值
         */
        if (str == null) {
            return 0;
        }
        str = str.trim();
        if (str.length() == 0) {
            return 0;
        }
        // 符号位
        int sign = 1;
        int index = 0;
        if (str.charAt(index) == '+') {
            index++;
        } else if (str.charAt(index) == '-') {
            sign = -1;
            index++;
        }

        long num = 0;
        for (; index < str.length(); index++) {
            if (str.charAt(index) < '0' || str.charAt(index) > '9') {
                break;
            }
            num = num * 10 + (str.charAt(index) - '0');
            if (num > Integer.MAX_VALUE) {
                break;
            }
        }

        if (num * sign > Integer.MAX_VALUE) {
            return Integer.MAX_VALUE;
        } else if (num * sign < Integer.MIN_VALUE) {
            return Integer.MIN_VALUE;
        } else {
            return (int) num * sign;
        }
    }

    /**
     * LeetCode.11 盛最多水的容器
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        /*
            双指针算法，计算出当前面积，然后移动left指针，看left指针的值是不是比
            right指针的小，如果小，就left++,反之right--。如果相等，两个指针都变。
         */
        if (height == null || height.length < 2) {
            return 0;
        }
        int area = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            area = Math.max(area, (right - left) * Math.min(height[left], height[right]));
            if (height[left] < height[right]) {
                left++;
            } else if (height[left] > height[right]) {
                right--;
            } else {
                left++;
                right--;
            }
        }
        return area;
    }

    /**
     * LeetCode.12 整数转罗马数字
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        /*
            暴力做法，最容易理解，全部列出来然后拼
         */
        String str = "";
        String[] digit = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        String[] ten = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] hund = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] thou = {"", "M", "MM", "MMM"};
        return str + thou[num / 1000] + hund[num % 1000 / 100] + ten[num % 100 / 10] + digit[num % 10];
    }

    /**
     * LeetCode.15 三数之和
     * 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，
     * 使得 a + b + c = 0 ？找出所有满足条件且不重复的三元组。
     * <p>
     * 注意：答案中不可以包含重复的三元组。
     * <p>
     * 示例：
     * <p>
     * 给定数组 nums = [-1, 0, 1, 2, -1, -4]，
     * <p>
     * 满足要求的三元组集合为：
     * [
     * [-1, 0, 1],
     * [-1, -1, 2]
     * ]
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        /*
            类似两数之和，固定一个数，找出另外两个数之和等于负的固定数
            先排序，再固定一个数，用双指针找另外两个。用Set去重
         */
        Arrays.sort(nums);
        int n = nums.length;
        Set<List<Integer>> set = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            // nums[i]被固定
            int l = i + 1;
            int r = n - 1;
            while (l < r) {
                if (nums[i] + nums[l] + nums[r] == 0) {
                    set.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    l++;
                    r--;
                } else if (nums[i] + nums[l] + nums[r] < 0) {
                    l++;
                } else {
                    r--;
                }
            }
        }

        List<List<Integer>> res = new ArrayList<>();
        res.addAll(set);
        return res;
    }

    /**
     * LeetCode.16 最接近的三数之和
     * 给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，
     * 使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
     * <p>
     * 例如，给定数组 nums = [-1，2，1，-4], 和 target = 1.
     * 与 target 最接近的三个数的和为 2. (-1 + 2 + 1 = 2).
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(int[] nums, int target) {
        /*
            做法类似 三数之和
         */
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];
        int n = nums.length;
        int delta = 0;
        for (int i = 0; i < n; i++) {
            int l = i + 1;
            int r = n - 1;
            while (l < r) {
                if (Math.abs(nums[i] + nums[l] + nums[r] - target) < Math.abs(res - target)) {
                    res = nums[i] + nums[l] + nums[r];
                }
                if (nums[i] + nums[l] + nums[r] < target) {
                    l++;
                } else if (nums[i] + nums[l] + nums[r] > target) {
                    r--;
                } else {
                    return res;
                }
            }
        }
        return res;
    }

    /**
     * LeetCode.17 电话号码的字母组合
     * <p>
     * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
     * 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
     * 2->abc, 3->def 4->ghi ... 9->wxyz
     *
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        /*
            首先用一个map存储对应关系，然后用DFS实现
         */
        Map<Character, char[]> map = new HashMap<>();
        map.put('2', new char[]{'a', 'b', 'c'});
        map.put('3', new char[]{'d', 'e', 'f'});
        map.put('4', new char[]{'g', 'h', 'i'});
        map.put('5', new char[]{'j', 'k', 'l'});
        map.put('6', new char[]{'m', 'n', 'o'});
        map.put('7', new char[]{'p', 'q', 'r', 's'});
        map.put('8', new char[]{'t', 'u', 'v'});
        map.put('9', new char[]{'w', 'x', 'y', 'z'});

        List<String> res = new ArrayList<>();
        if (digits.length() == 0) {
            return res;
        }
        dfs(digits, map, res, 0, "");
        return res;
    }


    /**
     * dfs
     *
     * @param digits
     * @param map
     * @param res
     * @param start  初始位置
     * @param cur    当前的字符
     */
    private void dfs(String digits, Map<Character, char[]> map, List<String> res, int start, String cur) {
        if (start >= digits.length()) {
            res.add(cur);
            return;
        }
        for (char c : map.get(digits.charAt(start))) {
            dfs(digits, map, res, start + 1, cur + c);
        }
    }

    /**
     * LeetCode.18 四数之和
     * <p>
     * 给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在
     * 四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满
     * 足条件且不重复的四元组。
     * <p>
     * 注意：
     * 答案中不可以包含重复的四元组。
     * <p>
     * 示例：
     * 给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。
     * 满足要求的四元组集合为：
     * [
     * [-1,  0, 0, 1],
     * [-2, -1, 1, 2],
     * [-2,  0, 0, 2]
     * ]
     *
     * @param nums
     * @param target
     * @return
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        /*
            类似三数之和，先排序。思想是两个for循环里嵌套一个两数之和
         */
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < nums.length - 3; i++) {
            // 重复情况
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; j++) {
                // 重复情况
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                int l = j + 1;
                int r = nums.length - 1;
                // 这个target是里面两数之和的target
                int curTarget = target - nums[i] - nums[j];
                while (l < r) {
                    int sum = nums[l] + nums[r];
                    if (sum == curTarget) {
                        // 将四个数加入res
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[i]);
                        list.add(nums[j]);
                        list.add(nums[l]);
                        list.add(nums[r]);
                        res.add(list);
                        l++;
                        while (l < r && nums[l] == nums[l - 1]) {
                            l++;
                        }
                        r--;
                        while (l < r && nums[r] == nums[r + 1]) {
                            r--;
                        }
                    } else if (sum < curTarget) {
                        l++;
                    } else {
                        r--;
                    }
                }
            }
        }
        return res;
    }

    /**
     * LeetCode.19 删除链表的倒数第N个节点
     * 给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
     * <p>
     * 示例：
     * <p>
     * 给定一个链表: 1->2->3->4->5, 和 n = 2.
     * 当删除了倒数第二个节点后，链表变为 1->2->3->5.
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        /*
            快慢指针法。
            首先让快指针移动n个位置，然后同时移动快慢指针，此时慢指针指向要
            移除的前一个元素，然后删除这个元素，即node.next = node.next.next
         */
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy;
        ListNode slow = dummy;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }

    /**
     * LeetCode.22 括号生成
     * <p>
     * 给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
     * <p>
     * 例如，给出 n = 3，生成结果为：
     * <p>
     * [
     * "((()))",
     * "(()())",
     * "(())()",
     * "()(())",
     * "()()()"
     * ]
     *
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        /*
            此类问题一般使用dfs。
            当left<n，说明左括号没有满，加左括号
            当left>right，说明左括号足够了，缺少右括号配对，所以加右括号
         */
        List<String> res = new ArrayList<>();
        if (n <= 0) {
            return res;
        }
        dfs(new StringBuilder(), res, n, 0, 0);
        return res;
    }

    /**
     * 22题帮助函数
     *
     * @param curr
     * @param res
     * @param n
     * @param left  左括号数
     * @param right 右括号数
     */
    private void dfs(StringBuilder curr, List<String> res, int n, int left, int right) {
        if (right == n) {
            res.add(curr.toString());
            return;
        }
        if (left < n) {
            dfs(curr.append("("), res, n, left + 1, right);
            curr.deleteCharAt(curr.length() - 1);
        }
        if (left > right) {
            dfs(curr.append(")"), res, n, left, right + 1);
            curr.deleteCharAt(curr.length() - 1);
        }
    }

    /**
     * LeetCode.24 两两交换链表中的节点
     * <p>
     * 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
     * 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
     * <p>
     * 示例:
     * 给定 1->2->3->4, 你应该返回 2->1->4->3.
     *
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        /*
            例如dummy->1->2->3->4
            1.让1指向3
            2.让2指向1
            3.此时2为第一个节点了，让当前节点指向2
            4.当前节点后移两位
         */
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode cur = dummy;
        while (cur.next != null && cur.next.next != null) {
            // 节点"1"，叫next是因为前面有个dummy
            ListNode next = cur.next;
            // 节点"2"
            ListNode nextnext = cur.next.next;
            // 步骤1234
            next.next = nextnext.next;
            nextnext.next = next;
            cur.next = nextnext;
            cur = cur.next.next;
        }
        return dummy.next;
    }

    /**
     * leetcode.29 两数相除
     * 给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
     * 返回被除数 dividend 除以除数 divisor 得到的商。
     * <p>
     * 示例 1:
     * 输入: dividend = 10, divisor = 3
     * 输出: 3
     * <p>
     * 示例 2:
     * 输入: dividend = 7, divisor = -3
     * 输出: -2
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        /*
            思路：例如32/3，首先3<<1，相当于乘2，结果为6，小于32，继续左移
            直到结果为24时，此时有8个3，32-24=8，然后做8/3，有2个3。然后做
            2/3，没有3了，结束。
         */
        if (divisor == 0) {
            return Integer.MAX_VALUE;
        }
        if (dividend == Integer.MIN_VALUE) {
            if (divisor == -1) {
                return Integer.MAX_VALUE;
            } else if (divisor == 1) {
                return Integer.MIN_VALUE;
            }
        }

        // 被除数
        long divd = (long) dividend;
        // 除数
        long divs = (long) divisor;
        // 结果正负号，初始给1
        int sign = 1;

        // 保证后面的操作都是正数之间的操作
        if (divd < 0) {
            divd = -divd;
            sign = -sign;
        }
        if (divs < 0) {
            divs = -divs;
            sign = -sign;
        }

        int res = 0;
        while (divd >= divs) {
            int shift = 0;
            while (divd >= divs << shift) {
                shift++;
            }
            res += (1 << (shift - 1));
            divd -= (divs << (shift - 1));
        }
        return sign * res;
    }

    /**
     * LeetCode.31 下一个排列
     * <p>
     * 实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
     * 如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
     * 必须原地修改，只允许使用额外常数空间。
     * <p>
     * 以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
     * 1,2,3 → 1,3,2
     * 3,2,1 → 1,2,3
     * 1,1,5 → 1,5,1
     *
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        /*
            题意：例如[1,4,3,2]，从后往前，1比4小，从1后面找比1大的最小的数，即
            2，然后交换两个位置，然后后面的排序。
            [1,4,3,2]->[2,1,3,4]
         */
        for (int i = nums.length - 1; i >= 0; i--) {
            if (i == 0) {
                Arrays.sort(nums);
                return;
            } else {
                if (nums[i] > nums[i - 1]) {
                    Arrays.sort(nums, i, nums.length);
                    for (int j = i; j < nums.length; j++) {
                        if (nums[j] > nums[i - 1]) {
                            int tmp = nums[j];
                            nums[j] = nums[i - 1];
                            nums[i - 1] = tmp;
                            return;
                        }
                    }
                }
            }
        }
    }

    /**
     * LeetCode.33 搜索旋转排序数组
     * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
     * <p>
     * ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
     * 搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
     * 你可以假设数组中不存在重复的元素。
     * <p>
     * 你的算法时间复杂度必须是 O(log n) 级别。
     * <p>
     * 示例 1:
     * 输入: nums = [4,5,6,7,0,1,2], target = 0
     * 输出: 4
     * <p>
     * 示例 2:
     * 输入: nums = [4,5,6,7,0,1,2], target = 3
     * 输出: -1
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        /*
            二分法
         */
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int start = 0, end = nums.length - 1;
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[start] < nums[mid]) {
                if (nums[start] <= target && target <= nums[mid]) {
                    end = mid;
                } else {
                    start = mid;
                }
            } else if (nums[mid] < nums[end]) {
                if (nums[end] >= target && target >= nums[mid]) {
                    start = mid;
                } else {
                    end = mid;
                }
            }
        }
        if (nums[start] == start) {
            return start;
        }
        if (nums[end] == end) {
            return end;
        }
        return -1;
    }

    /**
     * LeetCode.34 在排序数组中查找元素的第一个和最后一个位置
     * <p>
     * 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在
     * 数组中的开始位置和结束位置。
     * <p>
     * 你的算法时间复杂度必须是 O(log n) 级别。
     * 如果数组中不存在目标值，返回 [-1, -1]。
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        /*
            由于复杂度O(log n)，所以使用二分。做两次二分来分别
            找出起始点和终止点
            两个大体相同，区别是找起始点的时候，当nums[mid]==target时，移动右指针。
            而找终止点的时候，移动的是左指针。
         */
        int[] res = {-1, -1};
        if (nums == null || nums.length == 0) {
            return res;
        }
        int startPoint = -1, endPoint = -1;
        int start = 0, end = nums.length - 1;

        // 找起始点
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] >= target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (nums[start] == target) {
            startPoint = start;
        } else if (nums[end] == target) {
            startPoint = end;
        }
        if (startPoint == -1) {
            return res;
        }

        start = 0;
        end = nums.length - 1;
        // 找终止点
        while (start + 1 < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] > target) {
                end = mid;
            } else {
                start = mid;
            }
        }
        if (nums[end] == target) {
            endPoint = end;
        } else if (nums[start] == target) {
            endPoint = start;
        }
        res[0] = startPoint;
        res[1] = endPoint;
        return res;
    }

    /**
     * LeetCode.39 组合总和
     * 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中
     * 所有可以使数字和为 target 的组合。
     * candidates 中的数字可以无限制重复被选取。
     * <p>
     * 说明：
     * 所有数字（包括 target）都是正整数。
     * 解集不能包含重复的组合。 
     * <p>
     * 示例 1:
     * 输入: candidates = [2,3,6,7], target = 7,
     * 所求解集为:
     * [
     * [7],
     * [2,2,3]
     * ]
     * <p>
     * 示例 2:
     * 输入: candidates = [2,3,5], target = 8,
     * 所求解集为:
     * [
     *   [2,2,2,2],
     *   [2,3,3],
     *   [3,5]
     * ]
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        /*
            该类题采用DFS
         */
        List<List<Integer>> res = new ArrayList<>();
        if (candidates == null || candidates.length == 0 || target == 0) {
            return res;
        }
        Arrays.sort(candidates);
        dfs(candidates, target, 0, res, new ArrayList<Integer>());
        return res;
    }

    /**
     * LeetCode.39帮助函数
     * 通过dfs，看数组candidates的值跟target的关系。如果candidates[i] > target，
     * 说明不可能凑出来，直接break。否则就说明有可能，然后继续进行backtracking。
     *
     * @param candidates
     * @param target
     * @param index      当前需要取的数的下标，第一轮是0
     * @param res
     * @param curr       当前得到的组合总和
     */
    private void dfs(int[] candidates, int target, int index, List<List<Integer>> res, List<Integer> curr) {
        if (target == 0) {
            res.add(new ArrayList<>(curr));
            return;
        } else if (target > 0) {
            for (int i = index; i < candidates.length; i++) {
                if (candidates[i] > target) {
                    break;
                }
                curr.add(candidates[i]);
                dfs(candidates, target - candidates[i], i, res, curr);
                curr.remove(curr.size() - 1);
            }
        }
    }

    /**
     * LeetCode.40 组合总和II
     * 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有
     * 可以使数字和为 target 的组合。
     * candidates 中的每个数字在每个组合中只能使用一次。
     * <p>
     * 说明：
     * 所有数字（包括目标数）都是正整数。
     * 解集不能包含重复的组合。 
     * <p>
     * 示例 1:
     * 输入: candidates = [10,1,2,7,6,1,5], target = 8,
     * 所求解集为:
     * [
     * [1, 7],
     * [1, 2, 5],
     * [2, 6],
     * [1, 1, 6]
     * ]
     * <p>
     * 示例 2:
     * 输入: candidates = [2,5,2,1,2], target = 5,
     * 所求解集为:
     * [
     *   [1,2,2],
     *   [5]
     * ]
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates == null || candidates.length == 0 || target == 0) {
            return res;
        }
        Arrays.sort(candidates);
        dfs2(candidates, target, 0, res, new ArrayList<Integer>());
        return res;
    }

    /**
     * LeetCode.40帮助函数
     * <p>
     * 与组合总和I不同的是，每次判断之前，判断一下arr[index]是否等于arr[index-1]，如果相等，
     * 说明重复，重复分支不再做。并且同时当前下标i不等于index，因为是新分支。
     *
     * @param candidates
     * @param target
     * @param index
     * @param res
     * @param curr
     */
    private void dfs2(int[] candidates, int target, int index, List<List<Integer>> res, ArrayList<Integer> curr) {
        if (target == 0) {
            res.add(new ArrayList<>(curr));
            return;
        } else if (target > 0) {
            for (int i = index; i < candidates.length; i++) {
                if (i != index && candidates[i] == candidates[i - 1]) {
                    continue;
                }
                curr.add(candidates[i]);
                dfs2(candidates, target - candidates[i], i + 1, res, curr);
                curr.remove(curr.size() - 1);
            }
        }
    }

    /**
     * LeetCode.43 字符串相乘
     * 给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，
     * 它们的乘积也表示为字符串形式。
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        if (num1.length() == 0 || num2.length() == 0) {
            return "0";
        }
        int len1 = num1.length();
        int len2 = num2.length();
        int[] result = new int[len1 + len2];
        for (int i = len1 - 1; i >= 0; i--) {
            for (int j = len2 - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                // 低位和高位，例如3 * 5，低位为5，高位为进上去的1
                int posLow = i + j + 1;
                int posHigh = i + j;
                mul += result[posLow];
                result[posLow] = mul % 10;
                result[posHigh] += mul / 10;
            }
        }

        StringBuilder sb = new StringBuilder();
        for (int res : result) {
            if (!(sb.length() == 0 && res == 0)) {
                sb.append(res);
            }
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    /**
     * LeetCode.46 全排列
     * 给定一个没有重复数字的序列，返回其所有可能的全排列。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        /*
            dfs + 递归的题目，回溯思想
         */
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        dfs(res, new ArrayList<>(), nums, new boolean[nums.length]);
        return res;
    }

    /**
     * 46题帮助函数
     *
     * @param res
     * @param cur
     * @param nums
     * @param visited 判断当前数是否被遍历过了
     */
    public void dfs(List<List<Integer>> res, List<Integer> cur, int[] nums, boolean[] visited) {
        if (cur.size() == nums.length) {
            res.add(new ArrayList<>(cur));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            cur.add(nums[i]);
            visited[i] = true;
            dfs(res, cur, nums, visited);
            cur.remove(cur.size() - 1);
            visited[i] = false;
        }
    }

    /**
     * LeetCode.47 全排列II
     * 给定一个可包含重复数字的序列，返回所有不重复的全排列。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        /*
            思想：DFS + 递归， backtracking
            此处为了防止重复，做判断，如果当前下标的值与上一个相同并且上一个被访问过了，
            就直接跳过。
            此处，数组首先进行排序再进行dfs。
         */
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);
        dfs2(res, new ArrayList<>(), nums, new boolean[nums.length]);
        return null;
    }

    /**
     * 47题帮助函数
     *
     * @param res
     * @param cur
     * @param nums
     * @param visited
     */
    private void dfs2(List<List<Integer>> res, List<Integer> cur, int[] nums, boolean[] visited) {
        if (cur.size() == nums.length) {
            res.add(new ArrayList<>(cur));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (visited[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) {
                continue;
            }
            cur.add(nums[i]);
            visited[i] = true;
            dfs2(res, cur, nums, visited);
            cur.remove(cur.size() - 1);
            visited[i] = false;
        }
    }

    /**
     * LeetCode.48 翻转图像
     * 给定一个 n × n 的二维矩阵表示一个图像。
     * 将图像顺时针旋转 90 度。
     * <p>
     * 说明：
     * 你必须在原地旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要使用另一个矩阵来旋转图像。
     * <p>
     * 示例 1:
     * 给定 matrix =
     * [
     * [1,2,3],
     * [4,5,6],
     * [7,8,9]
     * ],
     * 原地旋转输入矩阵，使其变为:
     * [
     * [7,4,1],
     * [8,5,2],
     * [9,6,3]
     * ]
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        // 1.以对角线为轴进行交换
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    continue;
                }
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }

        // 2.以中轴线对称进行列交换
        for (int i = 0, j = matrix.length - 1; i < matrix.length / 2; i++, j--) {
            for (int k = 0; k < matrix.length; k++) {
                int tmp = matrix[k][i];
                matrix[k][i] = matrix[k][j];
                matrix[k][j] = tmp;
            }
        }
    }

    /**
     * LeetCode.49 字母异位词分组
     * 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
     * <p>
     * 示例:
     * <p>
     * 输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
     * 输出:
     * [
     * ["ate","eat","tea"],
     * ["nat","tan"],
     * ["bat"]
     * ]
     *
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        /*
            将每个字符串的字母排序，看是否是相同的，相同的就是异位词放在一组
         */
        // 存排好序的字符串对应哪一组
        Map<String, Integer> map = new HashMap<>();
        List<List<String>> res = new ArrayList<>();
        // 分组
        int count = 0;
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String newStr = new String(chars);
            if (!map.containsKey(newStr)) {
                // 开启新的一组
                res.add(new ArrayList<>());
                map.put(newStr, count);
                count++;
            }
            res.get(map.get(newStr)).add(str);
        }
        return res;
    }

    /**
     * LeetCode.50 Pow(x, n)
     * 实现 pow(x, n) ，即计算 x 的 n 次幂函数。
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        /*
            求x的n/2次方，因为pow(x, n) = pow(pow(x, n/2), n/2)
         */
        if (n == 0) return 1;
        if (n == 1) return x;
        if (n == -1) return 1 / x;
        double res = myPow(x, n / 2);
        res *= res;
        if (n % 2 == 1) {
            res *= x;
        } else if (n % 2 == -1) {
            res *= 1 / x;
        }
        return res;
    }

    /**
     * LeetCode.54 螺旋矩阵
     * <p>
     * 给定一个包含 m x n 个元素的矩阵（m 行, n 列），请按照顺时针螺旋顺序，返回矩阵中的所有元素。
     * <p>
     * 示例 1:
     * 输入:
     * [
     * [ 1, 2, 3 ],
     * [ 4, 5, 6 ],
     * [ 7, 8, 9 ]
     * ]
     * 输出: [1,2,3,6,9,8,7,4,5]
     *
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return new ArrayList<>();
        }
        List<Integer> res = new ArrayList<>();
        int top = 0, bottom = matrix.length - 1;
        int left = 0, right = matrix[0].length - 1;
        while (top < bottom && left < right) {
            for (int i = left; i < right; i++) {
                res.add(matrix[top][i]);
            }
            for (int i = top; i < bottom; i++) {
                res.add(matrix[i][right]);
            }
            for (int i = right; i > left; i--) {
                res.add(matrix[bottom][i]);
            }
            for (int i = bottom; i > top; i--) {
                res.add(matrix[i][left]);
            }
            left++;
            right--;
            top++;
            bottom--;
        }

        // 特殊情况1：只剩一行（含一个）
        if (top == bottom) {
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }
        } else if (left == right) { // 特殊情况2：只剩一列
            for (int i = top; i <= bottom; i++) {
                res.add(matrix[i][left]);
            }
        }
        return res;
    }

    /**
     * LeetCode.55 跳跃游戏
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        if (nums.length < 2) return true;
        int reach = 0, i = 0;
        for (i = 0; i < nums.length && i <= reach; i++) {
            reach = Math.max(nums[i] + i, reach);
            if (reach >= nums.length - 1) return true;
        }
        return false;
    }

    /**
     * LeetCode.56 合并区间
     * 给出一个区间的集合，请合并所有重叠的区间。
     * <p>
     * 示例 1:
     * 输入: [[1,3],[2,6],[8,10],[15,18]]
     * 输出: [[1,6],[8,10],[15,18]]
     * 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        /*
            定义两个变量previous和current，分别表示前一个区间和当前的区间
            如果没有融合，当前区间就变成新的previous，下一个区间成为新的current
            如果发生融合，更新前一个区间的结束时间：
         */
        // 将所有的区间按照起始时间的先后顺序排序
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));

        int[] previous = null;
        List<int[]> res = new ArrayList<>();

        for (int[] current : intervals) {
            // 第一个区间/和前一个区间没有重叠，直接加到结果
            if (previous == null || current[0] > previous[1]) {
                res.add(previous = current);
            } else {
                // 发生重叠，更新前一个区间的结束时间
                previous[1] = Math.max(previous[1], current[1]);
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    /**
     * LeetCode.59 螺旋矩阵II
     * <p>
     * 给定一个正整数 n，生成一个包含 1 到 n2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
     *
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        if (n <= 0) {
            return res;
        }
        int left = 0, right = res.length - 1;
        int top = 0, bottom = res.length - 1;
        int k = 1;
        while (left < right && top < bottom) {
            for (int i = left; i < right; i++) {
                res[top][i] = k++;
            }
            for (int i = top; i < bottom; i++) {
                res[i][right] = k++;
            }
            for (int i = right; i > left; i--) {
                res[bottom][i] = k++;
            }
            for (int i = bottom; i > top; i--) {
                res[i][left] = k++;
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        // 如果是奇数矩阵
        if (n % 2 != 0) {
            res[n / 2][n / 2] = k;
        }
        return res;
    }

    /**
     * LeetCode.60 第k个排列
     * 给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。
     * 按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
     * <p>
     * "123"
     * "132"
     * "213"
     * "231"
     * "312"
     * "321"
     * 给定 n 和 k，返回第 k 个排列。
     * <p>
     * 说明：
     * 给定 n 的范围是 [1, 9]。
     * 给定 k 的范围是[1,  n!]。
     * <p>
     * 示例 1:
     * 输入: n = 3, k = 3
     * 输出: "213"
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        /*
            有n!个不同的排列，将这些排列分成n组，每一组将会有(n-1)!个不同的排列。
            求每位：f[n] = f[n - 1] * n;
            对于k，先减1，不断的求最高位，然后在可选区间（例如首字符为1-3）内选择，然后取余。
         */
        char[] result = new char[n];
        List<Integer> nums = new ArrayList<>();
        int[] factorial = new int[n];
        factorial[0] = 1;
        for (int i = 1; i < n; i++) {
            factorial[i] = factorial[i - 1] * i;
        }
        // 可选范围
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }
        k--;
        for (int i = 0; i < n; i++) {
            // 从最高位添加一直到最低位
            result[i] = Character.forDigit(nums.remove(k / factorial[n - 1 - i]), 10);
            k = k % factorial[n - 1 - i];
        }
        return new String(result);
    }

    /**
     * LeetCode.61 旋转链表
     * 给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。
     * <p>
     * 示例 1:
     * 输入: 1->2->3->4->5->NULL, k = 2
     * 输出: 4->5->1->2->3->NULL
     * 解释:
     * 向右旋转 1 步: 5->1->2->3->4->NULL
     * 向右旋转 2 步: 4->5->1->2->3->NULL
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        /*
            快慢指针法。先让快指针移动k%n(n是链表长度)个节点。然后两个指针同时移动
            直到fast指针指到最后一个元素。然后对链表进行变换。
            1.  fast.next = head
            2.  head = slow.next
            3.  slow.next = null
         */
        if (head == null) {
            return head;
        }
        int len = 0;
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null) {
            fast = fast.next;
            len++;
        }
        fast = head;
        for (int i = 0; i < k % len; i++) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        fast.next = head;
        head = slow.next;
        slow.next = null;
        return head;
    }

    /**
     * LeetCode.62 不同路径
     * <p>
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
     * 问总共有多少条不同的路径？
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        /*
            dp问题。
            dp[i][j]: 到达(i, j)的路径数
            init: 首行首列都为1
            状态转移方程：dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
            return: dp[行数-1][列数-1]
         */
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * LeetCode.63 不同路径II
     * <p>
     * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
     * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
     * 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
     * <p>
     * 网格中的障碍物和空位置分别用 1 和 0 来表示。
     *
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1) {
                break;
            }
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1) {
                break;
            }
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * LeetCode.64 最小路径和
     * <p>
     * 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     * 说明：每次只能向下或者向右移动一步。
     *
     * @param grid
     * @return
     */
    public int minPathSum(int[][] grid) {
        /*
            dp问题。
            dp[i][j]：到达(i, j)的最小路径和
            init: 行和列等于前一个加上当前数
            方程:dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
            return: dp[行数-1][列数-1]
         */
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = dp[0][i - 1] + grid[0][i];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * LeetCode.71 简化路径
     *
     * @param path
     * @return
     */
    public String simplifyPath(String path) {
        String[] wordArr = path.split("/");

        // 保存去掉空串（例如/a//c所产生的)和"."(.代表当前路径)的剩余的
        List<String> wordList = new ArrayList<>();
        for (int i = 0; i < wordArr.length; i++) {
            if (wordArr[i].isEmpty() || wordArr[i].equals(".")) {
                continue;
            }
            wordList.add(wordArr[i]);
        }

        // 保存简化后的路径
        List<String> simpleWordList = new ArrayList<>();
        for (int i = 0; i < wordList.size(); i++) {
            if (wordList.get(i).equals("..")) {
                // 遇到".."，删除末尾的单词
                if (!simpleWordList.isEmpty()) {
                    simpleWordList.remove(simpleWordList.size() - 1);
                }
            } else {
                // 否则加到简化List里
                simpleWordList.add(wordList.get(i));
            }
        }

        String res = String.join("/", simpleWordList);
        res = "/" + res;
        return res;
    }

    /**
     * LeetCode.74 搜索二维矩阵
     * 编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：
     * 每行中的整数从左到右按升序排列。
     * 每行的第一个整数大于前一行的最后一个整数。
     * <p>
     * 示例 1:
     * 输入:
     * matrix = [
     * [1,   3,  5,  7],
     * [10, 11, 16, 20],
     * [23, 30, 34, 50]
     * ]
     * target = 3
     * 输出: true
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        /*
            首先从右上角开始，若右上角的数大于要找的数，则该数这一列的数都不可能是，
            所以向左移动再看下一个数。若下一个数还比要找的大，则这一个数的一列数都比他大
            继续向左移动。 当当前数小于要找的数，则这个数左边的数都比要找的数小，往下移
            以此类推
         */
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int row = 0, col = matrix[0].length - 1;
        while (row < matrix.length && col >= 0) {
            if (matrix[row][col] == target) {
                return true;
            }
            if (matrix[row][col] > target) {
                col--;
                continue;
            }
            if (matrix[row][col] < target) {
                row++;
                continue;
            }
        }
        return false;
    }

    /**
     * LeetCode.75 颜色分类
     * <p>
     * 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色
     * 的元素相邻，并按照红色、白色、蓝色顺序排列。
     * 此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
     * <p>
     * 注意:
     * 不能使用代码库中的排序函数来解决这道题。
     * <p>
     * 示例:
     * 输入: [2,0,2,1,1,0]
     * 输出: [0,0,1,1,2,2]
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        /*
            荷兰旗问题。创建两个指针first和last。first指向的为不是0的第一个位置。second指向
            的为不是2出现的第一个位置。然后遍历，看当前指针index指的是什么。如果是0和first交换，
            first后移。是2的话同理。当index>last时，所有都排好序了，结束。
         */
        if (nums == null || nums.length <= 1) {
            return;
        }
        int first = 0;
        while (first < nums.length && nums[first] == 0) {
            first++;
        }
        int last = nums.length - 1;
        while (last >= 0 && nums[last] == 2) {
            last--;
        }

        // 从第一个不是0的开始
        int index = first;
        while (index <= last) {
            if (nums[index] == 1) {
                index++;
            } else if (nums[index] == 0) {
                swap(nums, index, first);
                first++;
                index++;
            } else if (nums[index] == 2) {
                swap(nums, index, last);
                last--;
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    /**
     * LeetCode.77 组合
     * 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。
     * <p>
     * 示例:
     * 输入: n = 4, k = 2
     * 输出:
     * [
     * [2,4],
     * [3,4],
     * [2,3],
     * [1,2],
     * [1,3],
     * [1,4],
     * ]
     *
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        helper(1, n, k, res, new ArrayList<>());
        return res;
    }

    /**
     * LeetCode.77 帮助函数
     *
     * @param start 开始的位置
     * @param n
     * @param k
     * @param res
     * @param curr
     */
    private void helper(int start, int n, int k, List<List<Integer>> res, List<Integer> curr) {
        if (curr.size() == k) {
            res.add(new ArrayList<>(curr));
            return;
        }
        for (int i = start; i <= n - (curr.size()) + 1; i++) {
            curr.add(i);
            helper(start + 1, n, k, res, curr);
            curr.remove(curr.size() - 1);
        }
    }

    /**
     * LeetCode.78 子集
     * 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        /*
            空集是任何集合的子集，从空集开始，开始选nums数组的值，可以选也可以不选。
            如果不选，一直空下去。如果选的话，再下来的一个数也可以选也可以不选。
            最后得到的解集就是答案。
         */
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        helper(res, new ArrayList<>(), nums, 0);
        return res;
    }

    /**
     * 78题帮助函数
     *
     * @param res
     * @param cur
     * @param nums
     * @param index
     */
    private void helper(List<List<Integer>> res, List<Integer> cur, int[] nums, int index) {
        // base case
        if (index == nums.length) {
            res.add(new ArrayList<>(cur));
            return;
        }
        // not choice
        helper(res, cur, nums, index + 1);
        // choice
        cur.add(nums[index]);
        helper(res, cur, nums, index + 1);
        cur.remove(cur.size() - 1);
    }

    /**
     * LeetCode.79 单词搜索
     * <p>
     * 给定一个二维网格和一个单词，找出该单词是否存在于网格中。
     * <p>
     * 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直
     * 相邻的单元格。同一个单元格内的字母不允许被重复使用。
     * <p>
     * 示例:
     * board =
     * [
     * ['A','B','C','E'],
     * ['S','F','C','S'],
     * ['A','D','E','E']
     * ]
     * 给定 word = "ABCCED", 返回 true.
     * 给定 word = "SEE", 返回 true.
     * 给定 word = "ABCB", 返回 false.
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        /*
            搜索类题目，使用DFS
         */
        if (board == null) {
            return false;
        }
        // 判断要搜索的方向的字符是否被遍历过了
        boolean[][] used = new boolean[board.length][board[0].length];
        for (int row = 0; row < board.length; row++) {
            for (int col = 0; col < board[0].length; col++) {
                if (existHelper(board, used, word.toCharArray(), 0, row, col)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 79题帮助函数
     *
     * @param board
     * @param used  是否被访问过，true为被访问过，false未被访问过
     * @param word
     * @param index
     * @param row
     * @param col
     * @return
     */
    private boolean existHelper(char[][] board, boolean[][] used, char[] word, int index, int row, int col) {
        // base case
        if (index == word.length) {
            return true;
        }
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length) {
            return false;
        }
        if (used[row][col] == true || board[row][col] != word[index]) {
            return false;
        }

        used[row][col] = true;
        // 向上搜
        boolean exist = existHelper(board, used, word, index + 1, row - 1, col);
        if (exist) {
            return true;
        }
        // 向下搜
        exist = existHelper(board, used, word, index + 1, row + 1, col);
        if (exist) {
            return true;
        }
        // 向左搜
        exist = existHelper(board, used, word, index + 1, row, col - 1);
        if (exist) {
            return true;
        }
        // 向右搜
        exist = existHelper(board, used, word, index + 1, row, col + 1);
        if (exist) {
            return true;
        }
        used[row][col] = false;
        return false;
    }

    /**
     * LeetCode.80 删除排序数组中的重复项II
     * 给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。
     * <p>
     * 不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
     * <p>
     * 示例 1:
     * 给定 nums = [1,1,1,2,2,3],
     * 函数应返回新长度 length = 5, 并且原数组的前五个元素被修改为 1, 1, 2, 2, 3 。
     * 你不需要考虑数组中超出新长度后面的元素。
     * <p>
     * 示例 2:
     * 给定 nums = [0,0,1,1,1,1,2,3,3],
     * 函数应返回新长度 length = 7, 并且原数组的前五个元素被修改为 0, 0, 1, 1, 2, 3, 3 。
     * 你不需要考虑数组中超出新长度后面的元素。
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        /*
            双指针，loc指针代表下一个遇到的数要放在什么位置。
            遍历数组，如果当前位置index的数字不超过2个，则两个指针同时移动。如果超过了，index指针
            移动。然后此时index指的值需要移动到loc指针上，然后loc移动。以此类推。最后loc的值就是答案
         */
        if (nums == null || nums.length <= 2) {
            return nums.length;
        }
        int loc = 2;
        for (int index = 2; index < nums.length; index++) {
            if (!(nums[index] == nums[loc - 1] && nums[loc - 1] == nums[loc - 2])) {
                nums[loc++] = nums[index];
            }
        }
        return loc;
    }

    /**
     * LeetCode.81 搜索旋转排序数组II
     * 假设按照升序排序的数组在预先未知的某个点上进行了旋转。
     * ( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。
     * 编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。
     * <p>
     * 示例 1:
     * 输入: nums = [2,5,6,0,0,1,2], target = 0
     * 输出: true
     * <p>
     * 示例 2:
     * 输入: nums = [2,5,6,0,0,1,2], target = 3
     * 输出: false
     *
     * @param nums
     * @param target
     * @return
     */
    public boolean search2(int[] nums, int target) {
        /*
            与I不同的是，如果出现比较相同的情况，让start++，总会出现不一样的情况，
            可以使用I的方法。
         */
        if (nums == null || nums.length == 0) {
            return false;
        }
        int start = 0, end = nums.length - 1;
        int mid;
        while (start + 1 < end) {
            mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[mid] > nums[start]) {
                if (nums[mid] >= target && target >= nums[start]) {
                    end = mid;
                } else {
                    start = mid;
                }
            } else if (nums[mid] < nums[start]) {
                if (nums[mid] <= target && target <= nums[end]) {
                    start = mid;
                } else {
                    end = mid;
                }
            } else {
                start++;
            }
        }
        if (nums[start] == target || nums[end] == target) {
            return true;
        }
        return false;
    }

    /**
     * LeetCode.82 删除排序链表中的重复元素II
     * 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。
     * <p>
     * 示例 1:
     * 输入: 1->2->3->3->4->4->5
     * 输出: 1->2->5
     * <p>
     * 示例 2:
     * 输入: 1->1->1->2->3
     * 输出: 2->3
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        /*
            三指针。realNode是最后的答案。每次对比curNode的值和前一个，后一个比。若
            不相同则加到realNode中，同时preNode和curNode后移。如果有相同的，不加到realNode
            中，preNode和curNode后移。
         */
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(0);
        ListNode curNode = head;
        ListNode preNode = dummy;
        ListNode realNode = dummy;
        while (curNode != null) {
            if ((preNode == dummy || preNode.val != curNode.val) &&
                    (curNode.next == null || curNode.val != curNode.next.val)) {
                realNode.next = curNode;
                realNode = curNode;
            }
            preNode = curNode;
            curNode = curNode.next;
            preNode.next = null;
        }
        return dummy.next;
    }

    /**
     * LeetCode.86 分隔链表
     * 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。
     * <p>
     * 你应当保留两个分区中每个节点的初始相对位置。
     * <p>
     * 示例:
     * 输入: head = 1->4->3->2->5->2, x = 3
     * 输出: 1->2->2->4->3->5
     *
     * @param head
     * @param x
     * @return
     */
    public ListNode partition(ListNode head, int x) {
        /*
            三指针，left表示分界线，左边的小于x，右边的大于等于x。
            curr是当前处理的指针，prev是前一个指针。
            改变链表的步骤：(在纸上画图可以帮助理解)
            1.让prev.next = cur.next
            2.让curr.next = left.next
            3.让left.next = cur

            特殊情况，当prev和left一样时，若curr.val < x，直接移动三个指针。
            若curr.val >= x，移动prev和curr，left不变。
         */
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode left = dummy;
        ListNode curr = head;
        ListNode prev = dummy;
        while (curr != null) {
            // 特殊情况
            if (prev == left) {
                if (curr.val < x) {
                    left = left.next;
                }
                prev = curr;
                curr = curr.next;
            } else {
                if (curr.val >= x) {
                    prev = curr;
                    curr = curr.next;
                } else {
                    prev.next = curr.next;
                    curr.next = left.next;
                    left.next = curr;
                    left = left.next;
                    curr = prev.next;
                }
            }
        }
        return dummy.next;
    }

    /**
     * LeetCode.89 格雷编码
     * <p>
     * 格雷编码是一个二进制数字系统，在该系统中，两个连续的数值仅有一个位数的差异。
     * 给定一个代表编码总位数的非负整数 n，打印其格雷编码序列。格雷编码序列必须以 0 开头。
     * <p>
     * 示例 1:
     * 输入: 2
     * 输出: [0,1,3,2]
     * 解释:
     * 00 - 0
     * 01 - 1
     * 11 - 3
     * 10 - 2
     * <p>
     * 对于给定的 n，其格雷编码序列并不唯一。
     * 例如，[0,2,3,1] 也是一个有效的格雷编码序列。
     * 00 - 0
     * 10 - 2
     * 11 - 3
     * 01 - 1
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        /*
            格雷码公式：G(i) = i ^ i/2
         */
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < 1 << n; i++) {
            res.add(i ^ i >> 1);
        }
        return res;
    }

    /**
     * LeetCode.90 子集II
     * 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
     * <p>
     * 说明：解集不能包含重复的子集。
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);
        helper2(res, new ArrayList<>(), nums, 0);
        return res;
    }

    private void helper2(List<List<Integer>> res, List<Integer> cur, int[] nums, int index) {
        // base case
        if (index == nums.length) {
            res.add(new ArrayList<>(cur));
            return;
        }
        // choice
        cur.add(nums[index]);
        helper2(res, cur, nums, index + 1);
        cur.remove(cur.size() - 1);

        // 跳过重复情况
        while (index + 1 < nums.length && nums[index] == nums[index + 1]) {
            index++;
        }

        // no choice
        helper(res, cur, nums, index + 1);
    }

    /**
     * LeetCode.91 解码方法
     * 一条包含字母 A-Z 的消息通过以下方式进行了编码：
     * 'A' -> 1
     * 'B' -> 2
     * ...
     * 'Z' -> 26
     * <p>
     * 给定一个只包含数字的非空字符串，请计算解码方法的总数。
     * <p>
     * 示例 1:
     * 输入: "12"
     * 输出: 2
     * 解释: 它可以解码为 "AB"（1 2）或者 "L"（12）。
     * <p>
     * 示例 2:
     * 输入: "226"
     * 输出: 3
     * 解释: 它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        /*
            dp[i]:前0~i个字符作为前缀字符串，所有的解码方式有多少种
            init: dp[0] = 1
            状态转移方程：dp[i] = dp[i] + dp[i - 1]，当为两位数，在原有基础上
            dp[i] += dp[i - 2]
            return: dp[s.length() - 1]
         */
        int[] dp = new int[s.length()];
        if (s.charAt(0) == '0') {
            return 0;
        }
        dp[0] = 1;
        for (int i = 1; i < s.length(); i++) {
            // 保证每一位没有0
            if (s.charAt(i) != '0') {
                dp[i] += dp[i - 1];
            }
            if (s.charAt(i - 1) == '1' || (s.charAt(i - 1) == '2' && s.charAt(i) <= '6')) {
                // 两位数，十几/二十几的情况
                if (i - 2 >= 0) {
                    dp[i] += dp[i - 2];
                } else {
                    dp[i]++;
                }
            }
        }
        return dp[s.length() - 1];
    }

    /**
     * LeetCode.92 反转链表II
     * 反转从位置 m 到 n 的链表。请使用一趟扫描完成反转。
     * <p>
     * 说明:
     * 1 ≤ m ≤ n ≤ 链表长度。
     * <p>
     * 示例:
     * 输入: 1->2->3->4->5->NULL, m = 2, n = 4
     * 输出: 1->4->3->2->5->NULL
     *
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        /*
            根据m和n分成三段：m之前的，m~n的，n之后的
         */
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        // 第一段的最后一个节点
        ListNode first = dummy;
        for (int i = 1; i < m; i++) {
            first = first.next;
        }
        // 第二段的最后一个节点。翻转之前是first.next指的是第二段的第一个，翻转以后
        // 就是第二段的最后一个了
        ListNode second = first.next;
        if (second == null) {
            return dummy.next;
        }
        // 第二段的第一个节点，翻转之前是最后一个节点。翻转后就是第一个节点
        ListNode left = second;
        // 第三段的第一个节点
        ListNode right = left.next;

        // 翻转
        for (int i = m; i < n; i++) {
            ListNode next = right.next;
            right.next = left;
            left = right;
            right = next;
        }
        first.next = left;
        second.next = right;
        return dummy.next;
    }

    /**
     * LeetCode.93 复原IP地址
     * 给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。
     * <p>
     * 示例:
     * 输入: "25525511135"
     * 输出: ["255.255.11.135", "255.255.111.35"]
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        //TODO 不能通过100%，有问题
        List<String> res = new ArrayList<>();
        if (s == null || s.length() == 0) {
            return res;
        }
        // 根据.所分成的段
        List<String> segment = new ArrayList<>();
        dfs(s, 0, res, segment);
        return res;
    }

    /**
     * 93题帮助函数
     *
     * @param s
     * @param start
     * @param res
     * @param segment
     */
    private void dfs(String s, int start, List<String> res, List<String> segment) {
        if (start == s.length()) {
            // 合法
            if (segment.size() == 4) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < segment.size(); i++) {
                    sb.append(segment.get(i));
                    if (i != segment.size() - 1) {
                        sb.append('.');
                    }
                }
                res.add(sb.toString());
            }
            return;
        } else {
            // 两个点之间只能有小于等于3位的数
            if (segment.size() >= 4) {
                return;
            }
            for (int i = start; i < s.length() && i < start + 3; i++) {
                String str = s.substring(start, i + 1);
                // 不能以0开头，且长度大于1的话，非法
                if (str.charAt(0) == '0' && str.length() > 1) {
                    break;
                }
                int num = Integer.parseInt(str);
                if (num > 0 && num <= 255) {
                    segment.add(str);
                    dfs(s, i + 1, res, segment);
                    segment.remove(segment.size() - 1);
                } else {
                    break;
                }
            }
        }
    }

    /**
     * LeetCode.94 二叉树的中序遍历
     * 给定一个二叉树，返回它的中序遍历
     *
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root != null) {
            Stack<TreeNode> stack = new Stack<>();
            while (!stack.isEmpty() || root != null) {
                if (root != null) {
                    stack.push(root);
                    root = root.left;
                } else {
                    root = stack.pop();
                    res.add(root.val);
                    root = root.right;
                }
            }
        }
        return res;
    }

    /**
     * LeetCode.95 不同的二叉搜索树II
     * <p>
     * 输入: 3
     * 输出:
     * [
     *   [1,null,3,2],
     *   [3,2,null,1],
     *   [3,1,null,null,2],
     *   [2,1,3],
     *   [1,null,2,null,3]
     * ]
     * 解释:
     * 以上的输出对应以下 5 种不同结构的二叉搜索树：
     * <p>
     * 1          3     3      2      1
     * \        /     /       / \      \
     * 3      2     1        1   3      2
     * /     /       \                   \
     * 2    1         2                   3
     *
     * @param n
     * @return
     */
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> res = new ArrayList<>();
        if (n == 0) {
            return res;
        }
        return helper(1, n);
    }

    /**
     * 95题帮助函数
     *
     * @param left
     * @param right
     */
    private List<TreeNode> helper(int left, int right) {
        List<TreeNode> res = new ArrayList<>();
        if (left > right) {
            res.add(null);
            return res;
        }
        for (int i = left; i <= right; i++) {
            List<TreeNode> leftAll = helper(left, i - 1);
            List<TreeNode> rightAll = helper(i + 1, right);
            for (TreeNode l : leftAll) {
                for (TreeNode r : rightAll) {
                    TreeNode cur = new TreeNode(i);
                    cur.left = l;
                    cur.right = r;
                    res.add(cur);
                }
            }
        }
        return res;
    }

    /**
     * LeetCode.96 不同的二叉搜索树
     * 给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
     * <p>
     * 示例:
     * 输入: 3
     * 输出: 5
     * 解释:
     * 给定 n = 3, 一共有 5 种不同结构的二叉搜索树:
     * <p>
     * 1         3       3      2      1
     * \       /       /      / \      \
     * 3      2       1      1   3      2
     * /     /       \                  \
     * 2     1        2                  3
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        /*
            动态规划。dp[n]: 有n个节点，能组成多少种二叉搜索树
            F(i,n): 以i为根的不同二叉搜索树个数(1 <= i <= n)。
            dp[n] = ∑(i from 1 to n) dp[i - 1] * dp[n - i]
         */
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    /**
     * LeetCode.146 LRU缓存机制
     * <p>
     * 运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制。它应该支持以下操作：
     * 获取数据 get 和 写入数据 put 。
     * <p>
     * 获取数据 get(key) - 如果密钥 (key) 存在于缓存中，则获取密钥的值（总是正数），否则返回 -1。
     * 写入数据 put(key, value) - 如果密钥不存在，则写入其数据值。当缓存容量达到上限时，它应该在
     * 写入新数据之前删除最近最少使用的数据值，从而为新的数据值留出空间。
     * <p>
     * 进阶:
     * <p>
     * 你是否可以在 O(1) 时间复杂度内完成这两种操作？
     * <p>
     * 示例:
     * <p>
     * LRUCache cache = new LRUCache(2) // 缓存容量
     * cache.put(1,1);
     * cache.put(2,2);
     * cache.get(1);       // 返回  1
     * cache.put(3,3);    // 该操作会使得密钥 2 作废
     * cache.get(2);       // 返回 -1 (未找到)
     * cache.put(4,4);    // 该操作会使得密钥 1 作废
     * cache.get(1);       // 返回 -1 (未找到)
     * cache.get(3);       // 返回  3
     * cache.get(4);       // 返回  4
     */
    class LRUCache extends LinkedHashMap<Integer, Integer> {
        private int capacity;

        public LRUCache(int capacity) {
            super(capacity, 0.75F, true);
            this.capacity = capacity;
        }

        public int get(int key) {
            return super.getOrDefault(key, -1);
        }

        public void put(int key, int value) {
            super.put(key, value);
        }

        @Override
        protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
            return size() > capacity;
        }
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

