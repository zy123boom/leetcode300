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
                i = Math.max(i, map.get(s.charAt(j)) + 1);;
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
            奇数时看两个字符的中间字符，偶数时看两个字符的中间
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
        if (nums[start] == target) {
            return start;
        }
        if (nums[end] == target) {
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
        // reach是最远能到的地方
        if (nums.length < 2) {
            return true;
        }
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
        List<String> res = new ArrayList<>();
        if (s == null || s.length() == 0 || s.length() > 12) {
            return res;
        }
        helper(res, s, "", 0);
        return res;
    }

    /**
     * 93题帮助函数
     *
     * @param res
     * @param s
     * @param curr  当前状态下已经找到的部分，例如IP为xx.xxx.xx.xxx，curr可以前面的两个部分即xx.xxx
     * @param field IP分成的段，例如xx.xxx.xx.xxx，分成四个段。
     */
    private void helper(List<String> res, String s, String curr, int field) {
        // base case
        if (field == 4 && s.length() == 0) {
            // 最后形成的IP是 .xx.xx.xx.xxx，所以要把第一个.去掉
            res.add(curr.substring(1));
        } else if (field == 4 ^ s.length() == 0) {
            return;
        } else {
            // IP的某一段只有一位数字，此处curr + "."就是最后生成的IP多一个.的原因
            helper(res, s.substring(1), curr + "." + s.substring(0, 1), field + 1);
            // IP的某一段有两位的情况，要满足每一段的第一个数不能是0，并且后面还有数字
            if (s.charAt(0) != '0' && s.length() > 1) {
                helper(res, s.substring(2), curr + "." + s.substring(0, 2), field + 1);
                // IP的某一段有三位的情况，要满足这一段的数字不超过255.
                if (s.length() > 2 && Integer.valueOf(s.substring(0, 3)) <= 255) {
                    helper(res, s.substring(3), curr + "." + s.substring(0, 3), field + 1);
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
            动态规划。
            dp[n]: 有n个节点，能组成多少种二叉搜索树
            F(i,n): 以i为根的不同二叉搜索树个数(1 <= i <= n)。
            dp[n] = ∑(i from 1 to n) F(i, n)
            F(i,n) = dp[i - 1] * dp[n - i]
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
     * LeetCode.98 验证二叉搜索树
     * <p>
     * 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
     * <p>
     * 假设一个二叉搜索树具有如下特征：
     * 节点的左子树只包含小于当前节点的数。
     * 节点的右子树只包含大于当前节点的数。
     * 所有左子树和右子树自身必须也是二叉搜索树。
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        /*
            给定一个区间[min, max]，要保证root的值在这个区间之间，同时左右子树的值也要在
            区间之间。递归去观察，每次更新区间。左子树的区间是(min, root.val - 1)。右子
            树的区间是(root.val + 1, max)。
         */
        if (root == null) {
            return true;
        }
        return helper(root, null, null);
    }

    private boolean helper(TreeNode root, Integer min, Integer max) {
        if (root == null) {
            return true;
        }
        if ((max != null && root.val >= max) || (min != null && root.val <= min)) {
            return false;
        }
        boolean left = helper(root.left, min, root.val);
        boolean right = helper(root.right, root.val, max);
        return left && right;
    }

    /**
     * LeetCode.102 二叉树的层次遍历
     * 给定一个二叉树，返回其按层次遍历的节点值。 （即逐层地，从左到右访问所有节点）。
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        /*
         *   思路：利用队列（Queue）先进先出特性，将本层出队列，下一层入队列
         *       eg:    3
         *             / \
         *            9  20
         *           /\
         *          15 7
         *
         *  （第一回合）
         *
         * 先将根节点root入队，此时队列只有3，长度为1
         * 将3出队并放入第一层List，判断3的left,right是否为空，不为空就按顺序从左到右入队。
         * 第一层为[3]
         *
         * （第二回合）
         * 此时Queue长度为2，将Queue前两个节点出队
         * 9出队，放入第二层List，然后判断9的左右节点，如同（第一回合）的步骤2，没有便不需要入队
         * 20出队，放入第二层List，然后将它的left15,right7入队。
         * 此时第二层为[9,20]，且当前Queue为[15,17]，为下一回合做准备
         *
         * （第三回合）
         * 此时Queue长度为2，将前2个节点出队
         * 出队时发现两个节点都没有子节点，故本轮结束第三层为[15,7]，Queue为空
         * Queue为空，结束。最后结果
         * [
         * [3],
         * [9,20],
         * [15,7]
         * ]
         */
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        // 先将根节点入队
        queue.offer(root);
        while (!queue.isEmpty()) {
            // 代表res的每一层
            List<Integer> row = new ArrayList<>();
            // 代表上一回合的长度
            int length = queue.size();
            while (length > 0) {
                TreeNode temp = queue.poll();
                if (temp.left != null) {
                    queue.offer(temp.left);
                }
                if (temp.right != null) {
                    queue.offer(temp.right);
                }
                row.add(temp.val);
                length--;
            }
            // 循环后，将list加到结果中
            res.add(row);
        }
        return res;
    }

    /**
     * LeetCode.103 二叉树的锯齿形层次遍历
     * 给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，
     * 以此类推，层与层之间交替进行）。
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        /*
            与层次遍历一样，当奇数行时进行reverse
         */
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int count = 0;
        while (!queue.isEmpty()) {
            List<Integer> row = new LinkedList<>();
            int size = queue.size();
            while (size > 0) {
                TreeNode temp = queue.poll();
                if (temp.left != null) {
                    queue.add(temp.left);
                }
                if (temp.right != null) {
                    queue.add(temp.right);
                }
                row.add(temp.val);
                size--;
            }
            if (count % 2 == 1) {
                reverse(row);
            }
            count++;
            res.add(row);
        }
        return res;
    }

    /**
     * 103题帮助函数
     *
     * @param row
     */
    private void reverse(List<Integer> row) {
        int left = 0, right = row.size() - 1;
        while (left < right) {
            int temp = row.get(left);
            row.set(left, row.get(right));
            row.set(right, temp);
            left++;
            right--;
        }
    }

    /**
     * LeetCode.105 从前序与中序遍历序列构造二叉树
     * <p>
     * 根据一棵树的前序遍历与中序遍历构造二叉树。
     * 注意:
     * 你可以假设树中没有重复的元素。
     * <p>
     * 例如，给出
     * 前序遍历 preorder = [3,9,20,15,7]
     * 中序遍历 inorder = [9,3,15,20,7]
     * 返回如下的二叉树：
     * <p>
     * 3
     * / \
     * 9  20
     * /  \
     * 15   7
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return helper(preorder, inorder, 0, 0, inorder.length - 1);
    }

    /**
     * 105题帮助函数
     *
     * @param preorder
     * @param inorder
     * @param preStart 先序开始的位置（根节点）
     * @param inStart  中序开始的位置
     * @param inEnd    中序结束的位置
     */
    private TreeNode helper(int[] preorder, int[] inorder, int preStart, int inStart, int inEnd) {
        if (inStart > inEnd) {
            return null;
        }
        // 构造当前根节点
        int currentVal = preorder[preStart];
        TreeNode current = new TreeNode(currentVal);

        // 找到根节点在中序遍历中的位置
        int inIndex = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == currentVal) {
                inIndex = i;
            }
        }

        // 递归的构建左子树和右子树.preStart = preStart + 右子树.size
        TreeNode left = helper(preorder, inorder, preStart + 1, inStart, inIndex - 1);
        TreeNode right = helper(preorder, inorder, preStart + (inIndex - inStart + 1), inIndex + 1, inEnd);

        current.left = left;
        current.right = right;
        return current;
    }

    /**
     * LeetCode.106 从中序与后序遍历序列构造二叉树
     * <p>
     * 根据一棵树的中序遍历与后序遍历构造二叉树。
     * 注意:
     * 你可以假设树中没有重复的元素。
     * <p>
     * 例如，给出
     * 中序遍历 inorder = [9,3,15,20,7]
     * 后序遍历 postorder = [9,15,7,20,3]
     * 返回如下的二叉树：
     * <p>
     * 3
     * / \
     * 9  20
     * /  \
     * 15   7
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        return helper2(inorder, postorder, postorder.length - 1, 0, inorder.length - 1);
    }

    /**
     * 106题帮助函数
     *
     * @param inorder
     * @param postorder
     * @param postEnd
     * @param inStart
     * @param inEnd
     * @return
     */
    private TreeNode helper2(int[] inorder, int[] postorder, int postEnd, int inStart, int inEnd) {
        if (inStart > inEnd) {
            return null;
        }

        int currentVal = postorder[postEnd];
        TreeNode current = new TreeNode(currentVal);

        // 找到根节点在中序遍历中的位置
        int inIndex = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == currentVal) {
                inIndex = i;
            }
        }

        TreeNode left = helper2(inorder, postorder, postEnd - (inEnd - inIndex) - 1, inStart, inIndex - 1);
        TreeNode right = helper2(inorder, postorder, postEnd - 1, inIndex + 1, inEnd);

        current.left = left;
        current.right = right;
        return current;
    }

    /**
     * LeetCode.109 有序链表转换二叉搜索树
     * <p>
     * 给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
     * 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
     * <p>
     * 示例:
     * 给定的有序链表： [-10, -3, 0, 5, 9]
     * 一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：
     * <p>
     * 0
     * / \
     * -3   9
     * /   /
     * -10  5
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        /*
            通过快慢指针找出中间的节点，然后根据中间节点划分成左右两个子问题
         */
        if (head == null) {
            return null;
        }
        return helper(head, null);
    }

    /**
     * 109题帮助函数
     *
     * @param head
     * @param tail 总问题的结尾
     * @return
     */
    private TreeNode helper(ListNode head, ListNode tail) {
        if (head == null || head == tail) {
            return null;
        }

        // 找中间节点
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != tail && fast.next.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }

        // 根节点
        TreeNode current = new TreeNode(slow.val);
        current.left = helper(head, slow);
        current.right = helper(slow.next, tail);
        return current;
    }

    /**
     * LeetCode.113 路径总和II
     * <p>
     * 给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
     * 说明: 叶子节点是指没有子节点的节点。
     * <p>
     * 示例:
     * 给定如下二叉树，以及目标和 sum = 22，
     * <p>
     * 5
     * / \
     * 4    8
     * /    / \
     * 11  13  4
     * /  \    / \
     * 7   2  5   1
     * <p>
     * 返回:
     * [
     * [5,4,11,2],
     * [5,8,4,5]
     * ]
     *
     * @param root
     * @param sum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        /*
            类似路径总和I，使用dfs回溯法。首先将根节点加到结果中，然后递归的
            考虑左子树和右子树。考察左子树时让sum -= root.val进行考察
         */
        List<List<Integer>> res = new ArrayList<>();
        helper(root, sum, res, new ArrayList<>());
        return res;
    }

    /**
     * 113题帮助函数
     *
     * @param root
     * @param sum
     * @param res
     * @param curr
     */
    private void helper(TreeNode root, int sum, List<List<Integer>> res, List<Integer> curr) {
        // base case 1
        if (root == null) {
            return;
        }

        curr.add(root.val);

        // base case 2，左右子树为空且根节点值为sum，可以结束
        if (root.left == null && root.right == null && sum == root.val) {
            res.add(new ArrayList<>(curr));
            curr.remove(curr.size() - 1);
            return;
        }

        // 探索左子树和右子树
        helper(root.left, sum - root.val, res, curr);
        helper(root.right, sum - root.val, res, curr);
        curr.remove(curr.size() - 1);
    }

    /**
     * LeetCode.114 二叉树展开为链表
     * <p>
     * 给定一个二叉树，原地将它展开为链表。
     * <p>
     * 例如，给定二叉树
     * 1
     * / \
     * 2   5
     * / \   \
     * 3   4   6
     * <p>
     * 将其展开为：
     * 1
     * \
     * 2
     * \
     * 3
     * \
     * 4
     * \
     * 5
     * \
     * 6
     *
     * @param root
     */
    private TreeNode prev = null; // 下一个的状态

    public void flatten(TreeNode root) {
        /*
            让左子树为空，全部放到右子树。然后先看右子树再看左子树。让右子树的头节点连到
            左子树。
         */
        if (root == null) {
            return;
        }
        // 先右后左
        flatten(root.right);
        flatten(root.left);
        root.right = prev;
        root.left = null;
        prev = root;
    }

    /**
     * LeetCode.116 填充每个节点的下一个右侧节点指针
     * <p>
     * 给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：
     * <p>
     * struct Node {
     * int val;
     * Node *left;
     * Node *right;
     * Node *next;
     * }
     * <p>
     * 填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。
     * 初始状态下，所有 next 指针都被设置为 NULL。
     * <p>
     * 提示：
     * 你只能使用常量级额外空间。
     * 使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        /*
            类似层序遍历
         */
        if (root == null) {
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            Node current = null;
            while (size > 0) {
                Node node = queue.poll();
                // 注意，先右后左
                if (node.right != null) {
                    queue.offer(node.right);
                }
                if (node.left != null) {
                    queue.offer(node.left);
                }
                node.next = current;
                current = node;
                size--;
            }
        }
        return root;
        /*
            解法二：递归
            if (root == null) {
                return null;
            }
            if (root.left != null) {
                root.left.next = root.right;
            }
            if (root.right != null && root.next != null) {
                root.right.next = root.next.left;
            }
            connect(root.left);
            connect(root.right);
            return root;
         */
    }

    /**
     * LeetCode.117 填充每个节点的下一个右侧节点指针 II
     * <p>
     * 与I不同的是，完美二叉树变为二叉树。
     * <p>
     * 提示：
     * 树中的节点数小于 6000
     * -100 <= node.val <= 100
     *
     * @param root
     * @return
     */
    public Node connect2(Node root) {
        // 116题的层次遍历方法依旧可以通过
        return connect(root);
    }

    /**
     * LeetCode.120 三角形最小路径和
     * <p>
     * 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
     * <p>
     * 例如，给定三角形：
     * [
     * [2],
     * [3,4],
     * [6,5,7],
     * [4,1,8,3]
     * ]
     * 自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
     * <p>
     * 说明：
     * 如果你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。
     *
     * @param triangle
     * @return
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        /*
            从下往上算。最后一层 4 1 8 3。6位置上的数的结果是4和1取最小加上6，以此类推。
            为了达到O(n)的空间复杂度，每算完一层覆盖掉。最后返回的是左上角即dp[0]
         */
        if (triangle == null || triangle.size() == 0) {
            return 0;
        }
        int m = triangle.size();
        int[] dp = new int[m + 1];
        for (int i = m - 1; i >= 0; i--) {
            List<Integer> line = triangle.get(i);
            for (int j = 0; j < line.size(); j++) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + line.get(j);
            }
        }
        return dp[0];
    }

    /**
     * LeetCode.127 单词接龙
     * <p>
     * 给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列
     * 的长度。转换需遵循如下规则：
     * 每次转换只能改变一个字母。
     * 转换过程中的中间单词必须是字典中的单词。
     * <p>
     * 说明:
     * 如果不存在这样的转换序列，返回 0。
     * 所有单词具有相同的长度。
     * 所有单词只由小写字母组成。
     * 字典中不存在重复的单词。
     * 你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
     * <p>
     * 示例 1:
     * 输入:
     * beginWord = "hit",
     * endWord = "cog",
     * wordList = ["hot","dot","dog","lot","log","cog"]
     * 输出: 5
     * 解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     * 返回它的长度 5。
     * <p>
     * 示例 2:
     * 输入:
     * beginWord = "hit"
     * endWord = "cog"
     * wordList = ["hot","dot","dog","lot","log"]
     * 输出: 0
     * 解释: endWord "cog" 不在字典中，所以无法进行转换。
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    Map<String, List<String>> map = new HashMap<>();

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        /*
            此题考虑BFS。
            构建一个Map，key是String, value是List<String>。value的含义是对应的key下一步
            能变成哪些String。例如1下一步可以变成2,3,4,5。1就是key，
            2,3,4,5组成的List就是value。
         */
        if (beginWord.equals(endWord)) {
            return 0;
        }
        buildMap(wordList, beginWord);
        // dfs
        // 用来处理重复情况。例如1可以变成2，2又可以变成1，重复情况。
        Set<String> doneSet = new HashSet<>();
        Queue<String> queue = new LinkedList<>();

        queue.offer(beginWord);
        doneSet.add(beginWord);
        int steps = 1;
        while (queue.size() != 0) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String curr = queue.poll();
                // 找到了
                if (curr.equals(endWord)) {
                    return steps;
                }
                List<String> nextStrList = map.get(curr);
                for (String nextStr : nextStrList) {
                    if (!doneSet.contains(nextStr)) {
                        queue.offer(nextStr);
                        doneSet.add(nextStr);
                    }
                }
            }
            steps++;
        }
        return 0;
    }

    /**
     * 127题帮助函数，构造Map
     *
     * @param wordList
     * @param beginWord
     */
    private void buildMap(List<String> wordList, String beginWord) {
        // 生成映射表
        for (String str : wordList) {
            List<String> nList = new LinkedList<>();
            map.put(str, nList);
            for (String next : wordList) {
                if (diff(str, next) == 1) {
                    map.get(str).add(next);
                }
            }
        }
        // 如果beginWord不存在，则也需要加到map里。
        if (!map.containsKey(beginWord)) {
            List<String> nList = new LinkedList<>();
            map.put(beginWord, nList);
            for (String next : wordList) {
                if (diff(beginWord, next) == 1) {
                    map.get(beginWord).add(next);
                }
            }
        }
    }

    /**
     * 127题帮助函数，构建map所用，求距离。
     * 意思是两个字符串有几个字母不同
     *
     * @param s
     * @param t
     * @return
     */
    private int diff(String s, String t) {
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != t.charAt(i)) {
                count++;
            }
        }
        return count;
    }

    /**
     * LeetCode.129 求根到叶子节点数字之和
     * <p>
     * 给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。
     * 例如，从根到叶子节点路径 1->2->3 代表数字 123。
     * 计算从根到叶子节点生成的所有数字之和。
     * <p>
     * 说明: 叶子节点是指没有子节点的节点。
     * <p>
     * 示例 1:
     * 输入: [1,2,3]
     * 1
     * / \
     * 2   3
     * 输出: 25
     * 解释:
     * 从根到叶子节点路径 1->2 代表数字 12.
     * 从根到叶子节点路径 1->3 代表数字 13.
     * 因此，数字总和 = 12 + 13 = 25.
     * <p>
     * 示例 2:
     * 输入: [4,9,0,5,1]
     * 4
     * / \
     * 9   0
     *  / \
     * 5   1
     * 输出: 1026
     * 解释:
     * 从根到叶子节点路径 4->9->5 代表数字 495.
     * 从根到叶子节点路径 4->9->1 代表数字 491.
     * 从根到叶子节点路径 4->0 代表数字 40.
     * 因此，数字总和 = 495 + 491 + 40 = 1026.
     *
     * @param root
     * @return
     */
    private int sum = 0;

    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        helper(root, root.val);
        return sum;
    }

    /**
     * 129题帮助函数
     *
     * @param root
     * @param curSum
     */
    private void helper(TreeNode root, int curSum) {
        if (root.left == null && root.right == null) {
            sum += curSum;
            return;
        }

        if (root.left != null) {
            helper(root.left, curSum * 10 + root.left.val);
        }

        if (root.right != null) {
            helper(root.right, curSum * 10 + root.right.val);
        }
    }

    /**
     * LeetCode.130 被围绕的区域
     * <p>
     * 给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。
     * 找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。
     * <p>
     * 示例:
     * X X X X
     * X O O X
     * X X O X
     * X O X X
     * 运行你的函数后，矩阵变为：
     * X X X X
     * X X X X
     * X X X X
     * X O X X
     * <p>
     * 解释:
     * 被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，
     * 或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称
     * 它们是“相连”的。
     *
     * @param board
     */
    private int[][] direction = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public void solve(char[][] board) {
        // 先填充外侧，剩下的就是内侧了
        if (board == null || board.length == 0) {
            return;
        }
        int m = board.length;
        int n = board[0].length;

        for (int i = 0; i < m; i++) {
            dfs(board, i, 0);
            dfs(board, i, n - 1);
        }

        for (int i = 0; i < n; i++) {
            dfs(board, 0, i);
            dfs(board, m - 1, i);
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'T') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    /**
     * 130题帮助函数
     *
     * @param board
     * @param row
     * @param column
     */
    private void dfs(char[][] board, int row, int column) {
        if (row < 0 || row >= board.length || column < 0 || column >= board[0].length
                || board[row][column] != 'O') {
            return;
        }
        board[row][column] = 'T';
        for (int[] d : direction) {
            dfs(board, row + d[0], column + d[1]);
        }
    }

    /**
     * LeetCode.131 分割回文串
     * <p>
     * 给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。
     * 返回 s 所有可能的分割方案。
     * <p>
     * 示例:
     * 输入: "aab"
     * 输出:
     * [
     * ["aa","b"],
     * ["a","a","b"]
     * ]
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        List<List<String>> res = new ArrayList<>();
        dfs(s, res, new ArrayList<>());
        return res;
    }

    /**
     * 131帮助函数
     *
     * @param s
     * @param res
     * @param curr
     */
    private void dfs(String s, List<List<String>> res, List<String> curr) {
        // String开始的index通过dfs递归中的参数变化实现
        // base case - 由于不停的被分割子串，当字符串长度为0时无法分割子串，return
        if (s.length() == 0) {
            res.add(new ArrayList<>(curr));
            return;
        }

        // i代表子串的长度
        for (int i = 1; i <= s.length(); i++) {
            if (isPalindrome(s, 0, i)) {
                curr.add(s.substring(0, i));
                dfs(s.substring(i), res, curr);
                curr.remove(curr.size() - 1);
            }
        }
    }

    /**
     * 131题帮助函数
     * 判断字符串是否回文
     *
     * @param s
     * @param start
     * @param length
     * @return
     */
    private boolean isPalindrome(String s, int start, int length) {
        int end = start + length - 1;
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }

    /**
     * LeetCode.134 加油站
     * <p>
     * 在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
     * 你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从
     * 其中的一个加油站出发，开始时油箱为空。
     * 如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
     * <p>
     * 说明: 
     * 如果题目有解，该答案即为唯一答案。
     * 输入数组均为非空数组，且长度相同。
     * 输入数组中的元素均为非负数。
     * <p>
     * 示例 1:
     * 输入:
     * gas  = [1,2,3,4,5]
     * cost = [3,4,5,1,2]
     * 输出: 3
     * 解释:
     * 从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
     * 开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
     * 开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
     * 开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
     * 开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
     * 开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
     * 因此，3 可为起始索引。
     * <p>
     * 示例 2:
     * 输入:
     * gas  = [2,3,4]
     * cost = [3,4,3]
     * 输出: -1
     * 解释:
     * 你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
     * 我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油
     * 开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油
     * 开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油
     * 你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
     * 因此，无论怎样，你都不可能绕环路行驶一周。
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        /*
            两个结论：
            1.如果 sum(gas) < sum(cost) ，那么不可能环行一圈，这种情况下答案是 -1
            2.对于加油站 i ，如果 gas[i] - cost[i] < 0 ，则不可能从这个加油站出发，
            因为在前往 i + 1 的过程中，汽油就不够了。

            算法：
            1.初始化总油量totalTank和当前油量currTank为0，选择0位置为起点。
            2.遍历加油站，每次都加上gas[i] - cost[i]。当currTank小于0，表示到不了下一个加油站，
            以下一个加油站作为起始点，然后currTank归0。以此类推
            3.最后的结果就是start，如果totalTank小于0结果就是-1。
         */
        int n = gas.length;
        int totalTank = 0;
        int currTank = 0;
        int start = 0;
        for (int i = 0; i < n; i++) {
            totalTank += gas[i] - cost[i];
            currTank += gas[i] - cost[i];
            if (currTank < 0) {
                currTank = 0;
                start = i + 1;
            }
        }
        return totalTank >= 0 ? start : -1;
    }

    /**
     * LeetCode.137 只出现一次的数字II
     * <p>
     * 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
     * <p>
     * 说明：
     * 你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
     * <p>
     * 示例 1:
     * 输入: [2,2,3,2]
     * 输出: 3
     * <p>
     * 示例 2:
     * 输入: [0,1,0,1,0,1,99]
     * 输出: 99
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        // 出现一次的数字
        int once = 0;
        // 出现三次的数字
        int third = 0;
        for (int num : nums) {
            once = ~third & (once ^ num);
            third = ~once & (third ^ num);
        }
        return once;
    }

    /**
     * LeetCode.139 单词拆分
     * <p>
     * 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为
     * 一个或多个在字典中出现的单词。
     * <p>
     * 说明：
     * 拆分时可以重复使用字典中的单词。
     * 你可以假设字典中没有重复的单词。
     * <p>
     * 示例 1：
     * 输入: s = "leetcode", wordDict = ["leet", "code"]
     * 输出: true
     * 解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
     * <p>
     * 示例 2：
     * 输入: s = "applepenapple", wordDict = ["apple", "pen"]
     * 输出: true
     * 解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     *      注意你可以重复使用字典中的单词。
     * <p>
     * 示例 3：
     * 输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
     * 输出: false
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        /*
            dp[i]：前i个字符组成的串是否可以被拆分
            将字符串分成两段，0~j和j到i。0~j满足就是dp[j]，j到i就是取子串，看是否在词库里。
         */
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        Set<String> set = new HashSet<>(wordDict);
        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }

    /**
     * LeetCode.142 环形链表II
     * <p>
     * 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
     * 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
     * 如果 pos 是 -1，则在该链表中没有环。
     * <p>
     * 说明：不允许修改给定的链表。
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        /*
            快慢指针判环，如果有环，则让fast指针回到开头，变为每次走一步，
            然后两个指针一起走，相遇处就是入环节点。
         */
        if (head == null || head.next == null || head.next.next == null) {
            return null;
        }
        ListNode fast = head.next.next;
        ListNode slow = head.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    /**
     * LeetCode.143 重排链表
     * <p>
     * 给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
     * 将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…
     * 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
     * <p>
     * 示例 1:
     * 给定链表 1->2->3->4, 重新排列为 1->4->2->3.
     * <p>
     * 示例 2:
     * 给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        /*
            由题意知，最后要的结果是每次首尾节点相连组成的链表。结束的条件是直到
            到达了原来链表的中点。
            eg: 1->2->3->4->5->6
            1.找到链表中点
            2.根据中点将链表分成两部分，翻转后半部分，断掉前后部分的连接。
             1->2->3
             6->5->4
            3.依次连接。1->6->2->5->3->4
         */
        if (head == null) {
            return;
        }
        // 1.
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        // 2.
        ListNode newHead = slow.next;
        slow.next = null;
        newHead = reverseList(newHead);
        // 3.
        while (newHead != null) {
            ListNode temp = newHead.next;
            newHead.next = head.next;
            head.next = newHead;
            head = newHead.next;
            newHead = temp;
        }
    }

    /**
     * 143题帮助函数，反转链表
     *
     * @param head
     * @return
     */
    private ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }

    /**
     * LeetCode.144 二叉树的前序遍历
     *
     * @param root
     * @return
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        /*
            设一个栈，让根节点进栈。然后出栈顶元素打印该元素，然后看该节点
            有没有左右孩子，先看右，如果有右，压入栈。左边同理。
         */
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            root = stack.pop();
            res.add(root.val);
            if (root.right != null) {
                stack.push(root.right);
            }
            if (root.left != null) {
                stack.push(root.left);
            }
        }
        return res;
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
    class LRUCache {
        /*
            做法一：让类继承LinkedHashMap，get调用super.getOrDefault，put调用super.put。
            最后加上重写后的removeEldestEntry方法，返回语句为size() > capacity。
            此处使用做法二：双向链表+HashMap
         */
        class LinkNode {
            int key;
            int val;
            LinkNode prev;
            LinkNode next;

            public LinkNode(int key, int val) {
                this.key = key;
                this.val = val;
            }
        }

        private int capacity;
        Map<Integer, LinkNode> map = new HashMap<>();
        // 双向链表的头和尾
        LinkNode head = new LinkNode(0, 0);
        LinkNode tail = new LinkNode(0, 0);

        public LRUCache(int capacity) {
            this.capacity = capacity;
            head.next = tail;
            tail.prev = head;
        }

        public int get(int key) {
            if (map.containsKey(key)) {
                LinkNode node = map.get(key);
                // 该节点已被用到，所以放到第一个
                moveNodeToFirst(node);
                return node.val;
            } else {
                return -1;
            }
        }

        public void put(int key, int value) {
            // 如果当前key不存在于map中
            if (!map.containsKey(key)) {
                if (map.size() == capacity) {
                    // 删除最后一个节点
                    deleteLastNode();
                }
                // 头节点后的第一个节点
                LinkNode temp = head.next;
                // 新的节点，这个节点要放在第一个的位置
                LinkNode newNode = new LinkNode(key, value);
                head.next = newNode;
                newNode.prev = head;
                newNode.next = temp;
                temp.prev = newNode;
                map.put(key, newNode);
            } else {
                // key存在的话，只是更新值，并且把节点放到第一个。
                LinkNode node = map.get(key);
                node.val = value;
                moveNodeToFirst(node);
            }
        }

        private void deleteLastNode() {
            LinkNode lastNode = tail.prev;
            lastNode.prev.next = tail;
            tail.prev = lastNode.prev;
            map.remove(lastNode.key);
        }

        private void moveNodeToFirst(LinkNode node) {
            // 首先断开node的前后，然后让node前面的和后面的连上
            node.prev.next = node.next;
            node.next.prev = node.prev;
            // 把node放到第一个
            LinkNode temp = head.next;
            head.next = node;
            node.prev = head;
            node.next = temp;
            temp.prev = node;
        }
    }

    /**
     * LeetCode.147 对链表进行插入排序
     *
     * @param head
     * @return
     */
    public ListNode insertionSortList(ListNode head) {
        ListNode nextNode = null;
        ListNode currNode = head;
        if (currNode == null) {
            return head;
        }
        while (currNode.next != null) {
            nextNode = currNode.next;
            while (nextNode != null) {
                if (currNode.val > nextNode.val) {
                    int temp = currNode.val;
                    currNode.val = nextNode.val;
                    nextNode.val = temp;
                }
                nextNode = nextNode.next;
            }
            currNode = currNode.next;
        }
        return head;
    }

    /**
     * LeetCode.148 排序链表
     * <p>
     * 在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。
     * <p>
     * 示例 1:
     * 输入: 4->2->1->3
     * 输出: 1->2->3->4
     * <p>
     * 示例 2:
     * 输入: -1->5->3->4->0
     * 输出: -1->0->3->4->5
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        /*
            根据时空复杂度的要求，需要使用归并排序。
            第一步，找到链表中点。
            第二步，归并排序
         */
        return mergeSort(head);
    }

    /**
     * 148题帮助函数，归并排序
     *
     * @param head
     * @return
     */
    private ListNode mergeSort(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        // 1.找到链表中点
        ListNode fast = dummy;
        ListNode slow = dummy;
        while (fast != null && fast.next  != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        // 2.归并排序
        ListNode head2 = slow.next;
        slow.next = null;
        head = mergeSort(head);
        head2 = mergeSort(head2);
        return merge(head, head2);
    }

    /**
     * 148题帮助函数，归并
     *
     * @param head1
     * @param head2
     * @return
     */
    private ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        while (head1 != null && head2 != null) {
            if (head1.val < head2.val) {
                curr.next = head1;
                curr = curr.next;
                head1 = head1.next;
            } else {
                curr.next = head2;
                curr = curr.next;
                head2 = head2.next;
            }
        }
        if (head1 != null) {
            curr.next = head1;
        }
        if (head2 != null) {
            curr.next = head2;
        }
        return dummy.next;
    }

    /**
     * LeetCode.150 逆波兰表达式求值
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> num = new Stack<>();
        for (String token : tokens) {
            switch (token) {
                case "+":
                    num.push(num.pop() + num.pop());
                    break;
                case "-":
                    int sub = num.pop();
                    num.push(num.pop() - sub);
                    break;
                case "*":
                    num.push(num.pop() * num.pop());
                    break;
                case "/":
                    int divisor = num.pop();
                    num.push(num.pop() / divisor);
                    break;
                default:
                    num.push(Integer.parseInt(token));
            }
        }
        return num.pop();
    }

    /**
     * LeetCode.151 翻转字符串里的单词
     * <p>
     * 给定一个字符串，逐个翻转字符串中的每个单词。
     * <p>
     * 示例 1：
     * 输入: "the sky is blue"
     * 输出: "blue is sky the"
     * <p>
     * 示例 2：
     * 输入: "  hello world!  "
     * 输出: "world! hello"
     * 解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
     * <p>
     * 示例 3：
     * 输入: "a good   example"
     * 输出: "example good a"
     * 解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
     *
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        s = s.trim();
        List<String> list = Arrays.asList(s.split("\\s+"));
        Collections.reverse(list);
        return String.join(" ", list);
    }

    /**
     * LeetCode.152 乘积最大子数组
     * <p>
     * 给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字）。
     * <p>
     * 示例 1:
     * 输入: [2,3,-2,4]
     * 输出: 6
     * 解释: 子数组 [2,3] 有最大乘积 6。
     * <p>
     * 示例 2:
     * 输入: [-2,0,-1]
     * 输出: 0
     * 解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        /*
            类似53题最大子序列的和。由于是乘法，有可能出现两个负数相乘结果成了最大。
            所以此处需要三个值，maxToCurr, minToCurr, max。
            maxToCurr = max of {maxToCurr * num, minToCurr * num, num}
            minToCurr = min of {maxToCurr * num, minToCurr * num, num}
            max = max{maxToCurr, max}
         */
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int maxToCurr = nums[0];
        int minToCurr = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int nextMax = maxToCurr * nums[i];
            int nextMin = minToCurr * nums[i];
            maxToCurr = Math.max(Math.max(nextMax, nextMin), nums[i]);
            minToCurr = Math.min(Math.min(nextMax, nextMin), nums[i]);
            max = Math.max(maxToCurr, max);
        }
        return max;
    }

    /**
     * LeetCode.162 寻找峰值
     * <p>
     * 峰值元素是指其值大于左右相邻值的元素。
     * 给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。
     * 数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。
     * 你可以假设 nums[-1] = nums[n] = -∞。
     * <p>
     * 示例 1:
     * 输入: nums = [1,2,3,1]
     * 输出: 2
     * 解释: 3 是峰值元素，你的函数应该返回其索引 2。
     * <p>
     * 示例 2:
     * 输入: nums = [1,2,1,3,5,6,4]
     * 输出: 1 或 5
     * 解释: 你的函数可以返回索引 1，其峰值元素为 2；
     *      或者返回索引 5， 其峰值元素为 6。
     * <p>
     * 说明:
     * 你的解法应该是 O(logN) 时间复杂度的。
     *
     * @param nums
     * @return
     */
    public int findPeakElement(int[] nums) {
        /*
            根据题目要求，使用二分。
            如果nums[mid] < nums[mid + 1]，此时还是上升的趋势，答案肯定在后一半。
            反之，已经是下降的趋势，答案已经出现过了，所以在前半段。
         */
        int start = 0, end = nums.length - 1;
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] < nums[mid + 1]) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        return start;
    }

    /**
     * LeetCode.165 比较版本号
     * <p>
     * 比较两个版本号 version1 和 version2。
     * 如果 version1 > version2 返回 1，如果 version1 < version2 返回 -1， 除此之外返回 0。
     * 你可以假设版本字符串非空，并且只包含数字和 . 字符。
     * <p>
     *  . 字符不代表小数点，而是用于分隔数字序列。
     * 例如，2.5 不是“两个半”，也不是“差一半到三”，而是第二版中的第五个小版本。
     * <p>
     * 你可以假设版本号的每一级的默认修订版号为 0。例如，版本号 3.4 的第一级（大版本）和第二级（小版本）
     * 修订号分别为 3 和 4。其第三级和第四级修订号均为 0。
     *  
     * 示例 1:
     * 输入: version1 = "0.1", version2 = "1.1"
     * 输出: -1
     * <p>
     * 示例 2:
     * 输入: version1 = "1.0.1", version2 = "1"
     * 输出: 1
     * <p>
     * 示例 3:
     * 输入: version1 = "7.5.2.4", version2 = "7.5.3"
     * 输出: -1
     * <p>
     * 示例 4：
     * 输入：version1 = "1.01", version2 = "1.001"
     * 输出：0
     * 解释：忽略前导零，“01” 和 “001” 表示相同的数字 “1”。
     * <p>
     * 示例 5：
     * 输入：version1 = "1.0", version2 = "1.0.0"
     * 输出：0
     * 解释：version1 没有第三级修订号，这意味着它的第三级修订号默认为 “0”。
     *
     * @param version1
     * @param version2
     * @return
     */
    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        for (int i = 0; i < Math.max(v1.length, v2.length); i++) {
            int num1 = i < v1.length ? Integer.parseInt(v1[i]) : 0;
            int num2 = i < v2.length ? Integer.parseInt(v2[i]) : 0;
            if (num1 < num2) {
                return -1;
            } else if (num1 > num2) {
                return 1;
            }
        }
        return 0;
    }

    /**
     * leetcode.166 分数到小数
     * <p>
     * 给定两个整数，分别表示分数的分子 numerator 和分母 denominator，以 字符串形式返回小数 。
     * 如果小数部分为循环小数，则将循环的部分括在括号内。
     * 如果存在多个答案，只需返回 任意一个 。
     * 对于所有给定的输入，保证 答案字符串的长度小于 104 。
     * <p>
     * 示例 1：
     * 输入：numerator = 1, denominator = 2
     * 输出："0.5"
     * <p>
     * 示例 2：
     * 输入：numerator = 2, denominator = 1
     * 输出："2"
     * <p>
     * 示例 3：
     * 输入：numerator = 4, denominator = 333
     * 输出："0.(012)"
     *
     * @param numerator
     * @param denominator
     * @return
     */
    public String fractionToDecimal(int numerator, int denominator) {
        /*
            小数点后的循环体寻找标志，若一个数第二次出现，则前面的出现过的小数就是循环体
         */
        if (denominator == 0 || numerator == 0) {
            return "0";
        }

        StringBuilder res = new StringBuilder();
        // 获取符号位的技巧
        String sign = (numerator > 0) ^ (denominator > 0) ? "-" : "";
        res.append(sign);

        long num = Math.abs((long) numerator);
        long den = Math.abs((long) denominator);

        res.append(num / den);
        num %= den;

        // num %= den 为0说明能够整除，直接返回结果
        if (num == 0) {
            return res.toString();
        }

        // 开始小数部分，设置小数点
        res.append(".");

        // key: 每次计算得到的商，value: 对应所在位置
        Map<Long, Integer> map = new HashMap<>();
        map.put(num, res.length());

        // 没有除尽，就循环处理
        while (num != 0) {
            num *= 10;
            res.append(num / den);
            num %= den;

            // 若一个数第二次出现，则前面的出现过的小数就是循环体
            if (map.containsKey(num)) {
                int index = map.get(num);
                res.insert(index, "(");
                res.append(")");
                break;
            } else {
                map.put(num, res.length());
            }
        }

        return res.toString();
    }


    /**
     * 173. 二叉搜索树迭代器
     * <p>
     * 实现一个二叉搜索树迭代器类BSTIterator ，表示一个按中序遍历二叉搜索树（BST）的迭代器：
     * BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 的根节点 root 会作为构造函数的一部分给出。
     * 指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。
     * boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则返回 false 。
     * int next()将指针向右移动，然后返回指针处的数字。
     * 注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返回 BST 中的最小元素。
     * <p>
     * 你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序遍历中至少存在一个下一个数字。
     * <p>
     * 输入
     * ["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
     * [[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
     * 输出
     * [null, 3, 7, true, 9, true, 15, true, 20, false]
     * 解释
     * BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
     * bSTIterator.next();    // 返回 3
     * bSTIterator.next();    // 返回 7
     * bSTIterator.hasNext(); // 返回 True
     * bSTIterator.next();    // 返回 9
     * bSTIterator.hasNext(); // 返回 True
     * bSTIterator.next();    // 返回 15
     * bSTIterator.hasNext(); // 返回 True
     * bSTIterator.next();    // 返回 20
     * bSTIterator.hasNext(); // 返回 False
     */
    static class BSTIterator {
        private int index;
        private List<Integer> arr;


        public BSTIterator(TreeNode root) {
            index = 0;
            arr = new ArrayList<>();
            inOrderTraversal(root, arr);
        }

        private void inOrderTraversal(TreeNode root, List<Integer> arr) {
            if (root == null) {
                return;
            }

            inOrderTraversal(root.left, arr);
            arr.add(root.val);
            inOrderTraversal(root.right, arr);
        }

        public int next() {
            return arr.get(index++);
        }

        public boolean hasNext() {
            return index < arr.size();
        }
    }

    /**
     * LeetCode.179 最大数
     * <p>
     * 给定一组非负整数，重新排列它们的顺序使之组成一个最大的整数。
     * <p>
     * 示例 1:
     * 输入: [10,2]
     * 输出: 210
     * <p>
     * 示例 2:
     * 输入: [3,30,34,5,9]
     * 输出: 9534330
     * 说明: 输出结果可能非常大，所以你需要返回一个字符串而不是整数。
     *
     * @param nums
     * @return
     */
    public String largestNumber(int[] nums) {
        Integer[] n = new Integer[nums.length];
        for (int i = 0; i < nums.length; i++) {
            n[i] = nums[i];
        }

        Arrays.sort(n, (o1, o2) -> {
            String str1 = o1 + "" + o2;
            String str2 = o2 + "" + o1;
            return str2.compareTo(str1);
        });

        StringBuilder sb = new StringBuilder();
        for (Integer num : n) {
            sb.append(num);
        }

        String res = sb.toString();
        return res.charAt(0) == '0' ? "0" : res;
    }

    /**
     * LeetCode.187 重复的DNA序列
     * <p>
     * 所有 DNA 都由一系列缩写为 A，C，G 和 T 的核苷酸组成，例如：“ACGAATTCCG”。在研究 DNA 时，识
     * 别 DNA 中的重复序列有时会对研究非常有帮助。
     * 编写一个函数来查找 DNA 分子中所有出现超过一次的 10 个字母长的序列（子串）。
     * <p>
     * 示例：
     * 输入：s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
     * 输出：["AAAAACCCCC", "CCCCCAAAAA"]
     *
     * @param s
     * @return
     */
    public List<String> findRepeatedDnaSequences(String s) {
        /*
            重复串可以使用HashSet跟substring()方法
            每次遍历将子串放到seen中，如果放入失败，表示seen集合中出现了这个子串，是重复的，
            则加入到repeat中。
         */
        Set<String> seen = new HashSet<>();
        Set<String> repeat = new HashSet<>();

        for (int i = 0; i < s.length() - 9; i++) {
            String temp = s.substring(i, i + 10);
            if (!seen.add(temp)) {
                repeat.add(temp);
            }
        }

        return new ArrayList<>(repeat);
    }

    /**
     * LeetCode.199 二叉树的右视图
     * <p>
     * 给定一棵二叉树，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
     * <p>
     * 示例:
     * 输入: [1,2,3,null,5,null,4]
     * 输出: [1, 3, 4]
     * 解释:
     * 1            <---
     * /   \
     * 2     3         <---
     * \     \
     * 5     4       <---
     *
     * @param root
     * @return
     */
    public List<Integer> rightSideView(TreeNode root) {
        // 层次遍历，结果集只存最后的值。
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            Queue<TreeNode> nextLevel = new LinkedList<>();
            // 最后一个的值
            int lastVal = 0;
            while (!queue.isEmpty()) {
                TreeNode cur = queue.poll();
                lastVal = cur.val;
                if (cur.left != null) {
                    nextLevel.offer(cur.left);
                }
                if (cur.right != null) {
                    nextLevel.offer(cur.right);
                }
            }
            res.add(lastVal);
            queue.addAll(nextLevel);
        }
        return res;
    }

    /**
     * LeetCode.200 岛屿数量
     * <p>
     * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
     * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
     * 此外，你可以假设该网格的四条边均被水包围。
     * <p>
     * 示例 1:
     * 输入:
     * 11110
     * 11010
     * 11000
     * 00000
     * 输出: 1
     * <p>
     * 示例 2:
     * 输入:
     * 11000
     * 11000
     * 00100
     * 00011
     * 输出: 3
     * 解释: 每座岛屿只能由水平和/或竖直方向上相邻的陆地连接而成。
     *
     * @param grid
     * @return
     */
    int[][] directionForNumIslands = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public int numIslands(char[][] grid) {
        /*
            DFS+递归，模板题目
         */
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] != '0') {
                    numIslandsDFS(grid, i, j);
                    res++;
                }
            }
        }
        return res;
    }

    private void numIslandsDFS(char[][] grid, int row, int col) {
        if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length || grid[row][col] == '0') {
            return;
        }
        grid[row][col] = '0';
        for (int[] d : directionForNumIslands) {
            numIslandsDFS(grid, row + d[0], col + d[1]);
        }
    }

    /**
     * LeetCode.207 课程表
     * <p>
     * 你这个学期必须选修 numCourse 门课程，记为 0 到 numCourse-1 。
     * 在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，
     * 我们用一个匹配来表示他们：[0,1]
     * 给定课程总量以及它们的先决条件，请你判断是否可能完成所有课程的学习？
     * <p>
     * 示例 1:
     * 输入: 2, [[1,0]]
     * 输出: true
     * 解释: 总共有 2 门课程。学习课程 1 之前，你需要完成课程 0。所以这是可能的。
     * <p>
     * 示例 2:
     * 输入: 2, [[1,0],[0,1]]
     * 输出: false
     * 解释: 总共有 2 门课程。学习课程 1 之前，你需要先完成​课程 0；并且学习课程 0 之前，
     * 你还应先完成课程 1。这是不可能的。
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        /*
            本题考查图论，采用BFS。首先记录每个入度为0的先修课放入队列中，然后取出，让后修课
            的入度-1，当入度为0时放入队列中。不断重复上述步骤。当队列为空时，检查是否所有的课
            入度都是0，如果不是则返回false。
            具体解题步骤：
            1.构建HashMap，key为课的序号，Value为List，即该课的后续课。
            2.然后构建一个数组inDegree[n]用来存每个课的入度。以课的序号的下标存储。
            3.BFS。将入度为0的课（先导课）放入队列中，然后取队头，如果当前课有后续课，则将后续课入度-1，
            在inDegree中减1。
            4.检查入度数组，看每个值是不是都是0，如果都是0就返回true，否则返回false.
         */
        int[] inDegree = new int[numCourses];
        if (prerequisites == null || prerequisites.length == 0) {
            return true;
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int i = 0; i < prerequisites.length; i++) {
            inDegree[prerequisites[i][0]]++;
            if (graph.containsKey(prerequisites[i][1])) {
                graph.get(prerequisites[i][1]).add(prerequisites[i][0]);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(prerequisites[i][0]);
                graph.put(prerequisites[i][1], list);
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        // 3.
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int course = queue.poll();
            // 当前课的后续课
            List<Integer> subCourses = graph.get(course);
            for (int i = 0; subCourses != null && i < subCourses.size(); i++) {
                if (--inDegree[subCourses.get(i)] == 0) {
                    queue.offer(subCourses.get(i));
                }
            }
        }
        // 4.
        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * LeetCode.208 实现Trie(前缀树)
     * <p>
     * 实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。
     * <p>
     * 示例:
     * Trie trie = new Trie();
     * <p>
     * trie.insert("apple");
     * trie.search("apple");   // 返回 true
     * trie.search("app");     // 返回 false
     * trie.startsWith("app"); // 返回 true
     * trie.insert("app");
     * trie.search("app");     // 返回 true
     * <p>
     * 说明:
     * 你可以假设所有的输入都是由小写字母 a-z 构成的。
     * 保证所有输入均为非空字符串。
     */
    class Trie {
        /*
            children[0]存'a'，children[1]存'b'，以此类推。
            所以存的时候用当前字符减去'a'就是相应的children下标。
         */

        class TrieNode {
            TrieNode[] children;
            boolean isEnd;

            public TrieNode() {
                children = new TrieNode[26];
                isEnd = false;
                // 节点初始化为null
                for (int i = 0; i < 26; i++) {
                    children[i] = null;
                }
            }
        }

        TrieNode root;

        /**
         * Initialize your data structure here.
         */
        public Trie() {
            root = new TrieNode();
        }

        /**
         * Inserts a word into the trie.
         */
        public void insert(String word) {
            char[] strArr = word.toCharArray();
            TrieNode cur = root;
            for (int i = 0; i < strArr.length; i++) {
                // 当前孩子是否为null
                if (cur.children[strArr[i] - 'a'] == null) {
                    cur.children[strArr[i] - 'a'] = new TrieNode();
                }
                cur = cur.children[strArr[i] - 'a'];
            }
            cur.isEnd = true;
        }

        /**
         * Returns if the word is in the trie.
         */
        public boolean search(String word) {
            char[] strArr = word.toCharArray();
            TrieNode cur = root;
            for (int i = 0; i < strArr.length; i++) {
                // 当前孩子是否为null
                if (cur.children[strArr[i] - 'a'] == null) {
                    return false;
                }
                cur = cur.children[strArr[i] - 'a'];
            }
            return cur.isEnd;
        }

        /**
         * Returns if there is any word in the trie that starts with the given prefix.
         */
        public boolean startsWith(String prefix) {
            char[] strArr = prefix.toCharArray();
            TrieNode cur = root;
            for (int i = 0; i < strArr.length; i++) {
                if (cur.children[strArr[i] - 'a'] == null) {
                    return false;
                }
                cur = cur.children[strArr[i] - 'a'];
            }
            return true;
        }
    }

    /**
     * LeetCode.209 长度最小的子数组
     * <p>
     * 给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组。
     * 如果不存在符合条件的连续子数组，返回 0。
     * <p>
     * 示例: 
     * 输入: s = 7, nums = [2,3,1,2,4,3]
     * 输出: 2
     * 解释: 子数组 [4,3] 是该条件下的长度最小的连续子数组。。
     *
     * @param s
     * @param nums
     * @return
     */
    public int minSubArrayLen(int s, int[] nums) {
        /*
            双指针。
            1.用双指针 left 和 right 表示一个窗口。
            right 向右移增大窗口，直到窗口内的数字和大于等于了s。进行第2步。
            2.记录此时的长度，left 向右移动，开始减少长度，每减少一次，就更新最小长度。
            直到当前窗口内的数字和小于了s，回到第1步。
         */
        int n = nums.length;
        if (n == 0) {
            return 0;
        }
        int sum = 0;
        int left = 0, right = 0;
        // 窗口长度
        int minLen = Integer.MAX_VALUE;
        while (right < n) {
            sum += nums[right];
            right++;
            while (sum >= s) {
                // 走左指针
                minLen = Math.min(minLen, right - left);
                sum -= nums[left];
                left++;
            }
        }

        return minLen == Integer.MAX_VALUE ? 0 : minLen;
    }

    /**
     * LeetCode.210 课程表II
     * <p>
     * 现在你总共有 n 门课需要选，记为 0 到 n-1。
     * 在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个
     * 匹配来表示他们: [0,1]
     * 给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。
     * 可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。
     * <p>
     * 示例 1:
     * 输入: 2, [[1,0]]
     * 输出: [0,1]
     * 解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
     * <p>
     * 示例 2:
     * 输入: 4, [[1,0],[2,0],[3,1],[3,2]]
     * 输出: [0,1,2,3] or [0,2,1,3]
     * 解释: 总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排
     * 在课程 0 之后。因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
     *
     * @param numCourses
     * @param prerequisites
     * @return
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        /*
            思路同207题，不同的是对第三步BFS的优化，使用数组进行BFS，最后的结果也就是这个数组。
            使用双指针，首先让入度为0的课进数组，然后后移快指针。然后看慢指针的值，减后续课的入度，当后续课
            的入度为0时，后移快指针。每次看完慢指针的值以后慢指针后移。当快慢指针都==n时，则表示成功。
            如果快慢指针在之前就相遇，则失败，返回空数组。
         */
        int[] inDegree = new int[numCourses];
        int[] res = new int[numCourses];
        if (numCourses <= 0 || prerequisites == null) {
            return res;
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int i = 0; i < prerequisites.length; i++) {
            inDegree[prerequisites[i][0]]++;
            if (graph.containsKey(prerequisites[i][1])) {
                graph.get(prerequisites[i][1]).add(prerequisites[i][0]);
            } else {
                List<Integer> list = new ArrayList<>();
                list.add(prerequisites[i][0]);
                graph.put(prerequisites[i][1], list);
            }
        }
        // BFS
        int first = 0, last = 0;
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                res[last++] = i;
            }
        }
        while (first < last) {
            // 后续课
            List<Integer> subCourses = graph.get(res[first]);
            if (subCourses != null) {
                for (int i = 0; i < subCourses.size(); i++) {
                    if (--inDegree[subCourses.get(i)] == 0) {
                        res[last++] = subCourses.get(i);
                    }
                }
            }
            first++;
        }

        if (last != numCourses) {
            return new int[0];
        }
        return res;
    }

    /**
     * LeetCode.213 打家劫舍II
     * <p>
     * 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都围成一圈，
     * 这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相
     * 邻的房屋在同一晚上被小偷闯入，系统会自动报警。
     * 给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。
     * <p>
     * 示例 1:
     * 输入: [2,3,2]
     * 输出: 3
     * 解释: 你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
     * <p>
     * 示例 2:
     * 输入: [1,2,3,1]
     * 输出: 4
     * 解释: 你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     *      偷窃到的最高金额 = 1 + 3 = 4 。
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        /*
            动态规划，类似打家劫舍I。
            在I中，有两个数组偷和不偷。在这个题上，加上两个条件，偷当前&&偷第一个、不偷当前&&偷第一个、
            不偷当前&&偷第一个。不偷当前&&不偷第一个。四个数组，两两一组。
            rob_rowFirst、rob_notRowFirst、nr_rf、nr_nrf。
            分析方法同I。

            转移方程：
            rob_rf[i] = nRob_rf[i - 1] + nums[i]
            nRob_rf[i] = Math.max(rob_rf[i - 1], nRob_rf[i - 1])

            rob_nrf[i] = nRob_nrf[i - 1] + nums[i]
            nRob_nrf[i] = Math.max(rob_nrf[i - 1], nRob_nrf[i - 1])

            最后分析有效结果。
            1.偷当前&&偷第一个的情况，偷到最后的时候绕了一圈，由于第一个房屋和最后一个房屋是紧挨着的，
            不能偷，所以该情况是失效的。
            2.不偷当前&&偷第一个，结束的时候最后未偷，可以满足题意，有效。
            3.后两种情况，由于第一个没有偷，最后一个可以偷也可以不偷，所以后两种都有效。
            所以，最后的答案是在上面三个情况中选最大值。
         */
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }

        int[] rob_rf = new int[n];
        int[] nRob_rf = new int[n];
        int[] rob_nrf = new int[n];
        int[] nRob_nrf = new int[n];
        rob_rf[0] = nums[0];

        for (int i = 1; i < nums.length; i++) {
            rob_rf[i] = nRob_rf[i - 1] + nums[i];
            nRob_rf[i] = Math.max(rob_rf[i - 1], nRob_rf[i - 1]);
            rob_nrf[i] = nRob_nrf[i - 1] + nums[i];
            nRob_nrf[i] = Math.max(rob_nrf[i - 1], nRob_nrf[i - 1]);
        }

        return Math.max(nRob_rf[n - 1], Math.max(rob_nrf[n - 1], nRob_nrf[n - 1]));
    }

    /**
     * LeetCode.216 组合总和III
     * <p>
     * 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合
     * 中不存在重复的数字。
     * <p>
     * 说明：
     * 所有数字都是正整数。
     * 解集不能包含重复的组合。 
     * <p>
     * 示例 1:
     * 输入: k = 3, n = 7
     * 输出: [[1,2,4]]
     * <p>
     * 示例 2:
     * 输入: k = 3, n = 9
     * 输出: [[1,2,6], [1,3,5], [2,3,4]]
     *
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<>();
        if (k == 0 || n == 0) {
            return res;
        }
        dfs(res, new ArrayList<>(), 1, k, n);
        return res;
    }

    private void dfs(List<List<Integer>> res, List<Integer> curr, int index, int k, int n) {
        if (n == 0 && k == 0) {
            res.add(new ArrayList<>(curr));
            return;
        } else if (n > 0 && k > 0) {
            for (int i = index; i <= 9; i++) {
                curr.add(i);
                dfs(res, curr, i + 1, k - 1, n - i);
                curr.remove(curr.size() - 1);
            }
        }
    }

    /**
     * LeetCode.221 最大正方形
     * <p>
     * 在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。
     * <p>
     * 示例:
     * 输入:
     * 1 0 1 0 0
     * 1 0 1 1 1
     * 1 1 1 1 1
     * 1 0 0 1 0
     * 输出: 4
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        /*
            动态规划。最大的正方形就是由右下角的数字开始往左上角扩散，看形成的面积是多少。
            ①state: dp[i][j]：到(i, j)位置时，由1组成的最大正方形的边长。
            ②init: 长或宽某个为0，且该位置的值是'1'时，dp[i][j] = 1.
            ③func: 由于是由右下角往左上角依次扩散。所以要比对的值是左边，上边和左上角。如果当前位置为'1'，
            则：
                dp[i][j] = min{dp[i-1][j], dp[i][j-1], dp[i-1][j-1]} + 1
                最大边长：maxLen = max{maxLen, dp[i][j]}
            ④result:  maxLen * maxLen
         */
        int m = matrix.length;
        int n = matrix.length > 0 ? matrix[0].length : 0;
        int[][] dp = new int[m][n];
        // 最大边长
        int maxLen = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    maxLen = Math.max(maxLen, dp[i][j]);
                }
            }
        }
        return maxLen * maxLen;
    }

    /**
     * LeetCode.236 二叉树的最近公共祖先
     *
     * 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
     * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先
     * 表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以
     * 是它自己的祖先）。”
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        /*
            分治思想
            如果p和q都存在在root下面，return lca(p, q)
            如果p和q只有一个在，返回在的那个
            如果都不在，return null
         */
        if (root == null || root == p || root == q) {
            return root;
        }

        // Divide
        // 如果p和q都在左子树或者右子树，则其中另外一个会是null
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        // 当p和q分别位于左子树和右子树，在两个子树中不会有最近公共祖先，所以祖先就是root
        if (left != null && right != null) {
            return root;
        }

        if (left != null) {
            return left;
        }

        if (right != null) {
            return right;
        }

        return null;

    }

    /**
     * LeetCode.287 寻找重复数
     * <p>
     * 给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），
     * 可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。
     * <p>
     * 示例 1:
     * 输入: [1,3,4,2,2]
     * 输出: 2
     * <p>
     * 示例 2:
     * 输入: [3,1,3,4,2]
     * 输出: 3
     * <p>
     * 说明：
     * 不能更改原数组（假设数组是只读的）。
     * 只能使用额外的 O(1) 的空间。
     * 时间复杂度小于 O(n2) 。
     * 数组中只有一个重复的数字，但它可能不止重复出现一次。
     *
     * @param nums
     * @return
     */
    public int findDuplicate(int[] nums) {
        /*
                使用判定链表有环的快慢指针法。我们先设置慢指针slow和快
            指针fast，慢指针每次走一步，快指针每次走两步，两个指针在
            有环的情况下一定会相遇，此时我们再将slow 放置起点0，两个
            指针每次同时移动一步，相遇的点就是答案。
         */
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
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

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {
    }

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}

