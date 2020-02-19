package leetcode300_normal;


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
            例如1->2->3->4
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
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                if (nums[mid] >= nums[0]) {
                    left = mid + 1;
                } else {
                    if (target < nums[0]) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
            } else if (nums[mid] > target) {
                if (nums[mid] >= nums[0]) {
                    if (target < nums[0]) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                } else {
                    right = mid - 1;
                }
            }
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

