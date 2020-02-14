package leetcode300_normal;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * leetcode前300道，中等题
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
            方法: HashSet+双指针。
            使用双指针，快指针j，慢指针i。让j走，如果集合里不包含
            当前字符，则加到字符里。如果包含了，则走慢指针i，判断
            i指针对应的字符是否出现在集合里，出现则移除掉，同时指针
            右移。最后看set的size。即res = max{res, set.size()}
         */
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
                r--;
                l++;
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
                r--;
                l++;
            }
        }
        return s.substring(ll, rr + 1);
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

