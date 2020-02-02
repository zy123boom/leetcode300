package leetcode300_hard;

/**
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
}
