/**
 * description: 滑动窗口相关的算法
 * @date: 2020/10/22 08:31
 * @author: Grieey
 */
fun testSlidingWindow() {
  println("源字符\"ADOBECODEBANC\", 目标字符串\"ABC\" 输出${minWindow("ADOBECODEBANC", "ABC")}")
  println("源字符\"abc\", 目标字符串\"ac\" 输出${minWindow("abc", "ac")}")
  println("s1:\"ab\" s2:\"eidbaooo\" 输出${checkInclusion("ab", "eidbaooo")}")
  println("s1:\"ab\" s2:\"eidboaoo\" 输出${checkInclusion("ab", "eidboaoo")}")
  println("s1:\"ab\" s2:\"ba\" 输出${checkInclusion("ab", "ba")}")
  println("s1:\"cbaebabacd\" s2:\"abc\" 输出${findAnagrams("cbaebabacd", "abc")}")
  println("s1:\"baa\" s2:\"aa\" 输出${findAnagrams("baa", "aa")}")
  println("s1:\"abcabcbb\"  输出${lengthOfLongestSubstring("abcabcbb")}")
}

/**
 * 给你一个字符串 S、一个字符串 T 。请你设计一种算法，可以在 O(n) 的时间复杂度内，从字符串 S 里面找出：包含 T 所有字符的最小子串。
 *
 *  
 *
 * 示例：
 *
 * 输入：S = "ADOBECODEBANC", T = "ABC"
 * 输出："BANC"
 *  
 *
 * 提示：
 *
 * 如果 S 中不存这样的子串，则返回空字符串 ""。
 * 如果 S 中存在这样的子串，我们保证它是唯一的答案。
 */
fun minWindow(s: String, t: String): String {
  if (s.isBlank() || t.isBlank()) return ""
  val needs = mutableMapOf<Char, Int>()
  val window = mutableMapOf<Char, Int>()

  // 将目标字符串中的字符的频数添加到needs中
  t.forEach {
    needs[it] = (needs[it] ?: 0) + 1
  }

  // window中字符频数和needs中字符频数一致的字符的个数
  var validCountInWindow = 0
  // 窗口左边界
  var left = 0
  // 窗口有边界
  var right = 0
  // 最小覆盖子串的长度
  var minLength = Int.MAX_VALUE
  // 最小覆盖子串的起始位置
  var start = 0

  while (right <= s.lastIndex) {
    val rightChar = s[right]
    right++
    // 如果目标窗口中包含了该字符
    if (needs.containsKey(rightChar)) {
      // 将该字符添加到窗口中
      window[rightChar] = (window[rightChar] ?: 0) + 1
      // 当窗口中的字符的频数和目标窗口中该字符的频数一致时，增加validCountInWindow
      if (window[rightChar] == needs[rightChar]) validCountInWindow++
    }

    // 窗口中合法的字符个数和目标窗口一致时，收缩左边界
    while (validCountInWindow == needs.size) {
      // 如果目前的左右边界的区间小于之前的最小长度，进行更新
      if (right - left < minLength) {
        minLength = right - left
        start = left
      }

      val leftChar = s[left]
      left++
      // 目标窗口包含左边界字符时
      if (needs.containsKey(leftChar)) {
        // 如果频数一致，则减少合法个数，这个时候，左边界的收缩停止
        if (window[leftChar] == needs[leftChar]) validCountInWindow--
        // 减少窗口中该字符的频数
        window[leftChar] = (window[leftChar] ?: 1) - 1
      }
    }
  }

  return if (minLength == Int.MAX_VALUE) "" else s.substring(start, start + minLength)
}

/**
 * 给定两个字符串 s1 和 s2，写一个函数来判断 s2 是否包含 s1 的排列。
 *
 * 换句话说，第一个字符串的排列之一是第二个字符串的子串。
 *
 * 示例1:
 *
 * 输入: s1 = "ab" s2 = "eidbaooo"
 * 输出: True
 * 解释: s2 包含 s1 的排列之一 ("ba").
 *  
 *
 * 示例2:
 *
 * 输入: s1= "ab" s2 = "eidboaoo"
 * 输出: False
 *  
 *
 * 注意：
 *
 * 输入的字符串只包含小写字母
 * 两个字符串的长度都在 [1, 10,000] 之间
 */
fun checkInclusion(s1: String, s2: String): Boolean {
  if (s1 == s2) return true

  var left = 0
  var right = 0
  var valid = 0

  val need = mutableMapOf<Char, Int>()
  val window = mutableMapOf<Char, Int>()

  s1.forEach {
    need[it] = (need[it] ?: 0) + 1
  }

  while (right <= s2.lastIndex) {
    val rightChar = s2[right]
    right++

    if (need.containsKey(rightChar)) {
      window[rightChar] = (window[rightChar] ?: 0) + 1
      if (window[rightChar] == need[rightChar]) valid++
    }

    /**
     * 左边界收缩的条件是当窗口大于s1的长度时，就有可能包含了s1的排列
     */
    while (right - left >= s1.length) {
      // 当窗口中的合法个数和目标一致时，则找到了至少一种排列
      if (valid == need.size) return true

      val leftChar = s2[left]
      left++

      if (need.containsKey(leftChar)) {
        if (window[leftChar] == need[leftChar]) valid--
        window[leftChar] = (window[leftChar] ?: 1) - 1
      }
    }
  }

  return false
}

/**
 * 给定一个字符串s和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。
 *
 * 字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。
 *
 * 说明：
 *
 * 字母异位词指字母相同，但排列不同的字符串。
 * 不考虑答案输出的顺序。
 * 示例 1:
 *
 * 输入:
 * s: "cbaebabacd" p: "abc"
 *
 * 输出:
 * [0, 6]
 *
 * 解释:
 * 起始索引等于 0 的子串是 "cba", 它是 "abc" 的字母异位词。
 * 起始索引等于 6 的子串是 "bac", 它是 "abc" 的字母异位词。
 *  示例 2:
 *
 * 输入:
 * s: "abab" p: "ab"
 *
 * 输出:
 * [0, 1, 2]
 *
 * 解释:
 * 起始索引等于 0 的子串是 "ab", 它是 "ab" 的字母异位词。
 * 起始索引等于 1 的子串是 "ba", 它是 "ab" 的字母异位词。
 * 起始索引等于 2 的子串是 "ab", 它是 "ab" 的字母异位词。
 */
fun findAnagrams(s: String, p: String): List<Int> {
  if (s.isBlank() || s.length < p.length) return emptyList()

  val need = mutableMapOf<Char, Int>()
  val window = mutableMapOf<Char, Int>()
  val indexs = mutableListOf<Int>()

  p.forEach {
    need[it] = (need[it] ?: 0) + 1
  }

  var left = 0
  var right = 0
  var valid = 0

  while (right <= s.lastIndex) {
    val rightChar = s[right]
    right++

    if (need.containsKey(rightChar)) {
      window[rightChar] = (window[rightChar] ?: 0) + 1
      if (window[rightChar] == need[rightChar]) valid++
    }

    // 当检查的字符的长度等于目标窗口的长度时，进行判断是否是目标的排列
    while (right - left >= p.length) {

      // 频数一样则是排列，更新索引
      if (valid == need.size) {
        indexs.add(left)
      }

      val leftChar = s[left]
      left++

      if (need.containsKey(leftChar)) {
        if (window[leftChar] == need[leftChar]) {
          valid--
        }

        window[leftChar] = (window[leftChar] ?: 1) - 1
      }
    }
  }

  return indexs
}

/**
 * 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
 *
 * 示例 1:
 *
 * 输入: "abcabcbb"
 * 输出: 3
 * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
 * 示例 2:
 *
 * 输入: "bbbbb"
 * 输出: 1
 * 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
 * 示例 3:
 *
 * 输入: "pwwkew"
 * 输出: 3
 * 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
 *      请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
 */
fun lengthOfLongestSubstring(s: String): Int {
  if (s.isEmpty()) return 0

  val window = mutableMapOf<Char, Int>()

  var left = 0
  var right = 0
  var maxLen = 0

  while (right <= s.lastIndex) {
    val rightChar = s[right]
    right++

    window[rightChar] = (window[rightChar] ?: 0) + 1

    while ((window[rightChar] ?: 0) > 1) {
      val leftChar = s[left]
      left++
      window[leftChar] = (window[leftChar] ?: 1) - 1
    }

    maxLen = maxLen.coerceAtLeast(right - left)
  }

  return maxLen
}