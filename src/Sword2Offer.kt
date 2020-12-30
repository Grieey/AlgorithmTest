import jdk.nashorn.internal.runtime.PropertyHashMap
import java.util.*
import kotlin.collections.LinkedHashMap
import kotlin.random.Random

/**
 * description: 剑指offer的练习题
 * @date: 2020/12/15 15:22
 * @author: Grieey
 */

fun testOffer() {
//  val a = buildTree(intArrayOf(3, 9, 20, 15, 7), intArrayOf(9, 3, 15, 20, 7))
//  val fibRes = fib(45)
//  println(fibRes)

//  val array = Array(2) { CharArray(2) }
//  array[0] = charArrayOf('a', 'b')
//  array[1] = charArrayOf('c', 'd')
//  println(exist(array, "abdc"))

//  println(movingCount(1, 2, 1))

//  println(isNumber("-1E-16"))

//  val l1 = ListNode(1)
//  l1.next = ListNode(2)
//  l1.next?.next = ListNode(4)
//
//  val l2 = ListNode(1)
//  l2.next = ListNode(3)
//  l2.next?.next = ListNode(4)
//
//  var res = mergeTwoLists(l1, l2)
//  while (res != null) {
//    println(res.`val`)
//    res = res.next
//  }

  // [10,12,6,8,3,11]
  // [10,12,6,8]
//  val a = TreeNode(10)
//  a.left = TreeNode(12)
//  a.right = TreeNode(6)
//  a.left?.left = TreeNode(8)
//  a.left?.right = TreeNode(3)
//  a.right?.left = TreeNode(11)
//
//
//  val b = TreeNode(10)
//  b.left = TreeNode(12)
//  b.right = TreeNode(6)
//  b.left?.left = TreeNode(8)
//  println(isSubStructure(a, b))
//  preOrder(a)
//  inOrder(a)
//  lastOrder(a)
//  println("前序：$preBuilder ,中序:$inBuilder, 后序：$lastBuilder")

//  val a = spiralOrder(ArraysGet.twos2)
//  println(a.size)

//  println(validateStackSequences(ArraysGet.inStack, ArraysGet.outStack))

//  println(translateNum(25))

//  println(maxValue(ArraysGet.twos))
//  println(maxValue(ArraysGet.largeTwos))

//  println(firstUniqChar("leetcode"))

//  println(search(intArrayOf(1), 1))
//  println(search(intArrayOf(2, 2), 2))
//
//
//  println(search2(intArrayOf(1), 1))
//  println(search2(intArrayOf(2, 2), 2))

//  println(missingNumber(intArrayOf(0, 1, 2)))
//  println(missingNumber(intArrayOf(0)))

//  println(singleNumber(ArraysGet.threeTimesNum))

}


/**
 * 找出数组中重复的数字。
 * 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
 */
private fun findDupNums(nums: IntArray): Int {
  for (i in nums.indices) {
    while (nums[i] != i) {
      if (nums[i] == nums[nums[i]]) return nums[i]

      val temp = nums[i]
      nums[i] = nums[temp]
      nums[temp] = temp
    }
  }

  return -1
}

fun buildTree(preorder: IntArray, inorder: IntArray): TreeNode? {
  return buildTree(preorder, inorder, Int.MIN_VALUE)
}

private var pre = 0
private var inStop = 0

private fun buildTree(preorder: IntArray, inorder: IntArray, stop: Int): TreeNode? {
  if (pre >= preorder.size) return null

  if (inorder[inStop] == stop) {
    inStop++
    return null
  }

  val root = TreeNode(preorder[pre++])
  root.left = buildTree(preorder, inorder, root.`val`)
  root.right = buildTree(preorder, inorder, stop)
  return root
}

fun fib(n: Int): Int {
  if (n == 0) return 0
  if (n == 1 || n == 2) return 1

  var prev = 1
  var cur = 1
  for (i in 3..n) {
    val sum = cur + prev
    prev = cur
    cur = sum
  }

  return cur
}

fun minArray(numbers: IntArray): Int {
  var low = 0
  var high = numbers.lastIndex
  while (low <= high) {
    val mid = (low + high) / 2
    when {
      // 中间数小于最后一位时，将左边界增加，此时目标数在中间数右边
      numbers[mid] > numbers[high] -> low = mid + 1
      // 右边界减少，此时目标数在中间数左边
      numbers[mid] < numbers[low] -> high = mid
      // else 就是左边界、右边界及中间数都相等的情况，
      // 此时，在(low, high)这个区间一定存在一定数相等，只能进行线性遍历了
      else -> {
        var x = low
        for (i in (low + 1) until mid) {
          if (numbers[i] > numbers[x]) x = i
        }
        return numbers[x]
      }
    }
  }
  return numbers[low]
}

fun exist(board: Array<CharArray>, word: String): Boolean {
  if (word.isEmpty() || board.isEmpty() || board.size * board[0].size < word.length) return false

  val words = word.toCharArray()
  for (row in board.indices) for (col in board[0].indices) {
    if (dfs(board, words, row, col, 0)) return true
  }
  return false
}

private fun dfs(board: Array<CharArray>, words: CharArray, row: Int, col: Int, wordIndex: Int): Boolean {
  if (row < 0 || row >= board.size || col < 0 || col >= board[0].size || board[row][col] != words[wordIndex]) return false

  if (wordIndex == words.lastIndex) return true

  board[row][col] = ' '

  val res = dfs(board, words, row + 1, col, wordIndex + 1) ||
    dfs(board, words, row - 1, col, wordIndex + 1) ||
    dfs(board, words, row, col + 1, wordIndex + 1) ||
    dfs(board, words, row, col - 1, wordIndex + 1)
  board[row][col] = words[wordIndex]
  return res
}

fun movingCount(m: Int, n: Int, k: Int): Int {
  if (k == 0) return 1
  val visited = BooleanArray(m * n) { false }
  return core(k, m, n, 0, 0, visited)
}

private fun core(k: Int, rows: Int, cols: Int, row: Int, col: Int, visited: BooleanArray): Int {
  if (canArrived(k, rows, cols, row, col, visited)) {
    visited[row * cols + col] = true
    return 1 + core(k, rows, cols, row - 1, col, visited) + core(k, rows, cols, row + 1, col, visited) + core(k, rows, cols, row, col - 1, visited) + core(k, rows, cols, row, col + 1, visited)
  }

  return 0
}

private fun canArrived(k: Int, rows: Int, cols: Int, row: Int, col: Int, visited: BooleanArray): Boolean {
  if (row !in 0 until rows || col !in 0 until cols) return false

  if (visited[row * rows + col]) return false

  if ((getSum(row) + getSum(col)) > k) return false

  return true
}

private fun getSum(number: Int): Int {
  var sum = 0
  var num = number
  while (num > 0) {
    sum += num % 10
    num /= 10
  }
  return sum
}

fun cuttingRope(n: Int): Int {
  val dp = IntArray(n + 1)
  for (i in 2..n) for (j in 1..i) {
    dp[i] = Math.max(dp[i], Math.max(j * (i - j), j * dp[i - j]))
  }
  return dp[n]
}

/**
 * 自定义实现pow方法
 */
fun myPow(x: Double, n: Int): Double {
  if (n == 0) return 1.0
  if (n < 0) {
    return 1 / x * myPow(1 / x, -n - 1)
  }

  return if (n % 2 == 0) myPow(x * x, n / 2) else x * myPow(x * x, n / 2)
}

/**
 * 判断是否是数字
 */
fun isNumber(s: String): Boolean {
  if (s.isEmpty()) return false
  var index = 0
  val array = "$s|"

  val scanUnsignNumber = {
    val before = index
    while (array[index] in '0'..'9') index++
    index > before
  }

  val scanNumber = {
    if (array[index] == '+' || array[index] == '-') index++
    scanUnsignNumber()
  }

  while (s[index] == ' ') ++index
  var numberic = scanNumber()

  if (array[index] == '.') {
    index++
    numberic = scanUnsignNumber() || numberic
  }

  if (array[index] == 'e' || array[index] == 'E') {
    index++
    numberic = scanNumber() && numberic
  }

  while (array[index] == ' ') index++

  return numberic && array[index] == '|'
}

fun exchange(nums: IntArray): IntArray {
  return recoder(nums) {
    it % 2 == 0
  }
}

fun recoder(nums: IntArray, block: (n: Int) -> Boolean): IntArray {
  if (nums.isEmpty()) return intArrayOf()

  var left = 0
  var right = nums.lastIndex
  while (left < right) {
    while (left < right && !block(nums[left])) left++
    while (left < right && block(nums[right])) right++
    if (left < right) {
      val temp = nums[left]
      nums[left] = nums[right]
      nums[right] = temp
    }
  }

  return nums
}

fun reverseList(head: ListNode?): ListNode? {
  var next = head?.next
  var cur = head
  cur?.next = null
  while (next != null) {
    val tmp = next
    tmp.next = cur
    cur = tmp

    next = next.next
  }

  return cur
}

fun mergeTwoLists(l1: ListNode?, l2: ListNode?): ListNode? {
  if (l1 == null) return l2
  if (l2 == null) return l1

  var cur1 = l1
  var cur2 = l2
  var cur3: ListNode? = null
  if (cur1.`val` < cur2.`val`) {
    cur3 = ListNode(cur1.`val`)
    cur1 = cur1.next
  } else {
    cur3 = ListNode(cur2.`val`)
    cur2 = cur2.next
  }

  val res = cur3

  while (cur1 != null && cur2 != null) {
    if (cur1.`val` < cur2.`val`) {
      cur3?.next = ListNode(cur1.`val`)
      cur3 = cur3?.next
      cur1 = cur1.next
    } else {
      cur3?.next = ListNode(cur2.`val`)
      cur3 = cur3?.next
      cur2 = cur2.next
    }
  }

  when {
    cur1 == null -> {
      cur3?.next = cur2
    }
    cur2 == null -> {
      cur3?.next = cur1
    }
  }

  return res
}

/**
 * 递归用的出生入化
 */
fun isSubStructure(a: TreeNode?, b: TreeNode?): Boolean {
  return (a != null && b != null) && (recur(a, b) || isSubStructure(a.left, b) || isSubStructure(a.right, b))
}

private fun recur(a: TreeNode?, b: TreeNode?): Boolean {
  if (b == null) return true
  if (a == null || a.`val` != b.`val`) return false
  return recur(a.left, b) && recur(a.right, b)
}

val preBuilder = StringBuilder()
fun preOrder(a: TreeNode?) {
  if (a == null) {
    preBuilder.append("#").append(",")
    return
  }

  preBuilder.append(a.`val`).append(",")
  preOrder(a.left)
  preOrder(a.right)
}

val inBuilder = StringBuilder()
fun inOrder(a: TreeNode?) {
  if (a == null) {
    inBuilder.append("#").append(",")
    return
  }

  inOrder(a.left)
  inBuilder.append(a.`val`).append(",")
  inOrder(a.right)
}

val lastBuilder = StringBuilder()
fun lastOrder(a: TreeNode?) {
  if (a == null) {
    lastBuilder.append("#").append(",")
    return
  }

  lastOrder(a.left)
  lastOrder(a.right)
  lastBuilder.append(a.`val`).append(",")
}

fun isSymmetric(root: TreeNode?): Boolean {
  mutableListOf<Int>()
  return isSymmetric(root, root)
}

fun isSymmetric(a: TreeNode?, b: TreeNode?): Boolean {
  if (a == null && b == null) return true
  if (a == null || b == null) return false
  if (a.`val` != b.`val`) return false
  return isSymmetric(a.left, b.right) && isSymmetric(a.right, b.left)
}

fun spiralOrder(matrix: Array<IntArray>): IntArray {
  if (matrix.isEmpty() || matrix[0].isEmpty()) return intArrayOf()

  var left = 0
  var right = matrix[0].lastIndex
  var top = 0
  var bottom = matrix.lastIndex
  val array = IntArray((right + 1) * (bottom + 1))
  var start = 0

  while (true) {
    for (i in left..right) array[start++] = matrix[top][i]
    if (++top > bottom) break

    for (i in top..bottom) array[start++] = matrix[i][right]
    if (--right < left) break

    for (i in right downTo left) array[start++] = matrix[bottom][i]
    if (--bottom < top) break

    for (i in bottom downTo top) array[start++] = matrix[i][left]
    if (++left > right) break
  }

  return array
}

fun validateStackSequences(pushed: IntArray, popped: IntArray): Boolean {
  if (pushed.isEmpty() || popped.isEmpty() || pushed.size != popped.size) return true
  val inStack = Stack<Int>()

  var outIndex = 0
  for (i in pushed.indices) {
    inStack.push(pushed[i])
    while (!inStack.isEmpty() && inStack.peek() == popped[outIndex]) {
      inStack.pop()
      outIndex++
    }
  }


  return inStack.isEmpty()
}

fun levelOrder(root: TreeNode?): IntArray {
  if (root == null) return intArrayOf()
  val list = mutableListOf<Int>()
  val queue = LinkedList<TreeNode>()
  while (queue.isNotEmpty()) {
    val node = queue.poll()
    list.add(node.`val`)
    if (node.left != null) queue.offer(node.left)
    if (node.right != null) queue.offer(node.right)
  }

  return list.toIntArray()
}

fun levelOrder2(root: TreeNode?): List<List<Int>> {
  if (root == null) return emptyList()
  val list = LinkedHashMap<Int, MutableList<Int>>()
  val queue = LinkedList<TreeNode>()
  queue.offer(root)
  var death = 1

  while (queue.isNotEmpty()) {
    for (i in queue.indices) {
      val node = queue.poll()
      if (list[death] == null) {
        list[death] = mutableListOf(node.`val`)
      } else {
        list[death]?.add(node.`val`)
      }

      if (node.left != null) queue.offer(node.left)
      if (node.right != null) queue.offer(node.right)
    }

    death++
  }

  val res = mutableListOf<List<Int>>()

  list.forEach {
    res.add(it.value.toList())
  }

  return res.toList()
}

fun verifyPostorder(postorder: IntArray): Boolean {
  return verifyPostorder2(postorder, 0, postorder.lastIndex)
}

private fun verifyPostorder2(postorder: IntArray, start: Int, end: Int): Boolean {
  if (start >= end) return true
  var p = start
  while (postorder[p] < postorder[end]) p++
  val m = p
  while (postorder[p] > postorder[end]) p++
  return p == end && verifyPostorder2(postorder, start, m - 1) && verifyPostorder2(postorder, m, end - 1)
}

class Node(var `val`: Int) {
  var next: Node? = null
  var random: Node? = null
}

fun majorityElement(nums: IntArray): Int {
  var tar = nums.first()
  var times = 1
  // 解决思路是，依次判断出现数字的重复次数，当tar与下一个数一样的时候，就加一，不一样就减一，这样最后一个一定是出现次数最多的那个数
  for (i in 1..nums.lastIndex) {
    if (nums[i] == tar) {
      times++
    } else {
      times--
    }

    if (times == 0) {
      times = 1
      tar = nums[i]
    }
  }

  return tar
}

fun maxSubArray(nums: IntArray): Int {
  if (nums.size == 1) return nums.first()
  var dp0 = nums[0]
  var dp1 = 0
  var res = dp0
  // dpi的意义是，以index为i的数组成的数组的最大和
  for (i in 1..nums.lastIndex) {
    // 这里做出选择，是以当前nums[i]作为最大和还是和前面dp0结合组成最大和
    dp1 = Math.max(nums[i], nums[i] + dp0)
    dp0 = dp1
    res = Math.max(res, dp1)
  }

  return res
}

fun findNthDigit(n: Int): Int {
  var digit = 1
  var start = 1L
  var count = 9L
  var cur = n
  while (cur > count) { // 1.
    cur = (cur - count).toInt()
    digit += 1
    start *= 10
    count = digit * start * 9
  }
  val num = start + (n - 1) / digit // 2.
  return num.toString()[(n - 1) % digit] - '0' // 3.
}

fun minNumber(nums: IntArray): String {
  if (nums.size == 1) return nums.first().toString()

  val strArray = Array(nums.size) { "" }
  for (i in nums.indices) {
    strArray[i] = nums[i].toString()
  }

  strArray.sortWith(Comparator { x, y -> "$x$y".compareTo("$y$x") })
  val builder = StringBuffer()
  strArray.forEach {
    builder.append(it)
  }
  return builder.toString()
}

fun translateNum(num: Int): Int {
  if (num < 10) return 1
  var dp0 = 1
  var dp1 = 1
  val numStr = num.toString()
  for (i in 2..numStr.length) {
    val tmp = numStr.substring(i - 2, i)
    val cur = if (tmp in "10".."25") dp0 + dp1 else dp1
    dp0 = dp1
    dp1 = cur
  }

  return dp1
}

fun maxValue(grid: Array<IntArray>): Int {
  val m = grid.size
  val n = grid.first().size
  for (i in 1 until m) {
    grid[i][0] += grid[i - 1][0]
  }

  for (j in 1 until n) {
    grid[0][j] += grid[0][j - 1]
  }

  for (i in 1 until m) for (j in 1 until n) {
    grid[i][j] += Math.max(grid[i - 1][j], grid[i][j - 1])
  }

  return grid[m - 1][n - 1]
}

fun nthUglyNumber(n: Int): Int {
  var a = 0
  var b = 0
  var c = 0
  val dp = IntArray(n) { 0 }
  dp[0] = 1
  for (i in 1 until n) {
    val n2 = dp[a] * 2
    val n3 = dp[b] * 3
    val n5 = dp[c] * 5
    dp[i] = Math.min(Math.min(n2, n3), n5)
    if (dp[i] == n2) a++
    if (dp[i] == n3) b++
    if (dp[i] == n5) c++
  }

  return dp[n - 1]
}

fun firstUniqChar(s: String): Char {

  val hash = LinkedHashMap<Char, Boolean>()

  for (i in 0..s.lastIndex) {
    hash[s[i]] = hash[s[i]] == null
  }
  for ((c, isDum) in hash) {
    if (isDum) {
      return c
    }
  }

  return ' '
}

fun search(nums: IntArray, target: Int): Int {
  if (nums.isEmpty()) return 0

  var left = 0
  var right = nums.lastIndex
  var targetIndex = -1
  while (left <= right && targetIndex == -1) {
    val mid = left + (right - left) / 2
    when {
      nums[mid] > target -> right = mid - 1
      nums[mid] < target -> left = mid + 1
      else -> {
        targetIndex = mid
      }
    }
  }

  if (targetIndex == -1) return 0

  left = targetIndex
  right = targetIndex
  var times = 1
  if (targetIndex != 0) {
    while (left - 1 >= 0 && nums[--left] == target) times++
  }

  if (targetIndex != nums.lastIndex) {
    while (right + 1 <= nums.lastIndex && nums[++right] == target) times++
  }


  return times
}

fun search2(nums: IntArray, target: Int): Int {
  // 例如 2，4，5，5，5，8, 只需要寻找4的右边界和5的右边界，两个相减就是5重复的次数
  return searchRight(nums, target) - searchRight(nums, target - 1)
}

private fun searchRight(nums: IntArray, target: Int): Int {
  var left = 0
  var right = nums.lastIndex
  while (left <= right) {
    val mid = (left + right) / 2
    when {
      nums[mid] > target -> right = mid - 1
      nums[mid] < target -> left = mid + 1
      else -> left = mid + 1
    }
  }
  return left
}

fun missingNumber(nums: IntArray): Int {
  var left = 0
  var right = nums.lastIndex

  while (left <= right) {
    val mid = (right + left) / 2
    if (nums[mid] != mid) right = mid - 1 else left = mid + 1
  }

  return left
}

fun isBalanced(root: TreeNode?): Boolean {
  if (root == null) return true
  val left = getDeathOfTree(root.left)
  val right = getDeathOfTree(root.right)
  return left != -1 && right != -1 && Math.abs(left - right) <= 1
}

private fun getDeathOfTree(root: TreeNode?): Int {
  if (root == null) return 0

  val left = getDeathOfTree(root.left)
  if (left == -1) return left
  val right = getDeathOfTree(root.right)
  if (right == -1) return right

  return if (Math.abs(left - right) <= 1) Math.max(left, right) + 1 else -1
}


fun singleNumber(nums: IntArray): Int {
  var ones = 0
  var twos = 0
  nums.forEach { num ->
    ones = ones xor num and twos.inv()
    twos = twos xor num and ones.inv()
  }
  return ones
}

fun reverseLeftWords(s: String, n: Int): String {
  val builder = StringBuilder()
  for (i in n until n + s.length) {
    builder.append(s[i % s.length])
  }

  return builder.toString()
}

/**
 * 这个题的思路是，
 * 使用双端队列，先将k个数内的数据按照大小放到队列中，队头是最大的，然后其次
 * 在形成一个窗口了开始遍历，如果是头部正好是当前
 */
fun maxSlidingWindow(nums: IntArray, k: Int): IntArray {
  val queue = LinkedList<Int>()
  val res = IntArray(nums.size - k + 1)
  for (i in 0 until k) {
    while (queue.isNotEmpty() && queue.peekLast() < nums[i]) queue.removeLast()
    queue.addLast(nums[i])
  }

  res[0] = queue.peekFirst()
  for (i in k..nums.lastIndex) {
    if (queue.peekFirst() == nums[i - k]) queue.removeFirst()
    while (queue.isNotEmpty() && queue.peekLast() < nums[i]) queue.removeLast()
    queue.addLast(nums[i])
    res[i - k + 1] = queue.peekFirst()
  }

  return res
}

fun dicesProbability(n: Int): DoubleArray {
  var pre = DoubleArray(6) { 1 / 6.0 }

  for (i in 2..n) {
    val tmp = DoubleArray(5 * i + 1)
    for (j in 0..pre.lastIndex) for (x in 0 until 6) {
      tmp[j + x] += pre[j] / 6
    }

    pre = tmp
  }

  return pre
}

fun lastRemaining(n: Int, m: Int): Int {
  var f = 0
  var i = 1
  while (i != n + 1) {
    f = (f + m) % i
    i++
  }
  return f
}

var sum = 0

/**
 * 这个题要求不能使用减乘除和条件及循环语句
 * 使用&& 来规避判断很牛逼
 */
fun sumNums(n: Int): Int {
  val x = n > 1 && sumNums(n - 1) > 0
  sum += n
  return sum
}

fun constructArr(a: IntArray): IntArray {
  if (a.isEmpty()) return intArrayOf()

  val b = IntArray(a.size)
  var tmp = 1
  b[0] = 1

  for (i in 1..a.lastIndex) {
    b[i] = a[i - 1] * b[i - 1]
  }

  for (i in a.lastIndex - 1 downTo 0) {
    tmp *= a[i + 1]
    b[i] *= tmp
  }

  return b
}

fun bubble(nums: IntArray): IntArray {
  if (nums.isEmpty()) return nums

  for (i in nums.indices) for (j in 0 until nums.lastIndex - i) {
    if (nums[j + 1] < nums[j]) nums.swap(j + 1, j)
  }

  return nums
}

fun selectedSort(nums: IntArray): IntArray {
  if (nums.isEmpty()) return nums

  for (i in nums.indices) for (j in i + 1..nums.lastIndex) {
    if (nums[j] < nums[i]) nums.swap(i, j)
  }
  return nums
}

fun mergeSort(nums: IntArray): IntArray {
  return divide(nums, 0, nums.lastIndex)
}

private fun divide(nums: IntArray, lo: Int, hi: Int): IntArray {
  if (lo >= hi) return intArrayOf()

  val mid = lo + (hi - lo) / 2

  // 将左边的排序
  divide(nums, lo, mid)
  // 将右边的排序
  divide(nums, mid + 1, hi)

  // 将排好序的数组组在一起
  return merge(nums, lo, hi, mid)
}


private fun merge(nums: IntArray, lo: Int, hi: Int, mid: Int): IntArray {
  val copy = nums.copyOf()

  var index = lo
  // 左边起始位置
  var leftIndex = lo
  // 右边起始位置
  var rightIndex = mid + 1

  while (index <= hi) {
    when {
      // 合并时，左边界的起点已经大于了中位，说明左边已经排序好了，直接开始复制右边的
      leftIndex > mid -> nums[index++] = copy[rightIndex++]
      // 合并时，右边已经大于了末尾，说明右边已经排序好了，直接复制左边的
      rightIndex > hi -> nums[index++] = copy[leftIndex++]
      // 左边的值小于右边的, 先排序右边值的到该位置，否则就是左边的值
      nums[leftIndex] > nums[rightIndex] -> nums[index++] = copy[rightIndex++]
      else -> nums[index++] = copy[leftIndex++]
    }
  }

  return nums
}

/**
 * 快速排序，时间复杂度为O(nlogn)
 *          空间复杂度为O(1)
 */
fun quickSort(nums: IntArray, lo: Int, hi: Int): IntArray {
  if (lo >= hi) return intArrayOf()
  // 选择有一个随机基点，分割数组，分别对基点的两边进行排序
  val p = partition(nums, lo, hi)

  quickSort(nums, lo, p)
  quickSort(nums, p + 1, hi)

  return nums
}

fun partition(nums: IntArray, lo: Int, hi: Int): Int {
  // 将随机点的值和hi对应的值交互，便于下面遍历
  nums.swap(Random.nextInt(lo, hi), hi)

  // 基点最终的位置
  var targetIndex = lo
  for (i in lo until hi) {

    // i的值小于基点值时，将该值交互到target左边
    if (nums[i] <= nums[hi]) {
      nums.swap(targetIndex++, i)
    }
  }

  // 最后将基点值和targetIndex交换, 这样在基点左边的值都小于基点，基点右边的值都大于基点
  nums.swap(targetIndex, hi)

  return targetIndex
}

fun IntArray.swap(index1: Int, index2: Int) {
  val tmp = this[index1]
  this[index1] = this[index2]
  this[index2] = tmp
}

fun findKthLargest(nums: IntArray, k: Int): Int {
  var left = 0
  var right = nums.lastIndex
  val target = nums.size - k
  while (true) {
    val p = partition(nums, left, right)
    when {
      p > target -> right = p - 1
      p < target -> left = p + 1
      else -> return nums[target]
    }
  }
}

fun findKthLargestWithPriorityQueue(nums: IntArray, k: Int): Int {
  val p = PriorityQueue<Int>()
  for (i in nums.indices) {
    p.add(nums[i])
    if (p.size > k) {
      p.poll()
    }
  }

  return p.peek()
}

fun checkPermutation(s1: String, s2: String): Boolean {
  if (s1.length != s2.length) return false

  val map1 = HashMap<Char, Int>()
  val map2 = HashMap<Char, Int>()

  s1.forEach {
    map1[it] = map2.getOrDefault(it, 0) + 1
  }

  s2.forEach {
    map2[it] = map2.getOrDefault(it, 0) + 1
    if ((map1[it] ?: 0 > map2[it] ?: 0) || map1[it] == null || map2[it] == null) return false
  }

  return true
}

fun setZeroes(matrix: Array<IntArray>): Unit {
  if (matrix.isEmpty() || matrix.first().isEmpty()) return

  var isCol1Zero = false
  var isRow1Zero = false

  // 判断首列是否存在0
  for (i in matrix.indices) {
    if (matrix[i][0] == 0) isRow1Zero = true
  }

  // 判断首行是否存在0
  for (i in matrix.first().indices) {
    if (matrix[0][i] == 0) isCol1Zero = true
  }

  // 从1列1行开始遍历，如果存在0，将首行首列置为0
  for (i in 1..matrix.lastIndex) for (j in 1..matrix.first().lastIndex) {
    if (matrix[i][j] == 0) {
      matrix[i][0] = 0
      matrix[0][j] = 0
    }
  }

  // 根据首行首列是否为0来将整行整列置为0
  for (i in 1..matrix.lastIndex) for (j in 1..matrix.first().lastIndex) {
    if (matrix[i][0] == 0 || matrix[0][j] == 0) {
      matrix[i][j] = 0
    }
  }

  // 如果首列存在0，首列置为0
  for (i in matrix.indices) {
    if (isRow1Zero) matrix[i][0] = 0
  }

  // 如果首行存在0，首行置为0
  for (i in matrix.first().indices) {
    if (isCol1Zero) matrix[0][i] = 0
  }

}
