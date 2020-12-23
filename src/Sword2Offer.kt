import java.util.*
import kotlin.collections.LinkedHashMap

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
  val a = TreeNode(10)
  a.left = TreeNode(12)
  a.right = TreeNode(6)
  a.left?.left = TreeNode(8)
  a.left?.right = TreeNode(3)
  a.right?.left = TreeNode(11)


  val b = TreeNode(10)
  b.left = TreeNode(12)
  b.right = TreeNode(6)
  b.left?.left = TreeNode(8)
//  println(isSubStructure(a, b))
  preOrder(a)
  inOrder(a)
  lastOrder(a)
  println("前序：$preBuilder ,中序:$inBuilder, 后序：$lastBuilder")

//  val a = spiralOrder(ArraysGet.twos2)
//  println(a.size)

//  println(validateStackSequences(ArraysGet.inStack, ArraysGet.outStack))
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

