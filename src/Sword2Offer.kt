import java.util.*

/**
 * description: 剑指offer的练习题
 * @date: 2020/12/15 15:22
 * @author: Grieey
 */

fun testOffer() {
//  val a = buildTree(intArrayOf(3, 9, 20, 15, 7), intArrayOf(9, 3, 15, 20, 7))
  val fibRes = fib(45)
  println(fibRes)
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
        for (i in (low + 1) until high) {
          if (numbers[i] > numbers[x]) x = i
        }
        return numbers[x]
      }
    }
  }

  return numbers[low]
}