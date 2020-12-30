/**
 * description: n 数之和的问题
 * @date: 2020/10/31 12:10
 * @author: Grieey
 */
fun testNSum() {
  println("求2个数的和：")
  val res2Sum = nSum(intArrayOf(1, 2, 2, 4, 5, 7, 7, 8), 2, 0, 9)
  res2Sum.forEach {
    print("[${it[0]},${it[1]}]")
  }

  println("\n在1, 2, 3, 3, 4, 4, 7, 7, 8中求3个数的和：")
  val res3Sum = nSum(intArrayOf(1, 2, 3, 3, 4, 4, 7, 7, 8), 3, 0, 10)
  res3Sum.forEach {
    print("[${it[0]},${it[1]},${it[2]}]")
  }

  println("\n在-1, 0, 1, 2, -1, -4中求3个数的和：")
  val nums = intArrayOf(-1, 0, 1, 2, -1, -4)
  nums.sort()
  val res3Sum2 = nSum(nums, 3, 0, 0)
  res3Sum2.forEach {
    print("[${it[0]},${it[1]},${it[2]}]")
  }
}

/**
 * n数之和的通用方法
 * @param nums 源数组
 * @param countOfNum 需要几个数的和
 * @param start 从数组的第几个数开始查找
 * @param target 目标值
 * @return 返回不重复的countOfNum组合的数组
 */
fun nSum(nums: IntArray, countOfNum: Int, start: Int, target: Int): Array<IntArray> {
  val res = mutableListOf<IntArray>()
  // 至少需要两个数或者countOfNum 应该大于查找范围
  if (countOfNum < 2 || countOfNum > nums.size) return res.toTypedArray()

  if (countOfNum == 2) {
    // 两个数时，使用左右指针来求有多少个符合的数组
    var lo = start
    var hi = nums.lastIndex
    while (lo < hi) {
      val sum = nums[lo] + nums[hi]
      val left = nums[lo]
      val right = nums[hi]

      when {
        // 正好相等时，找到啦一组
        sum == target -> {
          // 将这组添加到数组中
          res.add(intArrayOf(nums[lo], nums[hi]))
          // 过滤掉低位相同的值
          while (left == nums[lo] && lo < hi) lo++
          // 过滤掉高位相同的值
          while (right == nums[hi] && lo < hi) hi--
        }
        // 目前的和大于了目标值，需要减小高位
        sum > target -> {
          while (right == nums[hi] && lo < hi) hi--
        }
        // 目前的和小于了目标值，需要增加低位
        sum < target -> {
          while (left == nums[lo] && lo < hi) lo++
        }
      }
    }
  } else {
    // 大于两数的和，需要穷举前一位的数，来递归调用自身，直到最后是求2数的和
    // 例如，求4数和，先遍历数组，穷举第一位，然后用target减去这一位，剩下的再调用nSum，就是求3数和了，同样，数组中需要去除这一位
    var i = start
    while (i < nums.size) {
      val last = nSum(nums, countOfNum - 1, i + 1, target - nums[i])
      last.forEach {
        res.add(it.plus(nums[i]))
      }

      // 过滤穷举的那一位数的重复数据
      while (i < nums.lastIndex && nums[i] == nums[i + 1]) i++
      i++
    }
  }

  return res.toTypedArray()
}