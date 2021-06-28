/**
 * description: 贪心算法
 * @date: 2020/10/31 10:39
 * @author: Grieey
 */
fun testRange() {
  val mock = Array(3) { IntArray(2) }
  mock[0][0] = 1
  mock[0][1] = 4
  mock[1][0] = 3
  mock[1][1] = 6
  mock[2][0] = 2
  mock[2][1] = 8

  println("区间[[1,4],[3,6],[2,8]] 输出${removeCoveredIntervals(mock)}")

  val result = merge(mock)
  result.forEach {
    println("区间[[1,4],[3,6],[2,8]] 输出[${it[0]},${it[1]}]")
  }

  val mock2 = Array(4) { IntArray(2) }
  mock2[0][0] = 1
  mock2[0][1] = 3
  mock2[1][0] = 2
  mock2[1][1] = 6
  mock2[2][0] = 8
  mock2[2][1] = 10
  mock2[3][0] = 15
  mock2[3][1] = 18
  val result2 = merge(mock2)
  println("区间[[1,3],[2,6],[8,10],[15,18]]输出")
  result2.forEach {
    print("[${it[0]},${it[1]}]")
  }
}
/**
 * 1288, 中等 ：给你一个区间列表，请你删除列表中被其他区间所覆盖的区间。
 *
 * 只有当 c <= a 且 b <= d 时，我们才认为区间 [a,b) 被区间 [c,d) 覆盖。
 *
 * 在完成所有删除操作后，请你返回列表中剩余区间的数目。
 *
 *  
 *
 * 示例：
 *
 * 输入：intervals = [[1,4],[3,6],[2,8]]
 * 输出：2
 * 解释：区间 [3,6] 被区间 [2,8] 覆盖，所以它被删除了。
 *  
 *
 * 提示：​​​​​​
 *
 * 1 <= intervals.length <= 1000
 * 0 <= intervals[i][0] < intervals[i][1] <= 10^5
 * 对于所有的 i != j：intervals[i] != intervals[j]
 */
fun removeCoveredIntervals(intervals: Array<IntArray>): Int {
  if (intervals.isEmpty() || intervals.size == 1) return 0

  intervals.sortWith(Comparator { o1, o2 ->
    if (o1[0] == o2[0]) {
      // 区间start 一致时，按照区间的end降序
      o2[1] - o1[1]
    } else {
      // 按照区间的start 升序
      o1[0] - o2[0]
    }
  })

  var res = 0
  // 初始化左右边界为第一个区间的位置
  var left = intervals[0][0]
  var right = intervals[0][1]

  for (i in 1..intervals.lastIndex) {
    val cur = intervals[i]
    when {
      // 被覆盖的情况，当前的区间位于左右边界内
      left <= cur[0] && cur[1] <= right -> res++
      // 右边界小于当前区间的左边界，那么是相交的情况，更新右边界，合并两个区间的范围
      right <= cur[0] -> right = cur[1]
      // 不相交，更新左右边界
      else -> {
        left = cur[0]
        right = cur[1]
      }
    }
  }

  return intervals.size - res
}

/**
 * 56. 合并区间 中等难度
 * 给出一个区间的集合，请合并所有重叠的区间。
 *
 *
 *
 * 示例 1:
 *
 * 输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
 * 输出: [[1,6],[8,10],[15,18]]
 * 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
 * 示例 2:
 *
 * 输入: intervals = [[1,4],[4,5]]
 * 输出: [[1,5]]
 * 解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
 * 注意：输入类型已于2019年4月15日更改。 请重置默认代码定义以获取新方法签名。
 *
 *
 *
 * 提示：
 *
 * intervals[i][0] <= intervals[i][1]
 */
fun merge(intervals: Array<IntArray>): Array<IntArray> {
  val res = mutableListOf<IntArray>()
  if (intervals.isEmpty()) return res.toTypedArray()

  intervals.sortBy {
    it[0]
  }
  res.add(intervals[0])

  for (i in 1..intervals.lastIndex) {
    val cur = intervals[i]
    when {
      // 当前区间的左边界小于合并后的区间的右边界，且当前区间的右边界大于合并后的区间的右边界时，进行合并
      cur[0] <= res.last()[1] && cur[1] > res.last()[1] -> res.last()[1] = cur[1]
      // 当前区间的左边界大于了合并后的右边界时，这是一个新的需要合并的区间了
      cur[0] > res.last()[1] -> res.add(cur)
    }
  }

  return res.toTypedArray()
}