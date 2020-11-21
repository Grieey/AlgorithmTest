import java.lang.Integer.max

/**
 * description: 动态规划相关的算法
 * @date: 2020/10/17 08:50
 * @author: Grieey
 */
fun testMaxProfit() {
//  println(maxProfitWithK(2, intArrayOf(3, 2, 6, 5, 0, 3)))
//  println(maxProfitWithK2(2, intArrayOf(3, 2, 6, 5, 0, 3)))
//  println(maxProfitWithK(1, intArrayOf(1, 2)))
//  println(maxProfitWithK2(1, intArrayOf(1, 2)))
//  println(maxProfitWithK2(2, intArrayOf(1, 2, 4, 2, 5, 7, 2, 4, 9, 0)))
//  println(superEggDrop(2, 6))
//  println(superEggDropWithDp(2, 6))
//  println(superEggDropWithDp2(3, 14))

  println(canPartition(intArrayOf(1, 5, 11, 5)))
}

/**
 * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
 *
 * 如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
 *
 * 注意：你不能在买入股票前卖出股票。
 *
 *  
 *
 * 示例 1:
 *
 * 输入: [7,1,5,3,6,4]
 * 输出: 5
 * 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
 * 注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
 * 示例 2:
 *
 * 输入: [7,6,4,3,1]
 * 输出: 0
 * 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
 */
fun maxProfit(prices: IntArray): Int {
  // 先对入参进行检查
  if (prices.isEmpty() || prices.size == 1) return 0

  // 初始化没有持有股票时的利润为0
  var profit0 = 0
  // 初始化持有股票时的利润为第一天的股票价格
  var profit1 = -prices[0]

  // 遍历每一天的股票价格
  for (day in 1..prices.lastIndex) {
    /**
     * 第i天的利润有两种状态可以选择，仍然持有股票或者没有持有股票
     * 1.针对没有持有股票的状态，有两种方式会导致这种状态
     *    一是第i天时，选择无操作，而前一天也是没有持有股票的状态；
     *    二是第i天时，选择卖出前一天持有的股票。
     * 2.针对有持有股票的状态，也有两种方式会导致这种状态
     *    一是第i天时，选择无操作，而前一天也是持有股票的状态；
     *    二是第i天时，选择买入当天的股票。
     */
    profit0 = profit0.coerceAtLeast(profit1 + prices[day])
    profit1 = profit1.coerceAtLeast(-prices[day])
  }

  return profit0
}

/**
 * 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
 *
 * 设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
 *
 * 注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
 *
 *  
 *
 * 示例 1:
 *
 * 输入: [7,1,5,3,6,4]
 * 输出: 7
 * 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
 *      随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
 * 示例 2:
 *
 * 输入: [1,2,3,4,5]
 * 输出: 4
 * 解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
 *      注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
 *      因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
 * 示例 3:
 *
 * 输入: [7,6,4,3,1]
 * 输出: 0
 * 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
 */
fun maxProfitUtrlK(prices: IntArray): Int {
  if (prices.isEmpty() || prices.size == 1) return 0

  var profit0 = 0
  var profit1 = -prices[0]

  for (day in 1..prices.lastIndex) {
    val lastProfit0 = profit0
    profit0 = profit0.coerceAtLeast(profit1 + prices[day])
    profit1 = profit1.coerceAtLeast(lastProfit0 - prices[day])
  }

  return profit0
}

/**
 * 给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。​
 *
 * 设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
 *
 * 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
 * 卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
 * 示例:
 *
 * 输入: [1,2,3,0,2]
 * 输出: 3
 * 解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
 */
fun maxProfitUtrlKWithCoolDown(prices: IntArray): Int {
  if (prices.isEmpty() || prices.size == 1) return 0

  var profit0 = 0
  var profit1 = -prices[0]
  var profitPre = 0

  for (day in 1..prices.lastIndex) {
    val lastProfit0 = profit0
    profit0 = profit0.coerceAtLeast(profit1 + prices[day])
    profit1 = profit1.coerceAtLeast(profitPre - prices[day])
    profitPre = lastProfit0
  }

  return profit0
}

/**
 * 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
 *
 * 设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
 *
 * 注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
 *
 * 示例 1:
 *
 * 输入: [2,4,1], k = 2
 * 输出: 2
 * 解释: 在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
 * 示例 2:
 *
 * 输入: [3,2,6,5,0,3], k = 2
 * 输出: 7
 * 解释: 在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
 *      随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。
 */
@Deprecated("这个解法有问题", replaceWith = ReplaceWith("maxProfitWithK2"))
fun maxProfitWithK(k: Int, prices: IntArray): Int {
  if (prices.isEmpty() || prices.size == 1) return 0

  if (k > prices.size / 2) {
    return maxProfitUtrlK(prices)
  }

  val dp = Array(prices.size) { Array(k + 1) { IntArray(2) } }

  for (i in 0..prices.lastIndex) for (eachK in k downTo 1) {
    if (i - 1 == -1) {
      dp[0][eachK][0] = 0
      dp[0][eachK][1] = Int.MIN_VALUE
      continue
    }
    dp[i][eachK][0] = max(dp[i - 1][eachK][0], dp[i - 1][eachK][1] + prices[i])
    dp[i][eachK][1] = max(dp[i - 1][eachK][1], dp[i - 1][eachK - 1][0] - prices[i])
  }

  return dp[prices.lastIndex][k][0]
}

fun maxProfitWithK2(k: Int, prices: IntArray): Int {
  if (k < 1) {
    return 0
  }

  // k 超过了上限，也就变成了 无限次交易问题
  if (k > prices.size / 2) {
    return maxProfitUtrlK(prices)
  }
  // 状态定义
  val dp = Array(k) { IntArray(2) }

  // 初始化
  for (i in 0 until k) {
    dp[i][0] = Integer.MIN_VALUE
  }
  // 遍历每一天，模拟 k 次交易，计算并更新状态
  for (p in prices) {

    dp[0][0] = dp[0][0].coerceAtLeast(0 - p)                  // 第 1 次买
    dp[0][1] = dp[0][1].coerceAtLeast(dp[0][0] + p)         // 第 1 次卖

    for (i in 1 until k) {
      dp[i][0] = dp[i][0].coerceAtLeast(dp[i - 1][1] - p)   // 第 i 次买
      dp[i][1] = dp[i][1].coerceAtLeast(dp[i][0] + p)     // 第 i 次卖
    }
  }
  return dp[k - 1][1]
}

private fun robInRange(nums: IntArray, start: Int, end: Int): Int {
  var dp: Int = 0
  var dp1: Int = 0
  var dp2: Int = 0

  for (i in end downTo start) {
    dp = Math.max(dp1, (nums[i] + dp2))
    dp2 = dp1
    dp1 = dp
  }

  return dp
}

val map = mutableMapOf<TreeNode, Int>()

/**
 * 337 中等难度
 * 在上次打劫完一条街道之后和一圈房屋后，小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为“根”。 除了“根”之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果两个直接相连的房子在同一天晚上被打劫，房屋将自动报警。
 *
 * 计算在不触动警报的情况下，小偷一晚能够盗取的最高金额。
 *
 * 示例 1:
 *
 * 输入: [3,2,3,null,3,null,1]
 *
 * 3
 * / \
 * 2   3
 * \   \
 * 3   1
 *
 * 输出: 7
 * 解释: 小偷一晚能够盗取的最高金额 = 3 + 3 + 1 = 7.
 * 示例 2:
 *
 * 输入: [3,4,5,1,3,null,1]
 *
 *      3
 * / \
 * 4   5
 * / \   \
 * 1   3   1
 *
 * 输出: 9
 * 解释: 小偷一晚能够盗取的最高金额 = 4 + 5 = 9.
 */
fun rob(root: TreeNode?): Int {
  if (root == null) return 0

  if (map.containsKey(root)) return map[root] ?: 0

  val robLeft = if (root.left == null) 0 else rob(root.left?.left) + rob(root.left?.right)
  val robRight = if (root.right == null) 0 else rob(root.right?.left) + rob(root.right?.right)

  val rob = root.`val` + robLeft + robRight

  val notRob = rob(root.left) + rob(root.right)

  val result = Math.max(rob, notRob)

  map[root] = result

  return result
}

data class TreeNode(val `val`: Int) {
  var left: TreeNode? = null
  var right: TreeNode? = null
}

/**
 * 887. 鸡蛋掉落 困难
 * 你将获得 K 个鸡蛋，并可以使用一栋从 1 到 N  共有 N 层楼的建筑。
 *
 * 每个蛋的功能都是一样的，如果一个蛋碎了，你就不能再把它掉下去。
 *
 * 你知道存在楼层 F ，满足 0 <= F <= N 任何从高于 F 的楼层落下的鸡蛋都会碎，从 F 楼层或比它低的楼层落下的鸡蛋都不会破。
 *
 * 每次移动，你可以取一个鸡蛋（如果你有完整的鸡蛋）并把它从任一楼层 X 扔下（满足 1 <= X <= N）。
 *
 * 你的目标是确切地知道 F 的值是多少。
 *
 * 无论 F 的初始值如何，你确定 F 的值的最小移动次数是多少？
 *
 *
 *
 * 示例 1：
 *
 * 输入：K = 1, N = 2
 * 输出：2
 * 解释：
 * 鸡蛋从 1 楼掉落。如果它碎了，我们肯定知道 F = 0 。
 * 否则，鸡蛋从 2 楼掉落。如果它碎了，我们肯定知道 F = 1 。
 * 如果它没碎，那么我们肯定知道 F = 2 。
 * 因此，在最坏的情况下我们需要移动 2 次以确定 F 是多少。
 * 示例 2：
 *
 * 输入：K = 2, N = 6
 * 输出：3
 * 示例 3：
 *
 * 输入：K = 3, N = 14
 * 输出：4
 *
 *
 * 提示：
 *
 * 1 <= K <= 100
 * 1 <= N <= 10000
 */
fun superEggDrop(K: Int, N: Int): Int {
  val memo = Array(K + 1) { IntArray(N + 1) { -1 } }
  fun dp(k: Int, n: Int): Int {
    if (k == 1) return n
    if (n == 0) return 0

    var res = Int.MAX_VALUE
    if (memo[k][n] != -1) return memo[k][n]
    var lo = 1
    var hi = n
    while (lo <= hi) {
      val mid = (lo + hi) / 2
      val broken = dp(k - 1, mid - 1)
      val notBroken = dp(k, n - mid)
      if (broken > notBroken) {
        hi = mid - 1
        res = Math.min(res, broken + 1)
      } else {
        lo = mid + 1
        res = Math.min(res, notBroken + 1)
      }
    }
    memo[k][n] = res
    return res
  }

  return dp(K, N)
}

/**
 * 自底向上的dp解法
 */
fun superEggDropWithDp(K: Int, N: Int): Int {
  val dp = Array(K + 1) { IntArray(N + 1) }
  for (m in 1..N) {
    dp[0][m] = 0
    for (k in 1..K) {
      dp[k][m] = dp[k][m - 1] + dp[k - 1][m - 1] + 1
      if (dp[k][m] >= N) return m
    }
  }

  return N
}

/**
 * 鸡蛋掉落，鹰蛋（Leetcode 887）：（经典dp）
 * 有 K 个鸡蛋，有 N 层楼，用最少的操作次数 F 检查出鸡蛋的质量。
 *
 * 思路：
 * 本题应该逆向思维，若你有 K 个鸡蛋，你最多操作 F 次，求 N 最大值。
 *
 * dp[k][f] = dp[k][f-1] + dp[k-1][f-1] + 1;
 * 解释：
 * 0.dp[k][f]：如果你还剩 k 个蛋，且只能操作 f 次了，所能确定的楼层。
 * 1.dp[k][f-1]：蛋没碎，因此该部分决定了所操作楼层的上面所能容纳的楼层最大值
 * 2.dp[k-1][f-1]：蛋碎了，因此该部分决定了所操作楼层的下面所能容纳的楼层最大值
 * 又因为第 f 次操作结果只和第 f-1 次操作结果相关，因此可以只用一维数组。
 *
 * 时复：O(K*根号(N))
 */
fun superEggDropWithDp2(K: Int, N: Int): Int {
  val dp = IntArray(K + 1)
  var ans = 0
  while (dp[K] < N) {
    for (k in K downTo 1) dp[k] = dp[k] + dp[k - 1] + 1
    ans++
  }

  return ans
}

/**
 * 416. 分割等和子集 中等
 * 给定一个只包含正整数的非空数组。是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
 *
 * 注意:
 *
 * 每个数组中的元素不会超过 100
 * 数组的大小不会超过 200
 * 示例 1:
 *
 * 输入: [1, 5, 11, 5]
 *
 * 输出: true
 *
 * 解释: 数组可以分割成 [1, 5, 5] 和 [11].
 *
 *
 * 示例 2:
 *
 * 输入: [1, 2, 3, 5]
 *
 * 输出: false
 *
 * 解释: 数组不能分割成两个元素和相等的子集.
 */
fun canPartition(nums: IntArray): Boolean {
  // 当数组为空或者数组的和为奇数的时候，肯定没法分割为等和的子集，所以直接返回false
  if (nums.isEmpty() || nums.sum() % 2 != 0) return false

  val halfSum = nums.sum() / 2
  val dp = BooleanArray(halfSum + 1) { false }
  // base case
  dp[0] = true

  for (index in nums.indices) {
    for (reduceSum in halfSum downTo 0) {
      // 这里dp[index] = dp[index] 的意义是，选择不加当前这个数
      // dp[index] = dp[reduceSum - nums[index]] 加了当前这个数正好等于
      if (reduceSum - nums[index] >= 0) dp[reduceSum] = dp[reduceSum] || dp[reduceSum - nums[index]]
    }
  }
  return dp[halfSum]
}



