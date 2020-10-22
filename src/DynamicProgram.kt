import java.lang.Integer.max

/**
 * description: 动态规划相关的算法
 * @date: 2020/10/17 08:50
 * @author: Grieey
 */
fun testMaxProfit() {
//  println(maxProfitWithK(2, intArrayOf(3, 2, 6, 5, 0, 3)))
  println(maxProfit(2, intArrayOf(3, 2, 6, 5, 0, 3)))
//  println(maxProfitWithK(1, intArrayOf(1, 2)))
  println(maxProfit(1, intArrayOf(1, 2)))
  println(maxProfit(2, intArrayOf(1, 2, 4, 2, 5, 7, 2, 4, 9, 0)))
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

fun maxProfit_k_any(max_k: Int, prices: IntArray): Int {
  val n = prices.size
  if (max_k > n / 2) return maxProfitUtrlK(prices)
  val dp = Array(n) { Array(max_k + 1) { IntArray(2) } }
  for (i in 0 until n) for (k in max_k downTo 1) {
    if (i - 1 == -1) { /* 处理 base case */
      dp[0][k][0] = 0
      dp[0][k][1] = Int.MIN_VALUE
      continue
    }
    dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i])
    dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i])
  }
  return dp[n - 1][max_k][0]
}

fun maxProfit(k: Int, prices: IntArray): Int {
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

