/**
 * description: 练习算法的项目
 * @date: 2020/10/17 08:30
 * @author: Grieey
 */
fun main(args: Array<String>) {
//  testMaxProfit()
//  testSlidingWindow()
//  testRange()
//  testNSum()
//  testTree()
//  testOffer()
  val a = bubble(intArrayOf(2, 1, 7, 5, 9, 8))
  val b = selectedSort(intArrayOf(2, 1, 7, 5, 9, 8))
  val c = mergeSort(intArrayOf(2, 1, 7, 5, 9, 8))
  val d = quickSort(intArrayOf(2, 1, 7, 5, 9, 8), 0, 5)
  val d2 = quickSort(intArrayOf(3, 2, 1, 5, 6, 4), 0, 5)
  val a1 = findKthLargest(intArrayOf(3, 2, 3, 1, 2, 4, 5, 5, 6), 2)
  val a2 = findKthLargestWithPriorityQueue(intArrayOf(3, 2, 3, 1, 2, 4, 5, 5, 6), 2)
  println(a.size)
}
