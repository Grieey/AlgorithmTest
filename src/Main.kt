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

//  val a = bubble(intArrayOf(2, 1, 7, 5, 9, 8))
//  val b = selectedSort(intArrayOf(2, 1, 7, 5, 9, 8))
//  val c = mergeSort(intArrayOf(2, 1, 7, 5, 9, 8))
//  val d = quickSort(intArrayOf(2, 1, 7, 5, 9, 8), 0, 5)
//  val d2 = quickSort(intArrayOf(3, 2, 1, 5, 6, 4), 0, 5)
//  val a1 = findKthLargest(intArrayOf(3, 2, 3, 1, 2, 4, 5, 5, 6), 2)
//  val a2 = findKthLargestWithPriorityQueue(intArrayOf(3, 2, 3, 1, 2, 4, 5, 5, 6), 2)
//  println(a.size)

//  println(checkPermutation("abb", "aab"))

  val l1 = ListNode(3)
  l1.next = ListNode(7)
//  l1.next?.next = ListNode(3)
  val l2 = ListNode(9)
  l2.next = ListNode(2)
//  l2.next?.next = ListNode(3)

  val res = addTwoNumbers(l1, l2)
  println(res?.`val`)
}
