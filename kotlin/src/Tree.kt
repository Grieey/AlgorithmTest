/**
 * description: 各种关于树的算法
 * @date: 2020/11/7 10:42
 * @author: Grieey
 */
fun testTree() {
  val root = TreeNode(4)
  root.left = TreeNode(7)
  root.right = TreeNode(1)
  root.left?.left = TreeNode(2)
  root.left?.right = TreeNode(3)
  root.right?.left = TreeNode(6)
  root.right?.right = TreeNode(9)
  val res = invertTree(root)
  println("${res?.`val`}")
}

/**
 * 226. 翻转二叉树
 * 翻转一棵二叉树。
 *
 * 示例：
 *
 * 输入：
 *
 * 4
 * /   \
 * 2     7
 * / \   / \
 * 1   3 6   9
 * 输出：
 *
 * 4
 * /   \
 * 7     2
 * / \   / \
 * 9   6 3   1
 */
fun invertTree(root: TreeNode?): TreeNode? {
  if (root == null) return null

  val new = TreeNode(root.`val`)
  new.left = invertTree(root.right)
  new.right = invertTree(root.left)
  return new
}