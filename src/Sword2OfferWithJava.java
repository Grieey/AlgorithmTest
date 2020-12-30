/**
 * description: Java代码
 *
 * @date: 2020/12/23 14:16
 * @author: Grieey
 */
class Sword2OfferWithJava {

  public Node copyRandomList(Node head) {
    if (head == null) {
      return null;
    }

    Node cur = head;
    while (cur != null) {
      Node tmp = new Node(cur.getVal());
      tmp.setNext(cur.getNext());
      cur.setNext(tmp);
      cur = cur.getNext();
    }

    cur = head;
    while (cur != null) {
      if (cur.getRandom() != null) {
        cur.getNext().setRandom(cur.getRandom().getNext());
      }

      cur = cur.getNext().getNext();
    }

    Node pre = head;
    cur = head.getNext();
    Node res = head.getNext();
    while (cur.getNext() != null) {
      pre.setNext(pre.getNext().getNext());
      cur.setNext(cur.getNext().getNext());
      pre = pre.getNext();
      cur = cur.getNext();
    }

    return res;
  }

  private TreeNode pre, head;

  public TreeNode treeToDoublyList(TreeNode root) {
    if (root == null) return null;
    dfs(root);
    head.setLeft(pre);
    pre.setRight(head);
    return head;
  }

  private void dfs(TreeNode cur) {
    if (cur == null) return;

    dfs(cur.getLeft());

    if (pre != null) {
      pre.setRight(cur);
    } else {
      head = cur;
    }

    cur.setRight(pre);
    pre = cur;

    dfs(cur.getRight());
  }

  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode fast = headA;
    ListNode slow = headB;
    while (fast != slow) {
      fast = fast != null ? fast.getNext() : headB;
      slow = slow != null ? slow.getNext() : headA;
    }
    return fast;
  }
}
