import java.util.Objects;

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

  /**
   * 这道题的解法，经典，
   * 相交的情况，因为有焦点，所以fast和slow最终会指向同一个点，那就是焦点
   * 不相交的情况，那就是fast和slow都走了链表1+链表2的长度，路程一样，最后同时指向null，退出循环
   * 所以返回值fast不是null就是焦点
   * @param headA
   * @param headB
   * @return
   */
  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode fast = headA;
    ListNode slow = headB;
    while (fast != slow) {
      fast = fast != null ? fast.getNext() : headB;
      slow = slow != null ? slow.getNext() : headA;
    }
    return fast;
  }

  /**
   * 只能访问链表中的某个节点，然后删除他，非头尾节点
   * 这个思路很清晰简洁
   */
  public void deleteNode(ListNode node) {
    node.setVal(Objects.requireNonNull(node.getNext()).getVal());
    node.setNext(node.getNext().getNext());
  }
}
