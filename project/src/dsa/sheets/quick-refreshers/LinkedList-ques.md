## LinkedList



**Q : Find Middle Node**
* Implement a method called findMiddleNode that returns the middle node of the linked list.

If the list has an even number of nodes, the method should return the second middle node.

```java

	public Node findMiddleNode(){
	    if(head == null || head.next==null){
	        return head;
	    }
	    else {
	        Node slow = head;
	        Node fast = head;
	        
	        while(fast.next!=null){
                slow= slow.next;
                fast= fast.next;
	            if(fast.next!=null){
                    fast = fast.next;
	            }
	        }
	        return slow;
	    }
	}
    
```
------------------------

**Q: LL: Has Loop**

* Write a method called hasLoop that is part of the linked list class.

* The method should be able to detect if there is a cycle or loop present in the linked list.

* Use Floyd's cycle-finding algorithm (also known as the "tortoise and the hare" algorithm) to detect the loop.

* **This algorithm uses two pointers:** a slow pointer and a fast pointer. The slow pointer moves one step at a time, while the fast pointer moves two steps at a time. If there is a loop in the linked list, the two pointers will eventually meet at some point. If there is no loop, the fast pointer will reach the end of the list.

```java

	public boolean hasLoop(){
	    Node fast = head; 
	    Node slow = head; 
	    
	    if(head == null || head.next == null){
	        return false;
	    }
	    
	    while(fast.next != null){
	        slow = slow.next; 
	        fast = fast.next; 
	            if(fast.next != null){
	                fast = fast.next;
	            }
	        if(fast == slow){
	            return true;
	        }
	    }
	    return false;
	}
```

**Q: LL: Find Kth Node From End**

* Implement a method called findKthFromEnd that returns the k-th node from the end of the list.
* If the list has fewer than k nodes, the method should return null.

```java

	public Node findKthFromEnd(int k){
	    Node fast = head;
	    Node slow = head;
	    
	    for(int i =0; i<k; i++){
	        if(fast == null){
	            return null;
	        } fast = fast.next;
	    }
	    while(fast != null){
	        slow = slow.next;
	        fast = fast.next;
	    }
	    return slow;
	}
```

**Q: LL: Remove Duplicates ( ** Interview Question)**
* You are given a singly linked list that contains integer values, where some of these values may be duplicated.

```java

  public void removeDuplicates() {
        Set<Integer> values = new HashSet<>();
        Node previous = null;
        Node current = head;
        while (current != null) {
            if (values.contains(current.value)) {
                previous.next = current.next;
                length -= 1;
            } else {
                values.add(current.value);
                previous = current;
            }
           current = current.next;
        }
    }
```

**Q LL: Binary to Decimal ( ** Interview Question)**
* You have a linked list where each node represents a binary digit (0 or 1). The goal of the binaryToDecimal function is to convert this binary number, represented by the linked list, into its decimal equivalent.

```java

    public int binaryToDecimal(){
        Node current = head;
        int sum = 0;
        
        if(current ==null){
            return 0;
        }
        
        while(current != null){
            sum= sum*2 + current.value;
            current = current.next;
        }
        return sum;
    }
```

**Q: LL: Partition List ( ** Interview Question)**
* Given a value x you need to rearrange the linked list such that all nodes with a value less than x come before all nodes with a value greater than or equal to x.

```java

   public void partitionList(int x){

        if(head == null){
            return;
        }
        
        Node dummy1 = new Node(0); 
        Node dummy2 = new Node(0);
        
        Node smaller = dummy1; 
        Node greater = dummy2;
        
        Node temp = head;

        
        while(temp != null){
            if(temp.value < x){
                smaller.next = temp;
                smaller = temp;
            } else{
                greater.next = temp;
                greater = temp;
            }
            temp = temp.next;

        }
        greater.next = null;

        smaller.next = dummy2.next;
        head = dummy1.next;
        
    }
```
**Q: reverse LinkedList Between two points**
* implement a method called reverseBetween that reverses the nodes of the list between indexes startIndex and endIndex (inclusive).

```java

    public void reverseBetween(int m, int n) {
    // Your implementation here
    
    Node dummy = new Node(0);
    dummy.next = head;
    
    if(head == null) return;
    
    Node previousNode = dummy;
    for(int i =0; i<m; i++){
        previousNode = previousNode.next;
    }
    
    Node current = previousNode.next;
    
    for(int i=0; i< (n-m); i++){
        Node nodeToMove = current.next;
        current.next = nodeToMove.next;
        nodeToMove.next = previousNode.next;
        previousNode.next = nodeToMove;
    }
    
    head = dummy.next;
    
    }
```

**LL: Swap Nodes in Pairs**
* Write a method swapPairs() inside a LinkedList class that swaps every two adjacent nodes in a singly linked list.
This method should update the linked list in-place by changing the next pointers — not by swapping values.

* The method should work correctly for:

	* empty lists,	
	* single-node lists,	
	* even-length lists,	
	* odd-length lists.

   ```java

     public void swapPairs() {
        
        if (head == null) return;
         
        Node dummy = new Node(0);
        dummy.next = head;
        Node previous = dummy;
        Node first = head;
    
        while (first != null && first.next != null) {
            Node second = first.next;
    
            // Perform the swap
            
            first.next = second.next; 
            previous.next = second;
            second.next= first;
    
            // Move pointers
            previous = first;
            first = first.next;
        }
    
        head = dummy.next;
    }

   ```
