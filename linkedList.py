class Node:
    """Represents a single node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Manages a singly linked list."""
    def __init__(self):
        self.head = None

    def append(self, data):
        """Add a node with the given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return

        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        """Print all elements in the list."""
        if not self.head:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        """
        Delete the nth node in the list (1-based index).
        Raises:
            IndexError: If index is out of range or list is empty.
        """
        if not self.head:
            raise IndexError("Cannot delete from an empty list.")

        if n <= 0:
            raise IndexError("Index must be a positive integer.")

        if n == 1:
            self.head = self.head.next
            return

        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1

        if not current or not current.next:
            raise IndexError("Index out of range.")

        current.next = current.next.next


# --- Testing the LinkedList implementation ---
if __name__ == "__main__":
    ll = LinkedList()
    print("Adding nodes: 10, 20, 30, 40")
    ll.append(10)
    ll.append(20)
    ll.append(30)
    ll.append(40)

    print("Initial list:")
    ll.print_list()

    print("\nDeleting 2nd node (value 20):")
    try:
        ll.delete_nth_node(2)
    except IndexError as e:
        print("Error:", e)
    ll.print_list()

    print("\nDeleting 1st node (value 10):")
    try:
        ll.delete_nth_node(1)
    except IndexError as e:
        print("Error:", e)
    ll.print_list()

    print("\nAttempting to delete node at index 10 (out of range):")
    try:
        ll.delete_nth_node(10)
    except IndexError as e:
        print("Error:", e)

    print("\nFinal list:")
    ll.print_list()

    print("\nDeleting remaining nodes:")
    try:
        ll.delete_nth_node(1)
        ll.delete_nth_node(1)
        ll.delete_nth_node(1)  # Should raise error
    except IndexError as e:
        print("Error:", e)

    ll.print_list()
