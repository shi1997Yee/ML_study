
class Node:
    def __init__(self, v):
        self.value = v
        self.left = None
        self.right = None


def func1(root):
    res = []
    stack = []
    while root or len(stack) > 0:
        if root:
            res.append(root.value)
            stack.append(root)
            root = root.left
        else:
            node = stack.pop()
            root = node.right
    return res


def func2(root):
    res = []
    stack = []
    while root or len(stack) > 0:
        if root:
            stack.append(root)
            root = root.left
        else:
            node = stack.pop()
            res.append(node.value)
            root = node.right
    return res


def func3(root):
    res = []
    stack = []
    while root or len(stack) > 0:
        if root:
            res.append(root.value)
            stack.append(root)
            root = root.right
        else:
            node = stack.pop()
            root = node.left
    return res[::-1]


if __name__ == '__main__':
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node4 = Node(4)
    node5 = Node(5)
    node6 = Node(6)
    node7 = Node(7)

    node2.left = node4
    node2.right = node5

    node3.left = node6
    node3.right = node7

    node1.left = node2
    node1.right = node3
    # # 先
    # res = func1(node1)
    # 中
    # res = func2(node1)
    # 后
    res = func3(node1)
    print('-------', res)
