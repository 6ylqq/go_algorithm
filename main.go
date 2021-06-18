package main

import (
	"container/list"
	"fmt"
	"math"
	"strconv"
)

// 旋转数组中的最小的数字
func minArray(numbers []int) int {
	length := len(numbers)
	for i := 1; i < length; i++ {
		if numbers[i] < numbers[i-1] {
			return numbers[i]
		}
	}
	return numbers[0]
}

// 矩阵中的路径——回溯法
func exist(board [][]byte, word string) bool {
	// 行列
	rows := len(board)
	columns := len(board[0])

	for i := 0; i < rows; i++ {
		for j := 0; j < columns; j++ {
			if findPath(board, i, j, word, 0) {
				return true
			} else {
				continue
			}
		}
	}

	return false
}

func findPath(board [][]byte, i int, j int, word string, k int) bool {
	if k < len(word) {
		if i <= len(board)-1 &&
			i >= 0 &&
			j <= len(board[0])-1 &&
			j >= 0 &&
			int(board[i][j]) == int(word[k]) {
			board[i][j] = byte('0')
			result := findPath(board, i+1, j, word, k+1) ||
				findPath(board, i-1, j, word, k+1) ||
				findPath(board, i, j+1, word, k+1) ||
				findPath(board, i, j-1, word, k+1)

			// 回填本次尝试所变更的路线，防止下次尝试board被污染
			board[i][j] = word[k]
			return result
		} else {
			return false
		}
	} else {
		return true
	}
}

// 剑指 Offer 13. 机器人的运动范围
func movingCount(m int, n int, k int) int {
	// 建立一个二维坐标数组
	board := make([][]int, m)
	for i := range board {
		board[i] = make([]int, n)
	}
	return isVailPath(0, 0, k, 0, board)
}

func isVailPath(i int, j int, k int, count int, board [][]int) int {
	if countIJ(i, j) <= k && i >= 0 && j >= 0 && i < len(board) && j < len(board[0]) && board[i][j] == 0 {
		board[i][j] = 1
		count++
		return int(math.Max(
			math.Max(float64(isVailPath(i+1, j, k, count, board)), float64(isVailPath(i-1, j, k, count, board))),
			math.Max(float64(isVailPath(i, j+1, k, count, board)), float64(isVailPath(i, j-1, k, count, board))),
		))
	}
	return count
}

func countIJ(i int, j int) int {
	result := 0
	for i > 0 || j > 0 {
		result += i % 10
		result += j % 10
		i = i / 10
		j = j / 10
	}
	return result
}

// 剑指 offer 14 剪绳子
func cuttingRope(n int) int {
	// 从下往上解决
	if n < 2 {
		return 0
	}
	if n == 2 {
		return 1
	}
	if n == 3 {
		return 2
	}
	var temp = make([]int, n+1)
	temp[0] = 0
	temp[1] = 1
	temp[2] = 2
	temp[3] = 3
	max := 0
	for i := 4; i <= n; i++ {
		max = 0
		// 遍历数组，找最大乘积
		for j := 1; j <= i/2; j++ {
			max = int(math.Max(float64(temp[j]*temp[i-j]), float64(max))) % 1000000007
		}
		// 记录某个长度下的最优解，然后往上继续求解
		temp[i] = max
	}
	return temp[n]
}

// 剪绳子pro，需要对1000000007取模，
//不能用动规解决，因为取模后会发生一次大小的转变，需要用贪心算法，
//数学证明段长为3最优，当所有绳段长度相等时，乘积最大
func cuttingRopePro(n int) int {
	return 0
}

// 剑指15，二进制中1的个数
func hammingWeight(num uint32) int {
	result := 0
	flag := uint32(1)
	// 循环32次后变为0
	for flag != 0 {
		if num&flag != 0 {
			result++
		}
		flag = flag << 1
	}
	return result
}

// 和减一后的数做与运算，就能直接计算有多少个1
func hammingWeight2(num uint32) int {
	result := 0
	for num != 0 {
		result++
		num = num & (num - 1)
	}
	return result
}

// 剑指 Offer 16. 数值的整数次方
func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return x
	}
	result := x
	if n < 0 {
		for i := 1; i < -n; i++ {
			result *= x
		}
		return 1 / result
	}
	if n > 0 {
		for i := 1; i < n; i++ {
			result *= x
		}
	}
	return result
}

// 快速幂
func myPowFast(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return x
	}
	result := myPowFast(x, n>>1)
	// 调用到底后相乘，时间复杂度为nlogn
	result *= result

	// 是否是奇数
	if uint32(n)&uint32(1) == 1 {
		result *= x
	}
	return result
}

// PrintNumbers 剑指 Offer 17. 打印从1到最大的n位数
func printNumbers(n int) []int {
	result := make([]int, n)
	for i := 1; i < int(math.Pow(float64(10), float64(n))); i++ {
		result = append(result, i)
	}
	return result[1:]
}

// PrintNumbers 递归解法，即求全排列
func PrintNumbers(n int) {
	result := make([]string, n)
	for i := 0; i < 10; i++ {
		result[0] = strconv.Itoa(i)
		PrintNumbersUseRecursion(result, n, 0)
	}
}

func PrintNumbersUseRecursion(result []string, length int, index int) {
	// 一次排列结束
	if index == length {
		fmt.Printf("%v\n", result)
		return
	}
	for i := 0; i < 10; i++ {
		result[index] = strconv.Itoa(i)
		PrintNumbersUseRecursion(result, length, index+1)
	}
}

type ListNode struct {
	Val  int
	Next *ListNode
}

// 剑指 Offer 18. 删除链表的节点
func deleteNode(head *ListNode, val int) *ListNode {
	if head.Val == val {
		head = head.Next
		return head
	}
	pre := head
	next := pre.Next
	for next.Val != val {
		pre = next
		next = pre.Next
	}
	pre.Next = next.Next
	return head
}

// 剑指 Offer 19. 正则表达式匹配
func isMatch(s string, p string) bool {
	if p[0] == '.' || p[0] == s[0] {
		// 相互匹配的时候，匹配下一个字符
		return isMatch(s[1:], p[1:])
	}
	if p[0] == '*' {
		// 如果匹配到*，则会有两种情况
		// 忽略，继续下一个匹配
		return isMatch(s, p[2:]) ||
			isMatch(s[1:], p)
	}

	return false
}

// 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
func exchange(nums []int) []int {
	head := 0
	tail := len(nums) - 1
	for head < tail {

	}
	return nums
}

// 剑指 Offer 22. 链表中倒数第k个节点
func getKthFromEnd(head *ListNode, k int) *ListNode {
	pre := head
	next := pre
	for i := 0; i < k; i++ {
		next = next.Next
	}
	for next != nil {
		pre = pre.Next
		next = next.Next
	}
	return pre
}

// 剑指 Offer 24. 反转链表
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	pre := head
	next := head.Next
	head.Next = nil
	for next != nil {
		temp := next.Next
		next.Next = pre
		pre = next
		next = temp
	}
	return pre
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 剑指 Offer 26. 树的子结构
func isSubStructure(A *TreeNode, B *TreeNode) bool {
	return true
}

// 剑指 Offer 27. 二叉树的镜像
func mirrorTree(root *TreeNode) *TreeNode {
	mirrorT(root)
	return root
}

func mirrorT(root *TreeNode) {
	if root.Left == nil && root.Right == nil {
		return
	}
	// 交换紫薯
	temp := root.Left
	root.Left = root.Right
	root.Right = temp
	if root.Left != nil {
		mirrorTree(root.Left)
	}
	if root.Right != nil {
		mirrorTree(root.Right)
	}
}

// 剑指 Offer 28. 对称的二叉树
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return false
	}
	return isSym(root.Left, root.Right)
}

func isSym(right *TreeNode, left *TreeNode) bool {
	if right == nil && left == nil {
		return true
	}
	if left == nil || right == nil {
		return false
	}
	if left.Val == right.Val {
		return false
	}
	return isSym(right.Left, left.Right) && isSym(right.Right, left.Left)
}

// 剑指 Offer 29. 顺时针打印矩阵
func spiralOrder(matrix [][]int) []int {
	// 斜对角打印
	result := make([]int, 0)
	start := 0
	for len(matrix) > start*2 && len(matrix[0]) > start*2 {
		result = append(result, printCircle(matrix, start)...)
		start++
	}
	return result
}

// 顺时针打印一个圈
func printCircle(matrix [][]int, start int) (result []int) {
	endR := len(matrix) - 1 - start
	endC := len(matrix[0]) - 1 - start
	// 从左到右打印一行
	for i := start; i < endC; i++ {
		result = append(result, matrix[start][i])
	}
	// 从上到下打印一列
	for i := start; i < endR; i++ {
		result = append(result, matrix[i][endC-start])
	}
	// 从右到左打印一行
	for i := endC - start; i > start; i-- {
		result = append(result, matrix[endR-start][i])
	}
	// 从下到上打印一列
	for i := endR - start; i > start; i-- {
		result = append(result, matrix[i][start])
	}
	return
}

// 剑指 Offer 32 - I. 从上到下打印二叉树
func levelOrder(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	result := make([]int, 0)
	// 使用队列去解决
	// 注意工程中不要自己去写一个简单的slice去代替，会造成内存泄露
	queer := list.New()
	queer.PushBack(root)
	for queer.Len() != 0 {
		if queer.Front().Value.(*TreeNode) != nil {
			// 子元素入队
			if queer.Front().Value.(*TreeNode).Left != nil {
				queer.PushBack(queer.Front().Value.(*TreeNode).Left)
			}
			if queer.Front().Value.(*TreeNode).Right != nil {
				queer.PushBack(queer.Front().Value.(*TreeNode).Right)
			}
		}
		result = append(result, queer.Front().Value.(*TreeNode).Val)
		// 出队
		queer.Remove(queer.Front())
	}
	return result
}

// 之型打印二叉树
func zhiLevelOrder(root *ListNode) [][]int {
	if root == nil {
		return nil
	}
	// 使用双向链表模拟堆栈or队列，两个装不同层的数据
	// 奇数层从左往右
	queer1 := list.New()
	// 偶数层从右往左
	queer2 := list.New()
	temp := make([][]int, 0)
	queer1.PushBack(root)
	flag := true
	for queer1.Len() != 0 || queer2.Len() != 0 {
		result := make([]int, 0)
		if flag {
			// 从左往右打印当层
			for queer1.Len() != 0 {
				result = append(result, queer1.Back().Value.(*TreeNode).Val)
				if queer1.Back().Value.(*TreeNode).Left != nil {
					queer2.PushBack(queer1.Back().Value.(*TreeNode).Left)
				}
				if queer1.Back().Value.(*TreeNode).Right != nil {
					queer2.PushBack(queer1.Back().Value.(*TreeNode).Right)
				}
				queer1.Remove(queer1.Back())
			}
		}
		if !flag {
			// 从右往左打印
			for queer2.Len() != 0 {
				result = append(result, queer2.Back().Value.(*TreeNode).Val)
				if queer2.Back().Value.(*TreeNode).Right != nil {
					queer1.PushBack(queer2.Back().Value.(*TreeNode).Right)
				}
				if queer2.Back().Value.(*TreeNode).Left != nil {
					queer1.PushBack(queer2.Back().Value.(*TreeNode).Left)
				}
				queer2.Remove(queer2.Back())
			}
		}
		flag = !flag
		temp = append(temp, result)
	}
	return temp
}

// 剑指 Offer 33. 二叉搜索树的后序遍历序列
func verifyPostorder(postorder []int) bool {
	if postorder == nil || len(postorder) == 0 {
		return true
	}
	// 根节点
	root := postorder[len(postorder)-1]
	// 左子树的值小于根节点
	i := 0
	for ; i < len(postorder)-1; i++ {
		if postorder[i] > root {
			break
		}
	}
	// 右子树的值大于根节点，根据二叉搜索树的特性，末尾一定是根节点
	// i，j双指针
	j := i
	for ; j < len(postorder)-1; j++ {
		if postorder[j] < root {
			return false
		}
	}
	left := true
	if i > 0 {
		left = verifyPostorder(postorder[:i])
	}
	right := true
	if i < len(postorder)-1 {
		right = verifyPostorder(postorder[i:j])
	}
	return right && left
}

// 剑指 Offer 34. 二叉树中和为某一值的路径
func pathSum(root *TreeNode, target int) [][]int {
	if root == nil {
		return nil
	}
	result := make([][]int, 0)
	tmep := make([]int, 0)
	FindPath(root, target, &tmep, &result)
	return result
}

// FindPath 前序遍历
func FindPath(root *TreeNode, target int, result *[]int, tmp *[][]int) {
	if root == nil {
		return
	}
	temp := append(*result, root.Val)
	// 遍历到叶子节点
	if root.Right == nil && root.Left == nil {
		if target == root.Val {
			result = &temp
			//TODO 想办法保存路径
			fmt.Println(result)
			k := append(*tmp, *result)
			tmp = &k
			return
		}
		if target != root.Val {
			return
		}
	}
	result = &temp
	FindPath(root.Left, target-root.Val, result, tmp)
	FindPath(root.Right, target-root.Val, result, tmp)
}

// 剑指 Offer 38. 字符串的排列
func permutation(s string) []string {
	// result:=make([]string,0)
	return nil
}

func main() {
	//arr := [][]byte{
	//	{'C', 'A', 'A'},
	//	{'A', 'A', 'A'},
	//	{'B', 'C', 'D'},
	//}

	//print(movingCount(7, 2, 3))

	//print(cuttingRope(120))

	//print(hammingWeight(9))

	// print(myPow(2, 10))

	// fmt.Printf("%v", spiralOrder([][]int{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}))

	// fmt.Printf("%v\n", verifyPostorder([]int{4, 8, 6, 12, 16, 14, 10}))
	tree := TreeNode{
		Val: 5,
		Left: &TreeNode{
			Val: 4,
			Left: &TreeNode{
				Val: 11,
				Left: &TreeNode{
					Val:   7,
					Left:  nil,
					Right: nil,
				},
				Right: &TreeNode{
					Val: 2, Left: nil, Right: nil,
				},
			},
			Right: nil,
		},
		Right: &TreeNode{
			Val:   8,
			Right: &TreeNode{Val: 13, Right: nil, Left: nil},
			Left: &TreeNode{
				Val:   4,
				Right: &TreeNode{Val: 5, Right: nil, Left: nil},
				Left:  &TreeNode{Val: 1, Right: nil, Left: nil},
			},
		},
	}
	fmt.Println(pathSum(&tree, 22))

	// fmt.Printf("the tree is :%v\n", levelOrder())

	// print(exist(arr, "AAB"))
}
