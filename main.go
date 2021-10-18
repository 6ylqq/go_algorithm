package main

import (
	"container/list"
	"fmt"
	"math"
	"sort"
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
var tmp []list.List

func pathSum(root *TreeNode, target int) [][]int {
	if root == nil {
		return nil
	}
	result := list.New()
	FindPath(root, target, result)
	// fmt.Println(tmp)
	return nil
}

// FindPath 前序遍历
func FindPath(root *TreeNode, target int, result *list.List) {
	if root == nil {
		return
	}
	result.PushBack(root.Val)
	// 遍历到叶子节点
	if root.Right == nil && root.Left == nil {
		if target == root.Val {
			//TODO 想办法保存路径
			tmp = append(tmp, *result)
			return
		}
		if target != root.Val {
			return
		}
	}
	FindPath(root.Left, target-root.Val, result)
	FindPath(root.Right, target-root.Val, result)
	result.Remove(result.Back())
	return
}

// 剑指 Offer 38. 字符串的排列
var res []string

func permutation(s string) []string {
	if s == "" {
		return nil
	}
	sortS([]byte(s), 0)
	// 去重
	hm := make(map[string]bool)
	p := make([]string, 0)
	for _, i := range res {
		if _, ok := hm[i]; !ok {
			p = append(p, i)
			hm[i] = true
		}
	}
	return p
}

func sortS(s []byte, begin int) {
	if begin == len(s)-1 {
		fmt.Println(string(s))
		res = append(res, string(s))
		return
	}
	for i := begin; i < len(s); i++ {
		temp := s[i]
		// 交换
		s[i] = s[begin]
		s[begin] = temp

		sortS(s, begin+1)

		temp = s[i]
		s[i] = s[begin]
		s[begin] = temp
	}
	return
}

func majorityElement(nums []int) int {
	m := make(map[int]int)
	max := 0
	result := 0
	for _, num := range nums {
		m[num]++
		if max < m[num] {
			max = m[num]
			result = num
		}
	}
	return result
}

func getLeastNumbers(arr []int, k int) []int {
	// 快速选择算法
	if k == 0 {
		return []int{0}
	}
	if len(arr) < k {
		return arr
	}
	return nil
}

func PartitionArray(arr []int, low int, high int, k int) {
	// m := Partition(arr, low, high)
	return
}

func Partition(arr []int, low int, high int) int {
	// 快排分治
	// 先随便选一个基准数
	index := high
	for i := 0; i < high; i++ {
		if arr[i] < arr[index] {

		}
	}
	return 0
}

func Swap(arr *[]int, i int, j int) {
	return
}

// 剑指 Offer 42. 连续子数组的最大和
func maxSubArray(nums []int) int {
	if nums == nil {
		return 0
	}
	// 简单动规
	result := nums[0]
	max := nums[0]
	for i := 0; i < len(nums); i++ {
		if result < 0 {
			result = nums[i]
			if result > max {
				max = result
			}
		} else if i != 0 {
			result += nums[i]
			if result > max {
				max = result
			}
		}
	}
	if result > max {
		max = result
	}
	return max
}

// 剑指 Offer 43. 1～n 整数中 1 出现的次数
func countDigitOne(n int) int {
	result := 0
	for i := 1; i <= n; i++ {
		result += getOne(i)
	}
	return result
}

func getOne(n int) int {
	result := 0
	for n != 0 {
		if n%10 == 1 {
			result++
		}
		n /= 10
	}
	return result
}

// 剑指 Offer 45. 把数组排成最小的数
func minNumber(nums []int) string {
	result := ""
	return result
}

// 最长回文字串
func longestPalindrome(s string) string {
	maxLength := 1
	maxS := string(s[0])
	for i := 0; i < len(s); i++ {
		for j := i + 1; j <= len(s); j++ {
			if isPalindrome(s[i:j]) && j-i+1 > maxLength {
				maxS = s[i:j]
				maxLength = j - i + 1
			}
		}
	}
	return maxS
}

func isPalindrome(s string) bool {
	return func() bool {
		i := 0
		j := len(s) - 1
		for i != j {
			if i == j-1 {
				return s[i] == s[j]
			}
			if s[i] == s[j] {
				i++
				j--
				continue
			}
			return false
		}
		return true
	}()
}

// 6. Z 字形变换
func convert(s string, numRows int) string {
	if numRows < 2 {
		return s
	}
	result := make([]string, numRows)
	flag := -1
	i := 0
	for _, s1 := range s {
		result[i] += string(s1)
		if i == 0 || i == numRows-1 {
			// 遇到转折点，则反向-1
			flag = -flag
		}
		i += flag
	}
	data := func() string {
		temp := ""
		for _, s2 := range result {
			temp += s2
		}
		return temp
	}()
	return data
}

func reverse(x int) (t int) {
	defer func() {
		if t > 2147483647-1 || -t > 2147483647-1 {
			t = 0
		}
		err := recover()
		if err != nil {
			t = 0
			fmt.Println(err)
		}
	}()
	result := func(x int, t int) int {
		for x != 0 {
			temp := x % 10
			x /= 10
			t = t*10 + temp
		}
		return t
	}(x, t)
	return result
}

func isPalindrome2(x int) bool {
	result := 0
	for x != 0 {
		temp := x % 10
		x /= 10
		result = result*10 + temp
		if result == x {
			return true
		}
		if result > x {
			result /= 10
			return result == x
		}
	}
	return false
}

// 11. 盛最多水的容器，移动的过程中不断消去不可能成为最大值的情况
func maxArea(height []int) int {
	i := 0
	j := len(height) - 1
	res1 := 0
	for i < j {
		if height[i] < height[j] {
			res1 = int(math.Max(float64(res1), float64(height[i]*(j-i))))
			i += 1
			continue
		}
		res1 = int(math.Max(float64(res1), float64(height[j]*(j-i))))
		j -= 1
	}
	return res1
}

func nextPermutation(nums []int) {
	if len(nums) == 1 || len(nums) == 0 {
		return
	}
	// 第一遍扫描，找到“较小数”，较小数需要尽量靠右
	little := func() int {
		i := len(nums) - 1
		for nums[i] <= nums[i-1] && i-1 > 0 {
			i--
		}
		return i - 1
	}()
	big := func() int {
		i := len(nums) - 1
		for nums[i] <= nums[little] && i > little {
			i--
		}
		return i
	}()
	if little == 0 && big == 0 {
		reverser(nums)
		return
	}
	temp := nums[little]
	nums[little] = nums[big]
	nums[big] = temp
	reverser(nums[little+1:])
}

// 反转数组
func reverser(a []int) {
	// 双指针
	for i, n := 0, len(a); i < n/2; i++ {
		a[i], a[n-1-i] = a[n-1-i], a[i]
	}
}

// 26. 删除有序数组中的重复项，切入点：数组有序
func removeDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 {
		return 1
	}
	i := 0
	j := i + 1
	for j < len(nums) {
		if nums[i] != nums[j] {
			t := nums[i+1]
			nums[i+1] = nums[j]
			nums[j] = t
			i++
		}
		j++
	}
	return i
}

func removeElement(nums []int, val int) int {
	if len(nums) == 0 {
		return 0
	}
	if len(nums) == 1 && nums[0] == val {
		return 0
	}
	i := 0
	j := i
	for j < len(nums) {
		if nums[j] != val {
			t := nums[i]
			nums[i] = nums[j]
			nums[j] = t
			i++
		}
		j++
	}
	return i
}

func strStr(haystack string, needle string) int {
	if needle == "" {
		return 0
	}
	if haystack == "" {
		return -1
	}
	i := 0
	j := 0
	for i < len(haystack) && j < len(needle) {
		if haystack[i] == needle[j] {
			j++
			i++
		} else {
			i -= j
			i++
			j = 0
		}
	}
	if j == len(needle) {
		return i - len(needle)
	}
	return -1
}

func divide(dividend int, divisor int) int {
	if dividend == 0 {
		return 0
	}
	// 都转成正数运算
	flag := func() int {
		// 除数与被除数异号
		if (dividend < 0 && divisor > 0) || (dividend > 0 && divisor < 0) {
			return -1
		}
		return 1
	}()
	a := func() int {
		if dividend < 0 {
			return 0 - dividend
		}
		return dividend
	}()
	b := func() int {
		if divisor < 0 {
			return -divisor
		}
		return divisor
	}()
	result := div(a, b)
	if result >= int(math.Pow(float64(2), float64(31))) && flag > 0 {
		result = int(math.Pow(float64(2), float64(31))) - 1
	}
	if flag > 0 {
		// 判断是不是大于32位有符号整数
		return result
	}
	return -result
}

// 翻倍
func div(dividend int, divisor int) int {
	if dividend < divisor {
		return 0
	}
	count := 1
	temp := divisor
	for temp+temp <= dividend {
		// 商翻倍
		count += count
		// 除数也翻倍
		temp += temp
	}
	return count + div(dividend-temp, divisor)
}

// 33 搜索旋转排序数组
func search(nums []int, target int) int {
	if len(nums) == 1 && nums[0] != target {
		return -1
	}
	if len(nums) == 1 && nums[0] == target {
		return nums[0]
	}
	// 变形的二分
	low, heigh := 0, len(nums)-1
	for low+1 != heigh {
		mid := (low + heigh) / 2
		// 如果左边有序，并且target在内
		if nums[mid] > nums[low] && nums[low] <= target && nums[mid] >= target {
			heigh = mid
			continue
		}
		// 左边有序，但target不在内
		if nums[mid] > nums[low] && !(nums[low] <= target && nums[mid] >= target) {
			low = mid
			continue
		}
		// 如果右边有序，并且target在内
		if nums[mid] < nums[heigh] && nums[mid] <= target && nums[heigh] >= target {
			low = mid
			continue
		}
		// 右边有序，但target不在内
		if nums[mid] < nums[heigh] && !(nums[mid] <= target && nums[heigh] >= target) {
			heigh = mid
			continue
		}
	}
	if nums[heigh] != target && nums[low] != target {
		return -1
	}
	if nums[heigh] == target {
		return heigh
	}
	return low
}

// 34. 在排序数组中查找元素的第一个和最后一个位置，进阶让时间复杂度为logn
func searchRange(nums []int, target int) []int {
	if len(nums) == 0 {
		return []int{-1, -1}
	}
	i, j := len(nums)-1, 0
	for key, num := range nums {
		if num == target && i >= key {
			i = key
		}
		if num == target && j <= key {
			j = key
		}
	}
	if i == len(nums)-1 && j == 0 && nums[0] != target {
		return []int{-1, -1}
	}
	return []int{i, j}
}

// 35. 搜索插入位置
func searchInsert(nums []int, target int) int {
	low, heigh := 0, len(nums)-1
	result := len(nums)
	for low <= heigh {
		mid := (low + heigh) / 2
		if target <= nums[mid] {
			result = mid
			heigh = mid - 1
			continue
		}
		low = mid + 1
	}
	return result
}

// 39. 组合总和，类似决策树
func combinationSum(candidates []int, target int) (result [][]int) {
	// 深搜
	var temp []int
	var dfs func(target int, index int)
	dfs = func(target int, index int) {
		// 当candidate数组用完
		if index == len(candidates) {
			return
		}
		// 将该次的组合加入结果中 回溯关键
		if target == 0 {
			result = append(result, append([]int(nil), temp...))
			return
		}
		// 不用当前的这个数
		dfs(target, index+1)
		// 使用当前数，但必须要求加上当前数后，target为正
		if target-candidates[index] >= 0 {
			// 加入到临时数组
			temp = append(temp, candidates[index])
			dfs(target-candidates[index], index)
			// dfs后，需要弹出最后一个元素
			temp = temp[:len(temp)-1]
		}
	}
	dfs(target, 0)
	return
}

// 40. 组合总和 II，要求每个数字在组合中只能使用一次
func combinationSum2(candidates []int, target int) [][]int {
	return nil
}

// 43. 字符串相乘
func multiply(num1 string, num2 string) (res string) {
	if num1 == "0" || num2 == "0" {
		return "0"
	}
	// 定义字符串加法
	var add func(n string, m string) (result string)
	add = func(n string, m string) (result string) {
		if n == "" {
			return m
		}
		//ad存储进位
		i, j, ad := len(n)-1, len(m)-1, 0
		for i >= 0 || j >= 0 || ad != 0 {
			a, b := 0, 0
			if i >= 0 {
				a = int(n[i] - '0')
			}
			if j >= 0 {
				b = int(m[j] - '0')
			}
			temp := a + b + ad
			result = strconv.Itoa(temp%10) + result
			ad = temp / 10
			i--
			j--
		}
		return
	}
	// 遍历乘数
	for i := len(num2) - 1; i >= 0; i-- {
		manyZero := func(t int) string {
			result := ""
			for t < len(num2)-1 {
				result += "0"
				t++
			}
			return result
		}(i)
		noZere := func() (result string) {
			for j := 0; j < int(num2[i]-'0'); j++ {
				// 相加n次
				result = add(result, num1)
			}
			return
		}()
		all := noZere + manyZero
		res = add(res, all)
	}
	return
}

// 46. 全排列
func permute(nums []int) (result [][]int) {
	// 回溯，排列，组合都用这
	// 标记数组
	vis := make(map[int]bool)
	isUse := func(num int, target map[int]bool) bool {
		_, ok := target[num]
		return ok
	}
	var backtrack func(output []int)
	backtrack = func(output []int) {
		if len(output) == len(nums) {
			result = append(result, append([]int{}, output...))
			return
		}
		for _, num := range nums {
			// 没有用过该数字
			if !isUse(num, vis) {
				output = append(output, num)
				vis[num] = true
				// 填入下一个元素
				backtrack(output)
				// 回溯后记得弹出原来的这个元素
				output = output[:len(output)-1]
				delete(vis, num)
			}
		}
	}
	backtrack([]int{})
	return
}

// 47. 全排列 II
func permuteUnique(nums []int) (result [][]int) {
	sort.Ints(nums)
	// 相比于普通的全排列，回溯的基础上，加上去重
	// 使用栈，模拟连续填入，因为回出现121这种情况，所以不能用map去做了
	vis := make([]bool, len(nums))
	var backtrack func(output []int)
	backtrack = func(output []int) {
		if len(output) == len(nums) {
			result = append(result, append([]int{}, output...))
			return
		}
		for i, num := range nums {
			// 没有用过该数字且不能出现重复的排列
			if !(vis[i] || i > 0 && !vis[i-1] && num == nums[i-1]) {
				output = append(output, num)
				vis[i] = true
				// 填入下一个元素
				backtrack(output)
				// 回溯后记得弹出原来的这个元素
				output = output[:len(output)-1]
				vis[i] = false
			}
		}
	}
	backtrack([]int{})
	return
}

// 48. 旋转图像
func rotate(matrix [][]int) {
	// 翻转的本质可以转为，左上右下对角线反转后，左右翻转一次
	for i, row := range matrix {
		for j, num := range row {
			if j == i {
				break
			}
			matrix[i][j] = matrix[j][i]
			matrix[j][i] = num
		}
	}
	for _, ints := range matrix {
		for k := 0; k <= (len(ints)-1)/2; k++ {
			temp := ints[k]
			ints[k] = ints[(len(ints)-1)-k]
			ints[(len(ints)-1)-k] = temp
		}
	}
	return
}

// 15. 三数之和
func threeSum(nums []int) (result [][]int) {
	if nums == nil || len(nums) == 0 {
		return
	}
	if len(nums) == 3 {
		if nums[0]+nums[1]+nums[2] == 0 {
			result = append(result, nums)
			return
		}
	}
	sort.Ints(nums)
	for i := 0; i < len(nums)-2; i++ {
		for j := i + 1; j < len(nums)-1; j++ {
			for k := j + 1; k < len(nums); k++ {
				if nums[i]+nums[j]+nums[k] == 0 {
					result = append(result, []int{nums[i], nums[j], nums[k]})
				}
			}
		}
	}
	// 去重
	m := make(map[string]bool)
	var t = make([][]int, 0)
	for _, ints := range result {
		if _, ok := m[fmt.Sprintf("%v", ints)]; !ok {
			t = append(t, ints)
			m[fmt.Sprintf("%v", ints)] = true
		}
	}
	result = t
	return
}

// 50. Pow(x, n) 快速幂
func myPow2(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return x
	}
	var quickPow func(i float64, k int) float64
	quickPow = func(i float64, k int) float64 {
		if k == 0 {
			return 1
		}
		y := quickPow(i, k/2)
		if k%2 == 0 {
			return y * y
		}
		return y * y * i
	}
	result := quickPow(x, int(math.Abs(float64(n))))
	if n < 0 {
		return 1.0 / result
	}
	return result
}

// 56. 合并区间
func merge(intervals [][]int) (result [][]int) {

	return
}

// 45. 跳跃游戏 II
func jump(nums []int) int {
	// 典型贪心
	addr := len(nums) - 1
	result := 0
	for addr > 0 {
		mini := addr
		// 找到最远能跳到当前step的位置
		for i := addr; i >= 0; i-- {
			if nums[i]+i >= addr && i <= mini {
				mini = i
			}
		}
		addr = mini
		result++
	}
	return result
}

// 94. 二叉树的中序遍历
func inorderTraversal(root *TreeNode) (result []int) {
	var inorder func(root *TreeNode)
	inorder = func(root *TreeNode) {
		if root == nil {
			return
		}
		inorder(root.Left)
		result = append(result, root.Val)
		inorder(root.Right)
	}
	inorder(root)
	return
}

// 92. 反转链表 II
func reverseBetween(head *ListNode, left int, right int) *ListNode {
	// 哨兵
	fNode := &ListNode{Val: -1}
	fNode.Next = head
	// left预留多一位
	leftNode := func(left int) *ListNode {
		temp := fNode
		for i := 0; i < left-1; i++ {
			temp = temp.Next
		}
		return temp
	}
	rightNode := func(right int) *ListNode {
		temp := fNode
		for i := 0; i < right; i++ {
			temp = temp.Next
		}
		return temp
	}
	var change func(head *ListNode, left *ListNode, right *ListNode)
	change = func(head *ListNode, left *ListNode, right *ListNode) {
		// 切断链表
		leftN := left.Next
		temp := right.Next
		left.Next = nil
		right.Next = nil

		var pre *ListNode
		cur := leftN
		for cur != nil {
			next := cur.Next
			cur.Next = pre
			pre = cur
			cur = next
		}
		// 接回去
		left.Next = right
		leftN.Next = temp
	}
	change(fNode, leftNode(left), rightNode(right))
	return fNode.Next
}

// 88. 合并两个有序数组
func merge2(nums1 []int, m int, nums2 []int, n int) {
	result := make([]int, 0, m+n)
	for i, j := 0, 0; ; {
		if i == m {
			// 把第二个数组的元素都加到第一个中去
			result = append(result, nums2[j:]...)
			break
		}
		if j == n {
			result = append(result, nums1[i:]...)
			break
		}
		if nums1[i] < nums2[j] {
			result = append(result, nums1[i])
			i++
		} else {
			// 插入到nums1的i
			result = append(result, nums2[j])
			j++
		}
	}
	copy(nums1, result)
	return
}

func arraysToList(nums []int, head *ListNode) *ListNode {
	fnode := head
	for _, num := range nums {
		temp := ListNode{Val: num}
		head.Next = &temp
		head = &temp
	}
	return fnode.Next
}

func GetEndPoint(order string) []int {
	if order == "" {
		return []int{0, 0}
	}
	x := 0
	y := 0
	step := 0
	for i := 0; i <= len(order)-1; i++ {
		if order[i] <= '9' && order[i] >= '0' {
			if step != 0 {
				step = step*10 + int(order[i]-'0')
			} else {
				step += int(order[i] - '0')
			}
		}
		switch order[i] {
		case 'W':
			if step <= 0 {
				step = 1
			}
			y += step
			step = 0
		case 'S':
			if step <= 0 {
				step = 1
			}
			y -= step
			step = 0
		case 'A':
			if step <= 0 {
				step = 1
			}
			x -= step
			step = 0
		case 'D':
			if step <= 0 {
				step = 1
			}
			x += step
			step = 0
		default:
			continue
		}
	}
	return []int{x, y}
	// write code here
}

func getValue(rowIndex int, columnIndex int) int {
	rowIndex--
	columnIndex--
	x := func() int {
		result := 1
		for i := 0; i < columnIndex; i++ {
			result = result * (rowIndex - i)
		}
		return result
	}()
	y := func() int {
		t := 1
		for i := columnIndex; i > 0; i-- {
			t *= i
		}
		return t
	}()
	return x / y
	// write code here
}

func findMinOverrideSubString(source string, target string) string {
	i := 0
	j := 0
	begin := len(source) - 1
	end := len(source) - 1
	for i <= len(source)-1 && j <= len(target)-1 {
		if source[i] == target[j] {
			if begin >= i {
				begin = i
			}
			if j == len(target)-1 {
				end = i
			}
			j++
		}
		i++
	}
	if j != len(target) {
		return ""
	}
	return source[begin : end+1]
	// write code here
}

func zipString(s string) (result string) {
	count := 0
	front := ""
	for _, i := range s {
		tp := string(i)
		if count == 0 {
			count++
			front = tp
			continue
		}
		if tp == front && count != 0 {
			count++
		}
		if tp != front {
			if count == 1 {
				result += front
			} else {
				result += strconv.Itoa(count) + front
			}
			front = tp
			count = 1
		}
	}
	if count == 1 {
		result += front
	} else {
		result += strconv.Itoa(count) + front
	}
	return
}

// 70. 爬楼梯
func climbStairs(n int) int {
	var rec func(m int) int
	rec = func(m int) int {
		if m == 1 {
			return 1
		}
		if m == 2 {
			return 2
		}
		return rec(m-1) + rec(m-2)
	}
	return rec(n)
}

// 102. 二叉树的层序遍历
func levelOrder2(root *TreeNode) (result [][]int) {
	if root == nil {
		return result
	}
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)
	for i := 0; len(queue) > 0; i++ {
		p := func() (res []int) {
			for _, node := range queue {
				res = append(res, node.Val)
			}
			return
		}()
		result = append(result, p)
		tmp := make([]*TreeNode, 0)
		for _, node := range queue {
			if node.Left != nil {
				tmp = append(tmp, node.Left)
			}
			if node.Right != nil {
				tmp = append(tmp, node.Right)
			}
		}
		queue = tmp
	}
	return
}

// 3. 无重复字符的最长子串
func lengthOfLongestSubstring(s string) int {
	if s == "" {
		return 0
	}
	m := make(map[uint8]int)
	m[s[0]] = 1
	i, j := 0, 0
	flag := func(mm map[uint8]int) bool {
		for _, i2 := range mm {
			if i2 > 1 {
				return true
			}
		}
		return false
	}
	for i < len(s)-1 && j < len(s)-1 {
		j++
		m[s[j]]++
		if flag(m) {
			// 如果重复，窗口向前移动
			m[s[i]]--
			i++
		}
	}
	return j - i + 1
}

// 53. 最大子序和
func maxSubArray2(nums []int) int {
	max := nums[0]
	pre := nums[0]
	for index, num := range nums {
		if index == 0 {
			continue
		}
		if pre >= 0 {
			pre += num
		} else {
			pre = num
		}
		max = int(math.Max(float64(pre), float64(max)))
	}
	return max
}

func main() {
	fmt.Println(lengthOfLongestSubstring(" "))

	//fmt.Println(climbStairs(4))

	// fmt.Println(getValue(3, 2))

	// fmt.Println(findMinOverrideSubString("abcd","bc"))

	// fmt.Println(GetEndPoint("W2D"))

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
	//tree := TreeNode{
	//	Val: 5,
	//	Left: &TreeNode{
	//		Val: 4,
	//		Left: &TreeNode{
	//			Val: 11,
	//			Left: &TreeNode{
	//				Val:   7,
	//				Left:  nil,
	//				Right: nil,
	//			},
	//			Right: &TreeNode{
	//				Val: 2, Left: nil, Right: nil,
	//			},
	//		},
	//		Right: nil,
	//	},
	//	Right: &TreeNode{
	//		Val:   8,
	//		Right: &TreeNode{Val: 13, Right: nil, Left: nil},
	//		Left: &TreeNode{
	//			Val:   4,
	//			Right: &TreeNode{Val: 1, Right: nil, Left: nil},
	//			Left:  &TreeNode{Val: 5, Right: nil, Left: nil},
	//		},
	//	},
	//}
	//fmt.Println(pathSum(&tree, 22))
	//
	//fmt.Println(permutation("aab"))

	// fmt.Println(majorityElement([]int{}))

	// fmt.Println(maxSubArray([]int{-1, 0, -2}))

	// fmt.Println(countDigitOne(12))

	// fmt.Println(isPalindrome("bab"))

	// fmt.Println(convert("PAYPALISHIRING", 3))

	//fmt.Println(reverse(1534236469))

	// fmt.Println(isPalindrome2(88888))

	// fmt.Println(maxArea([]int{1, 8, 6, 2, 5, 4, 8, 3, 7}))

	// fmt.Println(removeElement([]int{0, 1, 2, 2, 3, 0, 4, 2}, 2))

	// fmt.Println(divide(-2147483648, 1))

	//nums := []int{1, 2, 3}
	//nextPermutation(nums)
	//fmt.Println(nums)

	// fmt.Println(search([]int{4, 5, 6, 7, 0, 1, 2}, 1))

	// fmt.Println(multiply("9", "9"))

	// fmt.Println(jump([]int{2, 3, 1, 1, 4}))

	// fmt.Printf("the tree is :%v\n", levelOrder())

	// print(exist(arr, "AAB"))

	// fmt.Println(permuteUnique([]int{1, 1, 2}))

	// fmt.Println(myPow2(2, 10))

	// reverseBetween(arraysToList([]int{3, 5}, new(ListNode)), 1, 2)

	// println(merge([][]int{[]int{1, 2, 3, 0, 0, 0}, []int{2, 5, 6}}))

	// println(fmt.Sprintf("%s", zipString("xxxrrrasdasxxx")))

	// fmt.Println(threeSum([]int{0, 0, 0, 0}))
}
