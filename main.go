package main

import (
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
	for i, i2 := range p {
		switch i2 {
		case '.':
			continue
		case '*':

		}
	}
	return false
}

// 剑指 Offer 20. 表示数值的字符串
func isNumber(s string) bool {

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

	fmt.Printf("%v", printNumbers(1))

	// print(exist(arr, "AAB"))
}
