package go_design

import "testing"

func TestCombination(t *testing.T) {
	name := "yang"
	a := A{}
	a.Hello(name)

	b := B{&A{}}
	b.Hello(name)
	// 此时调用的是A方法，因为前面声明的时候指定了其接口的具体实现
	b.IHello.Hello(name)

	c := C{&A{}}
	c.Hello(name)

	// 改变c的组合方法
	c.IHello = &D{}
	c.Hello(name)
}
