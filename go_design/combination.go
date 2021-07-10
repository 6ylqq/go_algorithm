package go_design

import (
	"fmt"
)

// IHello 接口
type IHello interface {
	Hello(name string)
}

type A struct {
}

// Hello 隐式接口
func (*A) Hello(name string) {
	fmt.Println("hello" + name + ",i am a")
}

type D struct {
}

func (*D) Hello(name string) {
	fmt.Println("hello" + name + ",i am d")
}

type B struct {
	IHello
}

func (*B) Hello(name string) {
	fmt.Println("hello" + name + ",i am b")
}

type C struct {
	IHello
}
