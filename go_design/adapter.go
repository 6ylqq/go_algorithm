package go_design

type Target interface {
	Request() string
}

type Adaptee interface {
	SpecificRequest() string
}

// 被适配的目标类
type adapteeImpl struct{}

// Request 该目标实现类实现的两个接口
func (a *adapteeImpl) Request() string {
	panic("implement me")
}

func (a *adapteeImpl) SpecificRequest() string {
	panic("implement me")
}

func NewAdaptee() Adaptee {
	return &adapteeImpl{}
}

func NewAdapter(adaptee Adaptee) Target {
	return &adapter{adaptee}
}

// 适配器
type adapter struct {
	Adaptee
}

func (a *adapter) Request() string {
	panic("implement me")
}
