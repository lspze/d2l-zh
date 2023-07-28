from testllib import *

def data_elec(r, num_examples):
    '''生成电流电压数据'''
    # 电路采用均值分布
    I = torch.rand((num_examples, 1))
    U = torch.matmul(I,r)
    U += torch.normal(0, 0.1, U.shape)
    return I, U.reshape((-1,1))

def data_elec_iter(batch_size, features, labels):
    '''读取数据集并随机分batch'''
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        
def linreg_elec(I, r):
    '''线性回归模型：欧姆定律模型'''
    return torch.matmul(I ,r)

def loss_elec(U_predict, U_true):
    '''交叉熵损失'''
    return 0.5/len(U_true)*(U_predict-U_true.reshape(U_predict.shape)).norm()

def sgd_elec(param, lr):
    '''随机梯度下降优化算法'''
    with torch.no_grad():
        param -= lr*param.grad
        param.grad.zero_()
        

true_r = torch.tensor([3.5])
features, labels = data_elec(true_r, 3000)

batch_size = 30
learning_rate = 0.1
num_epochs = 10
net = linreg_elec
loss = loss_elec
optimizer = sgd_elec

r = torch.normal(0, 0.01, size=true_r.shape, requires_grad=True)
print(r)

print(f'epoch {0}, loss {float(loss(net(features, r), labels)):f}')
for epoch in range(num_epochs):
    for I, U in data_elec_iter(batch_size, features, labels):
        U_predict = net(I, r)
        l = loss(U_predict, U)
        l.backward()
        optimizer(r, learning_rate)
        with torch.no_grad():
            train_l = loss(net(features, r), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l):f}')
            
print(f'r的估计误差：{true_r - r.reshape(true_r.shape)}')