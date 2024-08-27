local Model = require("NeuroLua")

math.randomseed(os.time())

local inputTensor0 = Tensor({{1,1,1},{1,1,1},{1,1,1}})
local targetTensor0 = Tensor({1,1,1})

nn = Model('CNN')
nn.layer.convolution({3,3},{3,3},3,1,1)
nn.layer.pooling(1, {3,3}, {2,2})
nn.layer.dense({1,2,2}, {3}, 'ReLU')
nn.layer.dense({3}, {3}, 'Linear')

nn.data:add(inputTensor0, targetTensor0)

for i = 1, 200 do
    print(nn:learn(inputTensor0, targetTensor0, 'MSE', 0.01))
end
print(nn:forwardPropagation(inputTensor0))
nn:summary()
