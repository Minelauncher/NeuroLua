local Model = require("NeuroLua")

math.randomseed(os.time())

-- 학습에 사용할 임시 정보
local inputTensor0 = Tensor({{1,1,1},{1,1,1},{1,1,1}})
local targetTensor0 = Tensor({1,1,1})

-- 모델 구현
nn = Model('CNN')
nn.layer.convolution({3,3},{3,3},3,1,1)
nn.layer.pooling(1, {3,3}, {2,2})
nn.layer.dense({1,2,2}, {3}, 'ReLU')
nn.layer.dense({3}, {3}, 'Linear')

nn.data:add(inputTensor0, targetTensor0)

for i = 1, 200 do -- 200 에포크 온라인 학습 예시
    print(nn:learn(inputTensor0, targetTensor0, 'MSE', 0.01))
end
print(nn:forwardPropagation(inputTensor0))-- 결과 확인
nn:summary()-- 신경망 전체적인 정보 출력
