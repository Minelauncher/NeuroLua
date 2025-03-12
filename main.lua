local Model = require("NeuroLua")

math.randomseed(os.time())
local inputTensor0 = Tensor({
    {0,0,1,0,0},
    {0,0,1,0,0},
    {0,0,1,0,0},
    {0,0,1,0,0},
    {0,0,1,0,0}})
local targetTensor0 = Tensor({1,0,0})
local inputTensor1 = Tensor({
    {0,1,1,1,0},
    {1,0,0,0,1},
    {0,0,0,1,0},
    {0,1,1,0,0},
    {1,1,1,1,1}})
local targetTensor1 = Tensor({0,1,0})
local inputTensor2 = Tensor({
    {0,1,1,1,0},
    {1,0,0,0,1},
    {0,0,1,1,0},
    {1,0,0,0,1},
    {0,1,1,1,0}})
local targetTensor2 = Tensor({0,0,1})

local inputTensor012 = Tensor.stack(inputTensor0, inputTensor1, inputTensor2)
local targetTensor012 = Tensor.stack(targetTensor0, targetTensor1, targetTensor2)
cnn = Model('CNN', {5,5}, {3})
-- 합성곱: 특성맵, 입력크기, 필터크기, 필터개수, 패딩, 스트라이드, 활성화함수
-- 풀링: 특성맵, 입력크기, 풀링크기, 패딩, 스트라이드
-- 연결: 입력크기, 출력크기, 활성화함수, 레이어 정규화 여부
cnn.layer.convolution(1, {5,5}, {3,3}, 4, 1, 1, 'ReLU')
cnn.layer.convolution(4, {5,5}, {3,3}, 4, 1, 1, 'ReLU')
cnn.layer.pooling(4, {5,5}, {2,2}, 0, 2)
cnn.layer.dense({4,2,2}, {3}, 'SoftMax', false)
--
cnn:load("CNNdata.lua")
for i = 1, 500 do -- 500 온라인 학습 예시
    --Sleep(100)
    local start_time = os.time()
    local error1 = cnn:learn(inputTensor012, targetTensor012, 'CrossEntropy', 0.01)
    os.execute("cls")
    print(error1)
    local end_time = os.time()
    local elapsed_time = end_time - start_time
    print("Spend Time:", elapsed_time, "Second")
    print(i, "epoch")
end
cnn:save("CNNdata.lua")
print(cnn:forwardPropagation(inputTensor012))
--
local inputTensor3 = Tensor({
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0}})
print(cnn:forwardPropagation(inputTensor3))
cnn:summary()


local test = Tensor({
    {1,2,3,4},
    {2,3,4,5},
    {3,4,5,6},
    {4,5,6,7}})
local testTarget = Tensor({{1}})

local testTable = {}
for i = 1, 3 do
    table.insert(testTable, test)
end
-- 원래는 포지셔널 인코딩도 해야함
testTable = Tensor.stack(unpack(testTable))

local attention = Model('AT', {3,4,4}, {1})
attention.layer.attention(2, {4,4}, {4,4}, {4,4})
attention.layer.dense({4,4}, {1}, 'Linear', false)
--
attention:load("AttentionData.lua")
for i = 1, 500 do -- 200 온라인 학습 예시
    --Sleep(100)
    local start_time = os.time()
    local error1 = attention:learn(testTable, testTarget, 'MSE', 0.0001)
    os.execute("cls")
    print(error1)
    local end_time = os.time()
    local elapsed_time = end_time - start_time
    print("Spend Time:", elapsed_time, "Second")
    print(i, "epoch")
end
attention:save("AttentionData.lua")

print(attention:forwardPropagation(testTable))
