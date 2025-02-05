# NeuroLua
순수한 루아로 만든 간단한 신경망 코드입니다.

루아로만 이루어진 신경망 구현이 필요해서 개인적으로 간단하게나마 제작했습니다.

파일 하나로 합칠 것을 염두에 두고 제작하였습니다.

간단한 사용 예)

```
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

local inputTensor01 = Tensor.stack(inputTensor0, inputTensor1)
local targetTensor01 = Tensor.stack(targetTensor0, targetTensor1)
cnn = Model('CNN', {5,5}, {3})
-- 합성곱: 특성맵, 입력크기, 필터크기, 필터개수, 패딩, 스트라이드, 활성화함수
-- 풀링: 특성맵, 입력크기, 풀링크기, 패딩, 스트라이드
-- 연결: 입력크기, 출력크기, 활성화함수, 레이어 정규화 여부
cnn.layer.convolution(0, {5,5}, {3,3}, 4, 1, 1, 'ReLU')
cnn.layer.pooling(4, {5,5}, {2,2}, 0, 2)
cnn.layer.dense({4,2,2}, {3}, 'SoftMax', false)
--
cnn:load("CNNdata.lua")
for i = 1, 500 do -- 200 온라인 학습 예시
    --Sleep(100)
    local start_time = os.time()
    local error1 = cnn:learn(inputTensor01, targetTensor01, 'CrossEntropy', 0.01)
    os.execute("cls")
    print(error1)
    local end_time = os.time()
    local elapsed_time = end_time - start_time
    print("Spend Time:", elapsed_time, "Second")
    print(i, "epoch")
end
cnn:save("CNNdata.lua")
print(cnn:forwardPropagation(inputTensor01))
--
local inputTensor3 = Tensor({
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0}})
print(cnn:forwardPropagation(inputTensor3))
cnn:summary()
```
