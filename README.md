# NeuroLua
루아로 만든 간단한 신경망 코드입니다.

루아로만 이루어진 신경망 구현이 필요해서 개인적으로 간단하게나마 제작했습니다.

파일 하나로 합칠 것을 염두에 두고 제작하여 많이 분할해두지 않았습니다.

파라미터 수백~수천개 수준에서는 동작시킬만 합니다.

아직 수정할 여지 크기 때문에 지속적으로 수정중

## 사용 예

### 입력 데이터 정의
5 * 5 1과 2 숫자 모양 데이터
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
```
### 모델 정의
간단한 CNN 모델
```
-- 배치로 묶음
local inputTensor01 = Tensor.stack(inputTensor0, inputTensor1)
local targetTensor01 = Tensor.stack(targetTensor0, targetTensor1)

local Model = require('NeuroLua')
cnn = Model('CNN', {5,5}, {3})
-- 합성곱: 특성맵, 입력크기, 필터크기, 필터개수, 패딩, 스트라이드, 활성화함수
-- 풀링: 특성맵, 입력크기, 풀링크기, 패딩, 스트라이드
-- 연결: 입력크기, 출력크기, 활성화함수, 레이어 정규화 여부
cnn.layer.convolution(1, {5,5}, {3,3}, 4, 1, 1, 'ReLU')
cnn.layer.pooling(4, {5,5}, {2,2}, 0, 2)
cnn.layer.dense({4,2,2}, {3}, 'SoftMax', false)
```
### 학습
데이터가 존재한다면 불러오기 후 학습, 그리고 저장
```
cnn:load("CNNdata.lua") -- 기존 데이터가 있다면 사용
for i = 1, 500 do -- 500 에포크 학습
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
```
### 검증
다른 데이터로 학습 확인 후 신경망 정보 요약 출력
```
local inputTensor3 = Tensor({
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0},
    {0,1,0,0,0}})
print(cnn:forwardPropagation(inputTensor3))
cnn:summary()
```
