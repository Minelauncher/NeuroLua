# NeuroLua
루아로 만든 간단한 신경망 코드입니다.

루아로만 이루어진 신경망 구현이 필요해서 개인적으로 간단하게나마 제작했습니다.

파일 하나로 합칠 것을 염두에 두고 제작하여 많이 분할해두지 않았습니다.

파라미터 수백개 수준에서는 동작시킬만 합니다.

아직 수정할 여지 크기 때문에 지속적으로 수정중

## 사용 예

### 입력 데이터 정의
MNIST 데이터셋 로드
```Lua
local shape, trainImage = Loader.readIDX("MNIST/train-images.idx3-ubyte", true)
print("이미지 수:", shape[1])     -- 60000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 28 x 28

local shape, testImage = Loader.readIDX("MNIST/t10k-images.idx3-ubyte", true)
print("이미지 수:", shape[1])     -- 10000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 28 x 28

local shape, trainLabel = Loader.readIDX("MNIST/train-labels.idx1-ubyte", true)
print("이미지 수:", shape[1])     -- 60000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 0 x 0

local shape, testLabel = Loader.readIDX("MNIST/t10k-labels.idx1-ubyte", true)
print("이미지 수:", shape[1])     -- 10000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 0 x 0
```
### 모델 정의
간단한 CNN 모델
```Lua
local cnn = Model('CNN', {28,28}, {10})
-- 합성곱: 특성맵, 입력크기, 필터크기, 필터개수, 패딩, 스트라이드, 활성화함수
-- 풀링: 특성맵, 입력크기, 풀링크기, 패딩, 스트라이드
-- 연결: 입력크기, 출력크기, 활성화함수, 레이어 정규화 여부
cnn.layer.convolution(1, {28,28}, {3,3}, 8, 1, 1, 'ReLU')
cnn.layer.pooling(8, {28,28}, {2,2}, 0, 2)
cnn.layer.convolution(8, {14,14}, {3,3}, 8, 1, 1, 'ReLU')
cnn.layer.pooling(8, {14,14}, {2,2}, 0, 2)
cnn.layer.dense({8,7,7}, {10}, 'SoftMax', false)
```
### 학습
데이터가 존재한다면 불러오기 후 학습, 그리고 저장
```Lua
-- 처음 학습시에는 주석처리하고 사용
cnn:load("CNNdata.lua")

local start_time = os.time()
for i = 1, 1 do -- 200 온라인 학습 예시
    local batchSize = 5
    for j = 1, 100 do
        local miniBatch = math.random(1, 60000-batchSize)
        local miniTrainImage = slice(trainImage, miniBatch, miniBatch+batchSize-1)
        local miniTrainLabel = slice(trainLabel, miniBatch, miniBatch+batchSize-1)
        local oneHotEncodingMiniTrainLabel = {}
        for index = 1, batchSize do
            table.insert(oneHotEncodingMiniTrainLabel, oneHotEncoding(miniTrainLabel[index], 10))
        end
        local trainImageTensor = Tensor(miniTrainImage)/255
        local trainLabelTensor = Tensor(oneHotEncodingMiniTrainLabel)

        local error = cnn:learn(trainImageTensor, trainLabelTensor, 'CrossEntropy', 0.001)
        os.execute("cls")
        print("Error:", error)
        print(j, "-batch-")
        print(i, "-epoch-")
    end
end
local end_time = os.time()
local elapsed_time = end_time - start_time
print("Spend Time:", elapsed_time, "Second")

cnn:save("CNNdata.lua")
```
### 검증
다른 데이터로 학습 확인 후 신경망 정보 요약 출력
```Lua
-- evaluation
do
  local total   = 100--#testLabel      -- 테스트 샘플 개수
  local correct = 0

  for i = 1, total do
    -- 1. 한 장씩 꺼내 Tensor 형태로 변환 (batch size=1)
    local imgTensor    = Tensor({ testImage[i] }):detach() / 255
    -- 2. 순전파로 출력 확률 얻기
    local outputTensor = cnn:forwardPropagation(imgTensor)
    -- outputTensor.values[1] 은 Node 객체 배열 (크기 10)
    local probs        = outputTensor.values[1]

    -- 3. 최대 확률인 클래스 인덱스 찾기 (Lua 인덱스 1~10 → 레이블 0~9)
    local maxP, pred = -math.huge, nil
    for j, node in pairs(probs) do
      local p = node.value.real
      if p > maxP then
        maxP, pred = p, j - 1
      end
    end

    -- 4. 예측이 실제 라벨과 일치하면 correct 증가
    if pred == testLabel[i] then
      correct = correct + 1
    end
  end

  -- 5. 결과 출력
  local accuracy = correct / total * 100
  print(string.format("Test Accuracy: %d/%d (%.2f%%)", correct, total, accuracy))
end

cnn:summary()
```

96%의 분류성능을 보임.
![image](https://github.com/user-attachments/assets/96a16f21-cb75-466f-b0b2-f4e6ba547ec3)

