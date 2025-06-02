local Model = require("NeuroLua")
local memWatch  = require("memWatch")
local Loader = require("mnist_loader")

math.randomseed(os.time())

local shape, trainImage = Loader.readIDX("MNIST/train-images.idx3-ubyte", true)
--print("이미지 수:", shape[1])     -- 60000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 28 x 28

local shape, testImage = Loader.readIDX("MNIST/t10k-images.idx3-ubyte", true)
--print("이미지 수:", shape[1])     -- 10000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 28 x 28

local shape, trainLabel = Loader.readIDX("MNIST/train-labels.idx1-ubyte", true)
--print("이미지 수:", shape[1])     -- 60000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 0 x 0

local shape, testLabel = Loader.readIDX("MNIST/t10k-labels.idx1-ubyte", true)
--print("이미지 수:", shape[1])     -- 10000
--print("이미지 크기:", shape[2], "x", shape[3])  -- 0 x 0

local mlp = Model('MLP', {28,28}, {10})
-- 합성곱: 특성맵, 입력크기, 필터크기, 필터개수, 패딩, 스트라이드, 활성화함수
-- 풀링: 특성맵, 입력크기, 풀링크기, 패딩, 스트라이드
-- 연결: 입력크기, 출력크기, 활성화함수, 레이어 정규화 여부
mlp.layer.dense({28,28}, {100}, 'ReLU', false)
mlp.layer.dense({100}, {10}, 'SoftMax', false)

mlp:load("MLPdata.lua")

local function slice(table, first, last)
  -- unpack(t, i, j)는 t[i]부터 t[j]까지 값을 반환
  -- j를 생략하면 t[#t]까지 반환하므로 or #t 처리
  return { unpack(table, first, last or #table) }
end

local function oneHotEncoding(label, length)
    local function round(x)
        if x >= 0 then
            return math.floor(x + 0.5)
        else
            return math.ceil(x - 0.5)
        end
    end
    local encodingVector = {}
    for i = 1, length do
        table.insert(encodingVector, (i-1) == round(label) and 1 or 0)
    end
    return encodingVector
end

local start_time = os.time()
for i = 1, 1 do -- 200 온라인 학습 예시
    local batchSize = 10
    for j = 1, 1000/batchSize do
        local miniTrainImage = slice(trainImage, (j-1)*batchSize+1, (j)*batchSize)
        local miniTrainLabel = slice(trainLabel, (j-1)*batchSize+1, (j)*batchSize)
        local oneHotEncodingMiniTrainLabel = {}
        for index = 1, batchSize do
            table.insert(oneHotEncodingMiniTrainLabel, oneHotEncoding(miniTrainLabel[index], 10))
        end
        local trainImageTensor = Tensor(miniTrainImage)/255
        local trainLabelTensor = Tensor(oneHotEncodingMiniTrainLabel)

        local error = mlp:learn(trainImageTensor, trainLabelTensor, 'CrossEntropy', 0.01)
        os.execute("cls")
        print(error)
        print(j, "batch")
        print(i, "epoch")
    end
end
local end_time = os.time()
local elapsed_time = end_time - start_time
print("Spend Time:", elapsed_time, "Second")

mlp:save("MLPdata.lua")

do
  local total   = 100--#testLabel      -- 테스트 샘플 개수
  local correct = 0

  for i = 1, total do
    -- 1. 한 장씩 꺼내 Tensor 형태로 변환 (batch size=1)
    local imgTensor    = Tensor({ testImage[i] }) / 255
    -- 2. 순전파로 출력 확률 얻기
    local outputTensor = mlp:forwardPropagation(imgTensor)
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

mlp:summary()