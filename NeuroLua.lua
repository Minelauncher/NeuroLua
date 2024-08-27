local Tensor = require("Tensor")



local Convolution = {}
Convolution.__index = Convolution

local Pooling = {}
Pooling.__index = Pooling

local Dense = {}
Dense.__index = Dense

local Data = {}
Data.__index = Data

local Model = {}
Model.__index = Model

function Convolution.new(inputShape, filterShape, filterNum, padding, stride)
    local self = setmetatable({}, Convolution)
    self.inputShape = inputShape -- table

    self.filterShape = filterShape -- table
    self.filterNum = filterNum -- number

    self.filter = {}
    for i = 1, filterNum do
        self.filter[i] = Tensor.apply( Tensor.emptyTensor(filterShape) , function (x) return x + (2*math.random()-1) end)
    end
    self.Momentum_dL_dF = initializeTable(filterNum, 0)
    self.RMSprop_dL_dF = initializeTable(filterNum, 0)

    self.stride = stride or 1 -- number
    self.padding = padding or 1-- number

    self.outputShape = {} -- table
    for i = 1, #inputShape do
        self.outputShape[i] = math.floor((self.inputShape[i] - filterShape[i] + 2 * padding) / self.stride) + 1
    end
    return self
end
setmetatable(Convolution, {
    __call = function(_, inputShape, filterShape, filterNum, padding, stride)
        return Convolution.new(inputShape, filterShape, filterNum, padding, stride)
    end
})

function Convolution:forwardPropagation(inputTensor)

    --#region convolution 함수 구현부
    
    local function Convolution(paddedTensor, filter ,size)-- 매개변수 Tensor , Table
        local function addTable(t1, t2)
            local t3 = {}
            for i = 1, ((#t1+#t2)/2) do
                t3[i] = t1[i] + t2[i] - 1
            end
            return t3
        end

        local startIndex = initializeTable(#paddedTensor.size, 1)-- 전부 1부터 시작한다 stride만큼 증가
        local function indexMoving(tensor, dimensions, depth)
            local subTable = {}
            local dim = dimensions[depth]
            for i = 1, dim do
                if depth < #dimensions then
                    subTable[i] = indexMoving(tensor, dimensions, depth + 1)
                else
                    subTable[i] = (paddedTensor:slice(startIndex, addTable(startIndex, self.filterShape)) * filter):sum()

                    if (startIndex[#startIndex] + self.filterShape[#startIndex]) > tensor.size[#startIndex] then-- 맨 끝 인덱스가 현 차원의 끝이면
                        for j = 1, #startIndex do
                            if startIndex[#startIndex - j] < tensor.size[#startIndex - j]  then-- 인덱스 중에 끝부터 검사해서  최대차원 아닌거 stride만큼 증가
                                startIndex[#startIndex - j] = startIndex[#startIndex - j] + self.stride
                                startIndex[#startIndex] = 1-- 맨 끝 인덱스 초기화
                                break-- 인덱스 증가했으므로 탈출
                            elseif (startIndex[#startIndex - j] + self.filterShape[#startIndex - j]) > tensor.size[depth - (j)] then
                                startIndex[#startIndex - j] = 1 --혹시 최대값 걸리면 초기화
                            end
                        end
                    else
                        startIndex[#startIndex] = startIndex[#startIndex] + self.stride-- 맨 끝 인덱스가 현차원 끝이 아니면 1증가
                    end
                end
            end
            return subTable
        end
        return indexMoving(paddedTensor, size, 1)
    end
    --#endregion
    
    local paddingTensor = {}
    for index = 1, self.filterNum do
        paddingTensor[index] = Tensor.padding(inputTensor, self.padding)
        paddingTensor[index] = Tensor(Convolution(paddingTensor[index], self.filter[index], self.outputShape)):reshape(self.outputShape)
    end
    local outputTensor = Tensor.stack(table.unpack(paddingTensor))-- Tensor 테이블인 paddingTensor를 하나의 텐서로 변환

    return outputTensor
end

function Convolution:backPropagation(learningRate)
    for i = 1, self.filterNum do
        local dL_dF = self.filter[i]:grad()

        local b1, b2 , E = 0.9, 0.999, 1e-8
        self.Momentum_dL_dF[i] = ((b1 * self.Momentum_dL_dF[i]) + ((1-b1) * dL_dF))-- Momentum's dL_dF
        self.RMSprop_dL_dF[i] = ((b2 * self.RMSprop_dL_dF[i]) + ((1-b2) * dL_dF * dL_dF))-- RMSprop's dL_dF

        self.RMSprop_dL_dF[i] = Tensor.apply(self.RMSprop_dL_dF[i], function(x) return x^(-1/2) + E end)

        self.filter[i] = self.filter[i] - learningRate * ((self.RMSprop_dL_dF[i])) * (self.Momentum_dL_dF[i])

        self.filter[i]:grad()
    end
end

-- 레이어 정보를 문자열로
function Convolution:summary()
    local str = " [layer]"
    str = str..string.format(" Type: %10s", "conv")

    local parameterUnits = #self.filter
    str = str..string.format(" Parameter: %5d ", parameterUnits)

    local inputShape_str = "{ "
    for _, value in ipairs(self.inputShape) do
        inputShape_str = inputShape_str..value.." "
    end
    inputShape_str = inputShape_str.."}"
    local filterShape_str = "{ "
    for _, value in ipairs(self.filterShape) do
        filterShape_str = filterShape_str..value.." "
    end
    filterShape_str = filterShape_str.."}"
    str = str..string.format(" Input Shape: %10s", inputShape_str)
    str = str..string.format(" Filter Shape: %10s", filterShape_str)
    str = str..string.format(" Pad: %5d", self.padding)
    str = str..string.format(" Stride: %5d", self.stride)

    return str
end

--풀링 작업중
function Pooling.new(poolingNum, inputShape, poolingShape)
    local self = setmetatable({}, Pooling)
    self.poolingNum = poolingNum-- number
    self.inputShape = inputShape-- table
    --상기 둘을 합친 차원이 실질적인 입력차원

    self.poolingShape = poolingShape-- table

    self.outputShape = {}
    for i = 1, #inputShape do
        self.outputShape[i] = math.ceil(inputShape[i] / poolingShape[i])
    end

    self.poolingMethod = ""-- string
    return self
end
setmetatable(Pooling, {
    __call = function(_, poolingNum, inputShape, poolingShape)
        return Pooling.new(poolingNum, inputShape, poolingShape)
    end
})

function Pooling:forwardPropagation(inputTensor)
    local function addTable(t1, t2)
        local t3 = {}
        for i = 1, ((#t1+#t2)/2) do
            t3[i] = t1[i] + t2[i] - 1
        end
        return t3
    end
    local function pooling(tensor, dimensions, depth, index)
        local subTable = {}
        local dim = dimensions[depth]
        local index = index
        for i = 1, dim do
            if depth < #dimensions then
                index[depth] = i
                subTable[i] = pooling(tensor, dimensions, depth + 1, index)
            else
                index[depth] = i
                local flatten_sliced_tensor = ( tensor:slice(index, addTable(index, self.poolingShape)) ):reshape(mulTable(self.poolingShape))-- 자른 텐서 1차원화
                
                local maxValue = flatten_sliced_tensor.values[1]
                for j = 1, #flatten_sliced_tensor-1 do
                    if flatten_sliced_tensor.values[j] == nil then
                        flatten_sliced_tensor.values[j] = 0
                    end
                    if maxValue < flatten_sliced_tensor.values[j] then
                        maxValue = flatten_sliced_tensor.values[j]
                    end
                end
                subTable[i] = maxValue -- 최대 풀링
            end
        end
        return subTable
    end
    local slicedTensor = {}-- 필터 적용된 개수만큼 생긴 차원에 따라 텐서를 쪼갠다.
    for i = 1, self.poolingNum do
        local startIndex = initializeTable(#self.inputShape, 1)
        slicedTensor[i] = inputTensor:slice({i, table.unpack(startIndex)},{i, table.unpack(self.inputShape)}):reshape(self.inputShape)
        slicedTensor[i] = Tensor(pooling(slicedTensor[i],self.outputShape,1,{}))
    end

    local poolingTensor
    if self.poolingNum == 1 then-- 1일땐 스택이 안됨 나중에 Tensor.stack 수정해야함 인자 한개여도 스택가능하게
        poolingTensor = slicedTensor[1]:reshape(self.poolingNum, table.unpack(self.outputShape))
    else
        poolingTensor = Tensor.stack(slicedTensor)
    end
    return poolingTensor
end

function Pooling:backPropagation()
end

function Pooling:summary()
    local str = " [layer]"
    str = str..string.format(" Type: %10s", "pooling")

    local parameterUnits = 0
    str = str..string.format(" Parameter: %5d ", parameterUnits)

    local inputShape_str = "{ "
    for _, value in ipairs(self.inputShape) do
        inputShape_str = inputShape_str..value.." "
    end
    inputShape_str = inputShape_str.."}"
    local poolingShape_str = "{ "
    for _, value in ipairs(self.poolingShape) do
        poolingShape_str = poolingShape_str..value.." "
    end
    poolingShape_str = poolingShape_str.."}"
    local outputShape_str = "{ "
    for _, value in ipairs(self.outputShape) do
        outputShape_str = outputShape_str..value.." "
    end
    outputShape_str = outputShape_str.."}"
    str = str..string.format(" Input Shape: %10s", inputShape_str)
    str = str..string.format(" Pooling Shape: %10s", poolingShape_str)
    str = str..string.format(" Output Shape: %10s", outputShape_str)

    return str
end


function Dense.new(inputShape, layerShape, activationFunction)-- 매개변수 table, table, number, function
    local self = setmetatable({}, Dense)
    self.inputShape = inputShape -- 레이어 입력 형태 : 테이블
    self.layerShape = layerShape -- 레이어 형태 : 테이블 (사실상 출력 모양)

    self.len_inputShape = mulTable(inputShape) -- number
    self.len_layerShape = mulTable(layerShape) -- number

    self.weights = Tensor.apply( Tensor.emptyTensor({self.len_inputShape, self.len_layerShape}) , function(x)
            return x + GenerateStandardNormal(0, math.sqrt( 2 / self.len_inputShape * self.len_layerShape)) 
        end)
        self.weights = Tensor.deepcopy(self.weights)
    self.Momentum_dL_dW = 0
    self.RMSprop_dL_dW = 0

    self.gamma = Tensor.emptyTensor({1, self.len_layerShape}):fill(1)
    self.Momentum_dL_dG = 0
    self.RMSprop_dL_dG = 0
    
    self.beta = Tensor.emptyTensor({1, self.len_layerShape}):fill(1)
    self.Momentum_dL_dB = 0
    self.RMSprop_dL_dB = 0

    self.activationFunction = activationFunction-- 활성화 함수 이름 (index)
    self.activation = { -- 활성화 함수 모음
        ReLU = function(x)
            if x > 0 then
                return x
            else
                return x * 0.01
            end
        end,
        Tanh = function(x)
            return ( math.exp(1)^(2*x) - 1 ) / (math.exp(1)^(2*x) + 1)
        end,
        SoftMax = function(x)
            return Tensor.SoftMax(x) -- Tensor
        end,
        Linear = function(x)
            return x-- Node
        end
    }

    return self
end
setmetatable(Dense, {
    __call = function(_, inputShape, layerShape, activationFunction)
        return Dense.new(inputShape, layerShape, activationFunction)
    end
})

-- 레이어 정보를 문자열로
function Dense:summary()
    local str = " [layer]"
    str = str..string.format(" Type: %10s", "dense")

    local parameterUnits = #self.weights + #self.gamma + #self.beta
    str = str..string.format(" Parameter: %5d ", parameterUnits)

    local inputShape_str = "{ "
    for _, value in ipairs(self.inputShape) do
        inputShape_str = inputShape_str..value.." "
    end
    inputShape_str = inputShape_str.."}"
    local layerShape_str = "{ "
    for _, value in ipairs(self.layerShape) do
        layerShape_str = layerShape_str..value.." "
    end
    layerShape_str = layerShape_str.."}"
    str = str..string.format(" Input Shape: %10s", inputShape_str)
    str = str..string.format(" Layer Shape: %10s", layerShape_str)
    str = str..string.format(" Activation: %10s", self.activationFunction)

    return str
end

-- 레이어 순전파 
function Dense:forwardPropagation(inputTensor)

    inputTensor = inputTensor:reshape(1, self.len_inputShape) -- 행렬 연산을 위해 형태 조정 (행 벡터 형태로 만들어버림)
    --inputTensor = inputTensor + (math.random() * 2e-8 - 1e-8)-- 0 값을 방지하기 위한 노이즈

    local In_dot_Wt = Tensor.dot(inputTensor, self.weights) -- Tensor 입력값과 가중치 행렬곱
--
    -- 레이어 정규화
    local E = 1e-8
    local tempTensor = Tensor.deepcopy(In_dot_Wt) -- Tensor 평균과 표준편차 계산시에는 영향이 없어야 하므로 임시 복제
        local average = (tempTensor):sum() / #tempTensor -- Node
        local stdDEV = ( ( Tensor.apply(tempTensor, function(x) return (x - average)^2 end) ):sum() )^(1/2)-- Node
        local Normalized = Tensor.apply(In_dot_Wt, function(x) return (x - average) / (stdDEV + E) end) -- Tensor

        --레이어 정규화 레이어 별로 수행하고 감마 베타 적용하여 테이블에 삽입
        local gx_b = self.gamma * Normalized + self.beta -- Tensor
--]]
    --local gx_b = In_dot_Wt
    -- 활성화 함수 적용
    local output = Tensor.apply(gx_b, self.activation[self.activationFunction])

    return output:reshape(self.layerShape)
end

function Dense:backPropagation(learningRate)
    -- ADAM 옵티마이저
    local dL_dW = self.weights:grad()
    local dL_dG = self.gamma:grad()
    local dL_dB = self.beta:grad()

    local b1, b2 , E = 0.9, 0.999, 1e-8
    self.Momentum_dL_dW = ((b1 * self.Momentum_dL_dW) + ((1-b1) * dL_dW))-- Momentum's dL_dW
    self.RMSprop_dL_dW = ((b2 * self.RMSprop_dL_dW) + ((1-b2) * dL_dW * dL_dW))-- RMSprop's dL_dW
    self.Momentum_dL_dG = ((b1 * self.Momentum_dL_dG) + ((1-b1) * dL_dG))-- Momentum's dL_dG
    self.RMSprop_dL_dG = ((b2 * self.RMSprop_dL_dG) + ((1-b2) * dL_dG * dL_dG))-- RMSprop's dL_dG
    self.Momentum_dL_dB = ((b1 * self.Momentum_dL_dB) + ((1-b1) * dL_dB))-- Momentum's dL_dB
    self.RMSprop_dL_dB = ((b2 * self.RMSprop_dL_dB) + ((1-b2) * dL_dB * dL_dB))-- RMSprop's dL_dB

    -- Tensor 지수 연산
    self.RMSprop_dL_dW = Tensor.apply(self.RMSprop_dL_dW, function(x) return x^(-1/2) + E end)
    self.RMSprop_dL_dG = Tensor.apply(self.RMSprop_dL_dG, function(x) return x^(-1/2) + E end)
    self.RMSprop_dL_dB = Tensor.apply(self.RMSprop_dL_dB, function(x) return x^(-1/2) + E end)

    -- 오차 반영
    self.weights = self.weights - learningRate * ((self.RMSprop_dL_dW)) * (self.Momentum_dL_dW)
    self.gamma = self.gamma - learningRate * ((self.RMSprop_dL_dG)) * (self.Momentum_dL_dG)
    self.beta = self.beta - learningRate * ((self.RMSprop_dL_dB)) * (self.Momentum_dL_dB)
    self.weights = Tensor.deepcopy(self.weights)
    self.gamma = Tensor.deepcopy(self.gamma)
    self.beta = Tensor.deepcopy(self.beta)

    -- 혹시 모를 기울기 초기화
    self.weights:grad()
    self.gamma:grad()
    self.beta:grad()
end


function Model.new(name)
    local self = setmetatable({}, Model)

    self.name = name or "None"

    self.layers = {}

    self.layer = {dense = {}, convolution = {}, pooling = {}}
    self.layer.dense = setmetatable(self.layer.dense, {
        __call = function(_, inputShape, layerShape, activationFunction)-- Model.layer.dense() 식으로 호출
            self.layers[#self.layers + 1] = Dense(inputShape, layerShape, activationFunction)-- 레이어를 생성하고 빈공간에 넣는다.
            print("Dense layer generated")
        end
    })
    self.layer.convolution = setmetatable(self.layer.convolution, {
        __call = function(_, inputShape, filterShape, filterNum, padding, stride)-- Model.layer.convolution() 식으로 호출
            self.layers[#self.layers + 1] = Convolution(inputShape, filterShape, filterNum, padding, stride)-- 레이어를 생성하고 빈공간에 넣는다.
            print("Convolution layer generated")
        end
    })
    self.layer.pooling = setmetatable(self.layer.pooling, {
        __call = function(_, poolingNum, inputShape, poolingShape)-- Model.layer.convolution() 식으로 호출
            self.layers[#self.layers + 1] = Pooling(poolingNum, inputShape, poolingShape)-- 레이어를 생성하고 빈공간에 넣는다.
            print("Pooling layer generated")
        end
    })

    -- 손실 함수 모음 return은 오류값을 가진 Node
    self.lossFunction = {
        MSE = function(targetTensor, outputTensor)
            return ( Tensor.apply((targetTensor - outputTensor), function(x) return x * x end) ):sum()-- Node
        end
    }

    -- 데이터 저장
    self.data = Data()

    -- 모델의 출력값
    self.output = nil

    return self
end
setmetatable(Model, {
    __call = function(_,name)
        return Model.new(name)
    end
})

-- 모델 순전파
function Model:forwardPropagation(inputTensor)
    local output = Tensor.deepcopy(inputTensor)
    for _, layer in ipairs(self.layers) do
        output = layer:forwardPropagation(output)
    end
    return output-- Tensor
end

function Model:backPropagation(learningRate)
    learningRate = learningRate or 0.1

    for _, layer in ipairs(self.layers) do
        layer:backPropagation(learningRate)
    end
end

-- 데이터를 직접 받아 하는 훈련
function Model:learn(input, target, loss, learningRate)
    loss = loss or 'MSE'
    learningRate = learningRate or 0.1

    local error = 0
    local output = 0
    output = self:forwardPropagation(input)
    error = self.lossFunction[loss](target, output)
    error:backward()-- dL_d[] 꼴로 역전파
    
    self:backPropagation(learningRate)

    --self.data:add(input, target)

    return error
end

-- 데이터 셋에 의한 모델 훈련
function Model:train(loss, learningRate)
    loss = loss or 'MSE'
    learningRate = learningRate or 0.1

    local input , target = self.data:randomSampling()

    local error = 0
    local output = 0
    output = self:forwardPropagation(input)
    error = self.lossFunction[loss](target, output)
    error:backward()-- dL_d[] 꼴로 역전파
    
    self:backPropagation(learningRate)
    return error
end

-- 신경망 모델에 대한 정보 요약
function Model:summary()
    local str = "[Model]=======================================================================================================\n"
    str = str..string.format(" Name: %10s\n",self.name)

    str = str.. "--------------------------------------------------------------------------------------------------------------\n"

    for _, layer in ipairs(self.layers) do
        str = str..layer:summary().."\n"
    end

    str = str.. "--------------------------------------------------------------------------------------------------------------\n"
    print(str)
end

function Data.new()
    local self = setmetatable({}, Data)
    self.inputDatas = {} --Tensors
    self.targetDatas = {} --Tensors 
    return self
end
setmetatable(Data, {
    __call = function(_)
        return Data.new()
    end
})

-- 데이터 추가
function Data:add(inputData, targetData)
    table.insert(self.inputDatas, inputData)
    table.insert(self.targetDatas, targetData)
end

function Data:randomSampling()
    local index = math.random(1, (#self.inputDatas+#self.targetDatas)/2)
    local inputDatas = self.inputDatas[index]
    local targetDatas = self.targetDatas[index]
    return inputDatas, targetDatas-- 배치차원 포함된 자료
end

return Model