Tensor = require("Tensor")
DataManager = require("DataManager")

local Attention = {}
Attention.__index = Attention

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
--
function Attention.new(headNum, QShape, KShape, VShape)
    local self = setmetatable({}, Attention)
    self.headNum = headNum -- int
    self.QShape = QShape -- table {Qsqn, QFig0} * W -> Q: {Qsqn, hidden}
    self.KShape = KShape -- table {Ksqn, KFig0} * W -> K: {Ksqn, hidden}
    self.VShape = VShape -- table {Ksqn, VFig0} * W -> V: {Ksqn, hidden}
    -- 가능하면 KFig0 == VFig0 == QFig0
    self.hiddenDimension = self.VShape[2] / headNum -- 딱 맞아 떨어져야 함
    if self.VShape[2] % headNum ~= 0 then
        error("incorrect hiddenDimension")
    end
    -- output size: (batch) * Qsqn * (hidden*head)(== Fig 아마도)

    self.W_Qs = {}
    self.W_Ks = {}
    self.W_Vs = {}
    for i = 1, headNum do
        self.W_Qs[i] = Tensor.apply( Tensor.emptyTensor({self.QShape[2], self.hiddenDimension}) , function(x)
            return x + GenerateStandardNormal(0, math.sqrt( 2 / self.QShape[2] * self.hiddenDimension)) 
        end)
        self.W_Ks[i] = Tensor.apply( Tensor.emptyTensor({self.KShape[2], self.hiddenDimension}) , function(x)
            return x + GenerateStandardNormal(0, math.sqrt( 2 / self.KShape[2] * self.hiddenDimension)) 
        end)
        self.W_Vs[i] = Tensor.apply( Tensor.emptyTensor({self.VShape[2], self.hiddenDimension}) , function(x)
            return x + GenerateStandardNormal(0, math.sqrt( 2 / self.VShape[2] * self.hiddenDimension)) 
        end)
    end

    self.Momentum_dL_dW_Qs = initializeTable(headNum, 0)
    self.RMSprop_dL_dW_Qs = initializeTable(headNum, 0)
    self.Momentum_dL_dW_Ks = initializeTable(headNum, 0)
    self.RMSprop_dL_dW_Ks = initializeTable(headNum, 0)
    self.Momentum_dL_dW_Vs = initializeTable(headNum, 0)
    self.RMSprop_dL_dW_Vs = initializeTable(headNum, 0)

    self.iteration = 1

    self.input = nil
    self.output = nil

    return self
end
setmetatable(Attention, {
    __call = function(_, headNum, QShape, KShape, VShape)
        return Attention.new(headNum, QShape, KShape, VShape)
    end
})

function Attention:forwardPropagation(inputTensor)
    -- inputTensor.size {(batch), QKV(Mask), sqn, fig}
    -- outputTensor.size {batch, head, sqn, fig}
    local batchSize = inputTensor.dimension == 4 and inputTensor.size[1] or 1
    local QKVMaskSize = inputTensor.dimension == 4 and inputTensor.size[2] or inputTensor.size[1]
    local sequenceSize = inputTensor.dimension == 4 and inputTensor.size[3] or inputTensor.size[2]
    local figureSize = inputTensor.dimension == 4 and inputTensor.size[4] or inputTensor.size[3]
    inputTensor = inputTensor:reshape(batchSize, QKVMaskSize, sequenceSize, figureSize)
    inputTensor = Tensor.apply(inputTensor, function(x) return x + (math.random() * 2e-8 - 1e-8) end)

    local batchTensorTable = {}
    for batch = 1, batchSize do
        local X_Q = inputTensor[batch][1]
        local X_K = inputTensor[batch][2]
        local X_V = inputTensor[batch][3]
        local Mask = QKVMaskSize == 4 and inputTensor[batch][4] or nil

        local attentionTensorTable = {}
        for head = 1, self.headNum do
            local Q = Tensor.dot(X_Q, self.W_Qs[head])
            local K = Tensor.dot(X_K, self.W_Ks[head])
            local V = Tensor.dot(X_V, self.W_Vs[head])

            local Q_KT = Tensor.dot(Q, K:transpose())
            Q_KT = (Q_KT / math.sqrt(self.hiddenDimension))
            if Mask then
                Q_KT = Q_KT + Mask -- 마스크가 있으면 적용
            end
            for idx = 1, Q_KT.size[1] do
                Q_KT[idx] = Tensor.activation("SoftMax")(Q_KT[idx])
            end
            local attention = Tensor.dot(Q_KT, V) -- {Qsqn, hidden}
            table.insert(attentionTensorTable, attention)
        end
        local batchTensor = table.remove(attentionTensorTable,1)
        for key, attentionTensor in pairs(attentionTensorTable) do
            batchTensor = Tensor.concat(batchTensor, attentionTensor, 2) -- {Qsqn, hidden*head}
        end
        table.insert(batchTensorTable, batchTensor)
    end
    local outputTensor = Tensor.stack(unpack(batchTensorTable)) -- {batch, Qsqn, hidden*head} == {batch, Qsqn, QFig}

    self.input = inputTensor
    self.output = outputTensor
    return Tensor.deepcopy(outputTensor)
end

function Attention:backPropagation(learningRate, grad)
    local t = self.iteration or 1  -- 현재 업데이트 반복 횟수 저장
    self.output:reshape(grad.size):backwardTensor(grad)
    for head = 1, self.headNum do
        local dL_dW_Qs = self.W_Qs[head]:grad():detach_()
        local dL_dW_Ks = self.W_Ks[head]:grad():detach_()
        local dL_dW_Vs = self.W_Vs[head]:grad():detach_()

        local b1, b2 , E = 0.9, 0.999, 1e-8
        self.Momentum_dL_dW_Qs[head] = Tensor.deepcopy((b1 * self.Momentum_dL_dW_Qs[head]) + ((1-b1) * dL_dW_Qs))-- Momentum's dL_dW_Qs
        self.RMSprop_dL_dW_Qs[head] = Tensor.deepcopy((b2 * self.RMSprop_dL_dW_Qs[head]) + ((1-b2) * dL_dW_Qs * dL_dW_Qs))-- RMSprop's dL_dW_Qs
        self.Momentum_dL_dW_Ks[head] = Tensor.deepcopy((b1 * self.Momentum_dL_dW_Ks[head]) + ((1-b1) * dL_dW_Ks))-- Momentum's dL_dW_Ks
        self.RMSprop_dL_dW_Ks[head] = Tensor.deepcopy((b2 * self.RMSprop_dL_dW_Ks[head]) + ((1-b2) * dL_dW_Ks * dL_dW_Ks))-- RMSprop's dL_dW_Ks
        self.Momentum_dL_dW_Vs[head] = Tensor.deepcopy((b1 * self.Momentum_dL_dW_Vs[head]) + ((1-b1) * dL_dW_Vs))-- Momentum's dL_dW_Vs
        self.RMSprop_dL_dW_Vs[head] = Tensor.deepcopy((b2 * self.RMSprop_dL_dW_Vs[head]) + ((1-b2) * dL_dW_Vs * dL_dW_Vs))-- RMSprop's dL_dW_Vs

        -- 바이어스 보정 (Bias Correction)
        local m_hat_W_Qs = (self.Momentum_dL_dW_Qs) / (1 - b1^t)
        local v_hat_W_Qs = (self.RMSprop_dL_dW_Qs) / (1 - b2^t)
        v_hat_W_Qs = Tensor.apply(v_hat_W_Qs, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)
        local m_hat_W_Ks = (self.Momentum_dL_dW_Ks) / (1 - b1^t)
        local v_hat_W_Ks = (self.RMSprop_dL_dW_Ks) / (1 - b2^t)
        v_hat_W_Ks = Tensor.apply(v_hat_W_Ks, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)
        local m_hat_W_Vs = (self.Momentum_dL_dW_Vs) / (1 - b1^t)
        local v_hat_W_Vs = (self.RMSprop_dL_dW_Vs) / (1 - b2^t)
        v_hat_W_Vs = Tensor.apply(v_hat_W_Vs, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)

        self.W_Qs[head] = Tensor.deepcopy(self.W_Qs[head] - learningRate * (m_hat_W_Qs * v_hat_W_Qs + 0.1 * self.W_Qs[head]))
        self.W_Ks[head] = Tensor.deepcopy(self.W_Ks[head] - learningRate * (m_hat_W_Ks * v_hat_W_Ks + 0.1 * self.W_Ks[head]))
        self.W_Vs[head] = Tensor.deepcopy(self.W_Vs[head] - learningRate * (m_hat_W_Vs * v_hat_W_Vs + 0.1 * self.W_Vs[head]))
    end
    self.iteration = t + 1
    
    local inputGrad = self.input:grad()
    self.input = nil
    self.output = nil
    return inputGrad
end

function Attention:summary()
    local str = " [layer]"
    str = str..string.format("| Type: %10s |", "Attention")

    local parameterUnits = 0
    str = str..string.format("| Parameter: %5d |", parameterUnits)
end

function Attention:save(fileName, layerName)
    local data = DataManager.loadTableFromFile(fileName)
    if data then
        data[layerName] = {}
        data[layerName]["Weight_Qs"] = {}
        for key, W_Q in pairs(self.W_Qs) do
            table.insert(data[layerName]["Weight_Qs"], Tensor.toTable(W_Q))
        end
        data[layerName]["Weight_Ks"] = {}
        for key, W_K in pairs(self.W_Ks) do
            table.insert(data[layerName]["Weight_Ks"], Tensor.toTable(W_K))
        end
        data[layerName]["Weight_Vs"] = {}
        for key, W_V in pairs(self.W_Vs) do
            table.insert(data[layerName]["Weight_Vs"], Tensor.toTable(W_V))
        end
        DataManager.saveTableToFile(data, fileName)
    else
        error("Can't save Data!.")
    end
end

function Attention:load(fileName, layerName)
    local data = DataManager.loadTableFromFile(fileName)
    if data then
        for key, W_Q in pairs(data[layerName]["Weight_Qs"]) do
            if W_Q then
                self.W_Qs[key] = Tensor(W_Q)
            end
        end
        for key, W_K in pairs(data[layerName]["Weight_Ks"]) do
            if W_K then
                self.W_Ks[key] = Tensor(W_K)
            end
        end
        for key, W_V in pairs(data[layerName]["Weight_Vs"]) do
            if W_V then
                self.W_Vs[key] = Tensor(W_V)
            end
        end
    else
        error("Can't load Data!.")
    end
end
--]]
function Convolution.new(featureNum, inputShape, filterShape, filterNum, padding, stride, activationFunctionName)
    local self = setmetatable({}, Convolution)
    self.featureNum = featureNum -- number

    self.inputShape = inputShape -- table
    table.insert(self.inputShape, 1, featureNum)

    self.filterShape = filterShape -- table
    table.insert(self.filterShape, 1, featureNum)
    self.filterNum = filterNum -- number

    self.stride = stride or 1 -- number
    self.padding = padding or 1-- number

    self.outputShape = {1} -- table
    for i = 2, #self.filterShape do
        self.outputShape[i] = math.floor((self.inputShape[i] - self.filterShape[i] + 2 * self.padding) / self.stride) + 1
    end

    self.filter = {}
    self.bias = {}
    for i = 1, filterNum do
        self.filter[i] = Tensor.apply( Tensor.emptyTensor(self.filterShape) , function (x) return x + (2*math.random()-1) end)
        --self.bias[i] = Tensor.apply( Tensor.emptyTensor(self.outputShape) , function (x) return x + (2*math.random()-1) end)
        self.bias[i] = Tensor({2*math.random()-1})
    end

    self.activationFunctionName = activationFunctionName or 'ReLU'-- 활성화 함수 이름 (index)

    self.Momentum_dL_dF = initializeTable(filterNum, 0)
    self.RMSprop_dL_dF = initializeTable(filterNum, 0)
    self.Momentum_dL_dB = initializeTable(filterNum, 0)
    self.RMSprop_dL_dB = initializeTable(filterNum, 0)

    self.iteration = 1

    self.input = nil
    self.output = nil

    return self
end
setmetatable(Convolution, {
    __call = function(_, featureNum, inputShape, filterShape, filterNum, padding, stride, activationFunctionName)
        return Convolution.new(featureNum, inputShape, filterShape, filterNum, padding, stride, activationFunctionName)
    end
})

function Convolution:forwardPropagation(inputTensor)
    local batchSize = 1
    local extraDimension = #inputTensor.size - #self.inputShape
    if extraDimension >= 0 then
        batchSize = inputTensor.size[1]
    end

    inputTensor = inputTensor:reshape({batchSize, unpack(self.inputShape)})

    local batchFeatureTensorTable = {}
    for batch = 1, batchSize do
        local featureTensorTable = {}
        local featureTensor = inputTensor[batch]
        for filterNum = 1, self.filterNum do
            local function _convolution(tensor, filter, indexTable, outputDimension, depth)
                local dim = outputDimension[depth]
                local subTable = {}
                for i = 1, dim do
                    if depth < #outputDimension then
                        -- 하위 차원으로 재귀적으로 테이블 생성
                        local newIndexTable = tableCopy(indexTable)
                        newIndexTable[depth] = (i - 1) * self.stride
                        subTable[i] = _convolution(tensor, filter, newIndexTable, outputDimension, depth + 1)
                    else
                        indexTable[depth] = (i - 1) * self.stride
                        local filterSize = filter.size
                        local startIndex = addTableToTable(indexTable, initializeTable(#indexTable, 1))
                        local endIndex = addTableToTable(filterSize, indexTable)
                        local slicedTensor = tensor:slice(startIndex, endIndex)
                        subTable[i] = (slicedTensor * filter):sum()
                    end
                end
                return subTable
            end

            local filter = self.filter[filterNum]
            local padDimension = initializeTable(#featureTensor.size, self.padding*2)
            padDimension[1] = 0 -- 특성차원 패딩은 하지 않음
            local paddedTensor = Tensor.padding(featureTensor, padDimension, 0)
            local convolutionedTensor = Tensor(_convolution(paddedTensor, filter, initializeTable(#self.inputShape, 0), self.outputShape, 1)) + self.bias[filterNum].values[1] -- bias 는 Tensor이기 때문에 Node 만 뽑아서 연산
            -- 활성화 함수 적용
            convolutionedTensor = Tensor.activation(self.activationFunctionName)(convolutionedTensor)
            -- 테이블에 컨볼루션된 텐서 추가
            table.insert(featureTensorTable, convolutionedTensor) -- {1, 출력차원...} 크기를 가짐
        end
        -- 배치테이블에 병렬으로 처리된 특성맵들 추가
        local batchFeatureTensor = table.remove(featureTensorTable,1)
        for key, tensor in pairs(featureTensorTable) do
            batchFeatureTensor = Tensor.concat(batchFeatureTensor, tensor, 1) -- {filterNum, 출력차원} 크기를 가짐
        end
        table.insert(batchFeatureTensorTable, batchFeatureTensor)
    end
    local outputTensor = Tensor.stack(unpack(batchFeatureTensorTable)) -- {batch , filterNum, 출력차원} 크기를 가짐

    self.input = inputTensor
    self.output = outputTensor
    return Tensor.deepcopy(outputTensor)
end

function Convolution:backPropagation(learningRate, grad)
    local t = self.iteration or 1  -- 현재 업데이트 반복 횟수 저장
    self.output:reshape(grad.size):backwardTensor(grad)

    for i = 1, self.filterNum do
        local dL_dF = (self.filter[i]:grad()):detach_()
        local dL_dB = (self.bias[i]:grad()):detach_()

        local b1, b2 , E = 0.9, 0.999, 1e-8
        self.Momentum_dL_dF[i] = Tensor.deepcopy((b1 * self.Momentum_dL_dF[i]) + ((1-b1) * dL_dF)) -- Momentum's dL_dF
        self.RMSprop_dL_dF[i] = Tensor.deepcopy((b2 * self.RMSprop_dL_dF[i]) + ((1-b2) * dL_dF * dL_dF)) -- RMSprop's dL_dF
        self.Momentum_dL_dB[i] = Tensor.deepcopy((b1 * self.Momentum_dL_dB[i]) + ((1-b1) * dL_dB)) -- Momentum's dL_dB
        self.RMSprop_dL_dB[i] = Tensor.deepcopy((b2 * self.RMSprop_dL_dB[i]) + ((1-b2) * dL_dB * dL_dB)) -- RMSprop's dL_dB

        -- 바이어스 보정 (Bias Correction)
        local m_hat_F = (self.Momentum_dL_dF[i]) / (1 - b1^t)
        local v_hat_F = (self.RMSprop_dL_dF[i]) / (1 - b2^t)
        v_hat_F = Tensor.apply(v_hat_F, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)
        local m_hat_B = (self.Momentum_dL_dB[i]) / (1 - b1^t)
        local v_hat_B = (self.RMSprop_dL_dB[i]) / (1 - b2^t)
        v_hat_B = Tensor.apply(v_hat_B, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)

        self.filter[i] = Tensor.deepcopy(self.filter[i] - learningRate * (m_hat_F * v_hat_F + 0.1 * self.filter[i]))
        self.bias[i] = Tensor.deepcopy(self.bias[i] - learningRate * m_hat_B * v_hat_B)
    end
    self.iteration = t + 1

    local inputGrad = self.input:grad()
    self.input = nil
    self.output = nil
    return inputGrad
end

function Convolution:save(fileName, layerName)
    local data = DataManager.loadTableFromFile(fileName)
    if data then
        data[layerName] = {}
        data[layerName]["filterWeights"] = {}
        for key, filter in pairs(self.filter) do
            table.insert(data[layerName]["filterWeights"], Tensor.toTable(filter))
        end
        data[layerName]["bias"] = {}
        for key, bias in pairs(self.bias) do
            table.insert(data[layerName]["bias"], Tensor.toTable(bias))
        end
        DataManager.saveTableToFile(data, fileName)
    else
        error("Can't save Data!.")
    end
end

function Convolution:load(fileName, layerName)
    local data = DataManager.loadTableFromFile(fileName)
    if data then
        for key, filter in pairs(data[layerName]["filterWeights"]) do
            if filter then
                self.filter[key] = Tensor(filter)
            end
        end
        for key, bias in pairs(data[layerName]["bias"]) do
            if bias then
                self.bias[key] = Tensor(bias)
            end
        end
    else
        error("Can't load Data!.")
    end
end

-- 레이어 정보를 문자열로
function Convolution:summary()
    local str = " [layer]"
    str = str..string.format("| Type: %10s |", "Convolution")

    local parameterUnits = (mulTable(self.filterShape)) * self.filterNum
    str = str..string.format("| Parameter: %5d |", parameterUnits)

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
    str = str..string.format("| Input Shape: %10s |", inputShape_str)
    str = str..string.format("| Filter Shape: %10s |", filterShape_str)
    str = str..string.format("| FilterNum: %10d |", self.filterNum)

    return str
end

--풀링 작업중
function Pooling.new(poolingNum, inputShape, poolingShape, padding, stride, poolingMethod)
    local self = setmetatable({}, Pooling)
    self.poolingNum = poolingNum -- number
    self.inputShape = inputShape -- table
    --상기 둘을 합친 차원이 실질적인 입력차원

    self.poolingShape = poolingShape -- table

    self.stride = stride or 1 -- number
    self.padding = padding or 1-- number

    self.outputShape = {} -- table
    for i = 1, #self.poolingShape do
        self.outputShape[i] = math.floor((self.inputShape[i] - self.poolingShape[i] + 2 * self.padding) / self.stride) + 1
    end

    self.poolingMethod = poolingMethod -- string

    self.input = nil
    self.output = nil

    return self
end
setmetatable(Pooling, {
    __call = function(_, poolingNum, inputShape, poolingShape, padding, stride, poolingMethod)
        return Pooling.new(poolingNum, inputShape, poolingShape, padding, stride, poolingMethod)
    end
})

function Pooling:forwardPropagation(inputTensor)
    local batchSize = 1
    local extraDimension = #inputTensor.size - #self.inputShape
    if extraDimension == 2 or (extraDimension == 1 and self.featureNum == 0) then
        batchSize = inputTensor.size[1]
    else
        batchSize = 1
    end

    local poolingNum = self.poolingNum == 0 and 1 or self.poolingNum
    
    local poolingShape = self.inputShape

    inputTensor = inputTensor:reshape({batchSize, poolingNum, unpack(poolingShape)})

    local parallelBatchTensorTable = {}
    for batch = 1, batchSize do
        local parallelPoolingTensorTable = {}
        -- 특성맵 병렬 처리
        for pooling = 1, poolingNum do
            -- 한 입력 특성맵에서 필터 개수만큼 나오는 출력 특성맵 텐서 테이블
            local featureTensor = inputTensor[batch][pooling]

            -- TODO
            -- 입력 차원이랑 풀링 차원이랑 차이나는거 고려해서 슬라이스
            -- 풀링차원이랑 입력 차원에서 슬라이스 한거 크기 안맞으면 제로 패딩해서 크기 맞추기
            local function _pooling(tensor, indexTable, outputDimension, depth)
                local dim = outputDimension[depth]
                local subTable = {}
                for i = 1, dim do
                    if depth < #outputDimension then
                        -- 하위 차원으로 재귀적으로 테이블 생성
                        local newIndexTable = tableCopy(indexTable)
                        newIndexTable[depth] = (i - 1) * self.stride
                        subTable[i] = _pooling(tensor, newIndexTable, outputDimension, depth + 1)
                    else
                        indexTable[depth] = (i - 1) * self.stride
                        local poolingSize = self.poolingShape
                        local startIndex = addTableToTable(indexTable, initializeTable(#indexTable, 1))
                        local endIndex = addTableToTable(poolingSize, indexTable)
                        local slicedTensor = tensor:slice(startIndex, endIndex)

                        subTable[i] = slicedTensor:sum() / mulTable(self.poolingShape)
                    end
                end
                return subTable
            end

            local paddedTensor = Tensor.padding(featureTensor, initializeTable(#self.poolingShape, self.padding*2), 0)
            local poolingedTensor = Tensor(_pooling(paddedTensor, initializeTable(#self.inputShape, 0), self.outputShape, 1))
            -- 테이블에 컨볼루션 추가
            table.insert(parallelPoolingTensorTable, poolingedTensor)
        end
        -- 특성맵테이블에 병렬으로 처리된 특성맵들 추가
        local parallelFeatureTensor = Tensor.stack(unpack(parallelPoolingTensorTable))
        table.insert(parallelBatchTensorTable, parallelFeatureTensor)
    end
    local outputTensor = Tensor.stack(unpack(parallelBatchTensorTable))

    self.input = inputTensor
    self.output = outputTensor

    return Tensor.deepcopy(outputTensor)
end

function Pooling:backPropagation(learningRate, grad)
    -- 풀링은 역전파로 업데이트할 매개변수가 없음
    self.output:reshape(grad.size):backwardTensor(grad)
    local inputGrad = self.input:grad()
    self.input = nil
    self.output = nil
    return inputGrad
end

function Pooling:save(fileName, layerName)
end

function Pooling:load(fileName, layerName)
end

function Pooling:summary()
    local str = " [layer]"
    str = str..string.format("| Type: %10s |", "Pooling")

    local parameterUnits = 0
    str = str..string.format("| Parameter: %5d |", parameterUnits)

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
    str = str..string.format("| Input Shape: %10s |", inputShape_str)
    str = str..string.format("| Pooling Shape: %10s |", poolingShape_str)
    str = str..string.format("| Output Shape: %10s |", outputShape_str)

    return str
end

function Dense.new(inputShape, layerShape, activationFunctionName, normalized)-- 매개변수 table, table, number, function
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

    self.gamma = Tensor.emptyTensor({self.len_layerShape}):fill(1)
    self.Momentum_dL_dG = 0
    self.RMSprop_dL_dG = 0
    
    self.beta = Tensor.emptyTensor({self.len_layerShape}):fill(1)
    self.Momentum_dL_dB = 0
    self.RMSprop_dL_dB = 0

    self.iteration = 1

    self.activationFunctionName = activationFunctionName -- 활성화 함수 이름 (index)

    if normalized == nil then
        normalized = true
    end
    self.layerNormalized = normalized

    self.input = nil
    self.output = nil

    return self
end
setmetatable(Dense, {
    __call = function(_, inputShape, layerShape, activationFunctionName, normalized)
        return Dense.new(inputShape, layerShape, activationFunctionName, normalized)
    end
})

-- 레이어 순전파 
function Dense:forwardPropagation(inputTensor)
    local outputShape = {}
    inputTensor = Tensor.apply(inputTensor, function(x) return x + (math.random() * 2e-8 - 1e-8) end)

    local batchSize = 1 -- 입력 차원이 아닌 여분 차원의 길이
    local column = 1 -- 입력 차원의 길이
    for idx = 1, #inputTensor.size - #self.inputShape do
        table.insert(outputShape, inputTensor.size[idx])
        batchSize = batchSize * inputTensor.size[idx]
    end
    for idx = #inputTensor.size - #self.inputShape + 1, #inputTensor.size do
        column = column * inputTensor.size[idx]
    end
    for _, value in pairs(self.layerShape) do
        table.insert(outputShape, value)
    end
    inputTensor = inputTensor:reshape(batchSize, column) -- 행렬 연산을 위해 형태 조정 (행렬 형태로 만들어버림)
    
    local In_dot_Wt = Tensor.dot(inputTensor, self.weights) -- Tensor 입력값과 가중치 행렬곱
    local gx_b = In_dot_Wt
    for i = 1, batchSize do
        if self.layerNormalized == true then
            -- 레이어 정규화
            local E = 1e-8
            local tempTensor = (In_dot_Wt[i])--In_dot_Wt[i]--Tensor.deepcopy(In_dot_Wt[i]) -- Tensor 평균과 표준편차 계산시에는 영향이 없어야 하므로 임시 복제
            local average = ((tempTensor):sum() / tempTensor:__len()) -- Node
            local stdDEV = (( ( Tensor.apply(tempTensor, function(x) return (x - average)^2 end) ):sum() / tempTensor:__len() )^(1/2)) -- Node
            local Normalized = Tensor.apply(In_dot_Wt[i], function(x) return (x - average) / (stdDEV + E) end) -- Tensor
            --레이어 정규화 레이어 별로 수행하고 감마 베타 적용하여 테이블에 삽입
            gx_b[i] = self.gamma * Normalized + self.beta -- Tensor
        else
            gx_b[i] = In_dot_Wt[i] + self.beta
        end

        -- 활성화 함수 적용
        gx_b[i] = Tensor.activation(self.activationFunctionName)(gx_b[i])
    end

    local outputTensor = gx_b:reshape(outputShape)
    self.input = inputTensor
    self.output = outputTensor

    return Tensor.deepcopy(outputTensor)--gx_b:reshape(outputShape) -- 형태 변경으로 정상화
end

function Dense:backPropagation(learningRate, grad)
    local t = self.iteration or 1  -- 현재 업데이트 반복 횟수 저장
    self.output:reshape(grad.size):backwardTensor(grad)
    -- ADAM 옵티마이저
    local dL_dW = (self.weights:grad()):detach_()
    local dL_dG = (self.gamma:grad()):detach_()
    local dL_dB = (self.beta:grad()):detach_()

    local b1, b2 , E = 0.9, 0.999, 1e-8
    -- 얕은 복사로 인해 메모리 누수 발생 따라서 깊은 복사
    self.Momentum_dL_dW = Tensor.deepcopy((b1 * self.Momentum_dL_dW) + ((1-b1) * dL_dW))-- Momentum's dL_dW
    self.RMSprop_dL_dW = Tensor.deepcopy((b2 * self.RMSprop_dL_dW) + ((1-b2) * dL_dW * dL_dW))-- RMSprop's dL_dW
    self.Momentum_dL_dG = Tensor.deepcopy((b1 * self.Momentum_dL_dG) + ((1-b1) * dL_dG))-- Momentum's dL_dG
    self.RMSprop_dL_dG = Tensor.deepcopy((b2 * self.RMSprop_dL_dG) + ((1-b2) * dL_dG * dL_dG))-- RMSprop's dL_dG
    self.Momentum_dL_dB = Tensor.deepcopy((b1 * self.Momentum_dL_dB) + ((1-b1) * dL_dB))-- Momentum's dL_dB
    self.RMSprop_dL_dB = Tensor.deepcopy((b2 * self.RMSprop_dL_dB) + ((1-b2) * dL_dB * dL_dB))-- RMSprop's dL_dB

    -- 바이어스 보정 (Bias Correction)
    local m_hat_W = (self.Momentum_dL_dW) / (1 - b1^t)
    local v_hat_W = (self.RMSprop_dL_dW) / (1 - b2^t)
    v_hat_W = Tensor.apply(v_hat_W, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)
    local m_hat_G = (self.Momentum_dL_dG) / (1 - b1^t)
    local v_hat_G = (self.RMSprop_dL_dG) / (1 - b2^t)
    v_hat_G = Tensor.apply(v_hat_G, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)
    local m_hat_B = (self.Momentum_dL_dB) / (1 - b1^t)
    local v_hat_B = (self.RMSprop_dL_dB) / (1 - b2^t)
    v_hat_B = Tensor.apply(v_hat_B, function(x) return (x > Node(0) and x^(-1/2) or 0) + E end)

    -- 오차 반영 (L2 정규화 포함)
    self.weights = Tensor.deepcopy(self.weights - learningRate * (m_hat_W * v_hat_W + 0.2 * self.weights))
    self.gamma = Tensor.deepcopy(self.gamma - learningRate * m_hat_G * v_hat_G)
    self.beta = Tensor.deepcopy(self.beta - learningRate * m_hat_B * v_hat_B)

    self.iteration = t + 1

    local inputGrad = self.input:grad()
    self.input = nil
    self.output = nil
    return inputGrad
end

function Dense:save(fileName, layerName)
    local data = DataManager.loadTableFromFile(fileName)
    if data then
        data[layerName] = {}
        data[layerName]["weights"] = {}
        table.insert(data[layerName]["weights"], Tensor.toTable(self.weights))
        data[layerName]["gamma"] = {}
        table.insert(data[layerName]["gamma"], Tensor.toTable(self.gamma))
        data[layerName]["beta"] = {}
        table.insert(data[layerName]["beta"], Tensor.toTable(self.beta))
        DataManager.saveTableToFile(data, fileName)
    else
        error("Can't save Data!.")
    end
end

function Dense:load(fileName, layerName)
    local data = DataManager.loadTableFromFile(fileName)
    if data then
        for key, weight in pairs(data[layerName]["weights"]) do
            if weight then
                self.weights = Tensor(weight)
            end
        end
        for key, gamma in pairs(data[layerName]["gamma"]) do
            if gamma then
                self.gamma = Tensor(gamma)
            end
        end
        for key, beta in pairs(data[layerName]["beta"]) do
            if beta then
                self.beta = Tensor(beta)
            end
        end
    else
        error("Can't load Data!.")
    end
end

-- 레이어 정보를 문자열로
function Dense:summary()
    local str = " [layer]"
    str = str..string.format("| Type: %10s |", "Dense")

    local parameterUnits = self.weights:__len() + self.gamma:__len() + self.beta:__len()
    str = str..string.format("| Parameter: %5d |", parameterUnits)

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
    str = str..string.format("| Input Shape: %10s |", inputShape_str)
    str = str..string.format("| Layer Shape: %10s |", layerShape_str)
    str = str..string.format("| Activation: %10s |", self.activationFunctionName)

    return str
end

function Model.new(name, inputShape, outputShape)
    local self = setmetatable({}, Model)

    self.name = name or "None"

    self.inputShape = inputShape -- table
    self.outputShape = outputShape -- table

    self.layers = {}

    self.layer = {dense = {}, convolution = {}, pooling = {}, attention = {}}
    self.layer.dense = setmetatable(self.layer.dense, {
        __call = function(_, inputShape, layerShape, activationFunctionName, normalized) -- Model.layer.dense() 식으로 호출
            table.insert(self.layers, Dense(inputShape, layerShape, activationFunctionName, normalized)) -- 레이어를 생성하고 빈공간에 넣는다.
            print("Dense layer generated")
        end
    })
    self.layer.convolution = setmetatable(self.layer.convolution, {
        __call = function(_,featureNum, inputShape, filterShape, filterNum, padding, stride, activationFunctionName) -- Model.layer.convolution() 식으로 호출
            table.insert(self.layers, Convolution(featureNum, inputShape, filterShape, filterNum, padding, stride, activationFunctionName)) -- 레이어를 생성하고 빈공간에 넣는다.
            print("Convolution layer generated")
        end
    })
    self.layer.pooling = setmetatable(self.layer.pooling, {
        __call = function(_, poolingNum, inputShape, poolingShape, padding, stride, poolingMethod) -- Model.layer.pooling() 식으로 호출
            table.insert(self.layers, Pooling(poolingNum, inputShape, poolingShape, padding, stride, poolingMethod)) -- 레이어를 생성하고 빈공간에 넣는다.
            print("Pooling layer generated")
        end
    })
    self.layer.attention = setmetatable(self.layer.attention, {
        __call = function(_, headNum, QShape, KShape, VShape) -- Model.layer.attention() 식으로 호출
            table.insert(self.layers, Attention(headNum, QShape, KShape, VShape)) -- 레이어를 생성하고 빈공간에 넣는다.
            print("attention layer generated")
        end
    })

    -- 손실 함수 모음 return은 오류값을 가진 Node
    self.lossFunction = {
        MSE = function(targetTensor, outputTensor)
            return ( Tensor.apply((targetTensor - outputTensor), function(x) return x * x end) ):sum()-- Node
        end,
        Gradient = function(targetTensor, outputTensor)
            return ( Tensor.apply((targetTensor * outputTensor), function(x) return x end) ):sum()-- Node
        end,
        CrossEntropy = function(targetTensor, outputTensor)
            return ( (targetTensor * Tensor.apply((outputTensor), function(x) return -Node.log(x,math.exp(1)) end)) ):sum()-- Node
        end
    }

    -- 데이터 저장
    self.data = Data()

    -- 모델의 출력값
    self.output = nil

    return self
end
setmetatable(Model, {
    __call = function(_, name, inputShape, outputShape)
        return Model.new(name, inputShape, outputShape)
    end
})

-- 모델 순전파
function Model:forwardPropagation(inputTensor)
    --local output = Tensor.deepcopy(inputTensor)
    local output = inputTensor
    for _, layer in ipairs(self.layers) do
        output = layer:forwardPropagation(output)
    end
    return output-- Tensor
end

function Model:backPropagation(learningRate, grad)
    learningRate = learningRate or 0.1
    --[[
    for _, layer in ipairs(self.layers) do
        grad = layer:backPropagation(learningRate, grad)
    end
    --]]
    for i = #self.layers, 1, -1 do
        local layer = self.layers[i]
        grad = layer:backPropagation(learningRate, grad)
    end
end

function Model:save(fileName)
    DataManager.saveTableToFile({}, fileName)
    for key, layer in ipairs(self.layers) do
        local layerName = "layer"..tostring(key)
        layer:save(fileName, layerName)
    end
end

function Model:load(fileName)
    for key, layer in ipairs(self.layers) do
        local layerName = "layer"..tostring(key)
        layer:load(fileName, layerName)
    end
end

-- 데이터를 직접 받아 하는 훈련
function Model:learn(inputTensor, targetTensor, loss, learningRate)
    loss = loss or 'MSE'
    learningRate = learningRate or 0.01

    local outputTensor = nil
    outputTensor = self:forwardPropagation(inputTensor)

    local batchDimension = 1
    if #inputTensor.size - #self.inputShape  == 1 then
        batchDimension = inputTensor.size[1]
    elseif #inputTensor.size - #self.inputShape  == 0 then
        batchDimension = 1
    else
        error("")
    end
    local errorTensor = Tensor.emptyTensor({batchDimension},0)
    for i = 1, batchDimension do
        errorTensor[i] = self.lossFunction[loss](targetTensor[i], outputTensor[i])
    end

    local loss = errorTensor:sum()/batchDimension
    loss:backward()-- dL_d[] 꼴로 역전파

    local grad = outputTensor:grad()
    self:backPropagation(learningRate, grad)

    --self.data:add(input, target)

    return loss
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
    local layerStrTable = {}
    for _, layer in ipairs(self.layers) do
        table.insert(layerStrTable, layer:summary().."\n")
    end

    local str = "\n".."[Model]"..string.rep("=", #layerStrTable[1]).."\n"
    str = str..string.format(" Name: %10s",self.name)

    local inputShape_str = "{ "
    for _, value in ipairs(self.inputShape) do
        inputShape_str = inputShape_str..value.." "
    end
    inputShape_str = inputShape_str.."}"
    local outputShape_str = "{ "
    for _, value in ipairs(self.outputShape) do
        outputShape_str = outputShape_str..value.." "
    end
    outputShape_str = outputShape_str.."}"
    str = str..string.format(" Input Shape: %10s",inputShape_str)
    str = str..string.format(" Output Shape: %10s\n",outputShape_str)

    str = str..string.rep("-", #layerStrTable[1]).."\n"

    for _, s in ipairs(layerStrTable) do
        str = str..s
    end

    str = str..string.rep("-", #layerStrTable[1]).."\n"
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