local Node = require("Node")

Tensor = {}
Tensor.__index = Tensor

-- Tensor 생성
function Tensor.new(values)
    local self = setmetatable({}, Tensor)
    self.values = nil--Tensor의 값
    self.dimension = 0--Tensor의 차원
    self.size = {}--Tensor의 크기
    self.size = setmetatable(self.size, {
        __eq = function(a, b) -- 동등
            if #a.size == #b.size then
                for i = 1, #a.size do
                    if a[i] ~= b[i] then
                        return false
                    end
                end
                return true
            else
                return false
            end
        end
    })

    local function tableToNode(table, dimension)
        for i, v in pairs(table) do
            if type(v) == "table" and getmetatable(v) ~= Node then
                self.dimension = math.max(self.dimension, dimension)
                self.size[self.dimension] = #table
                tableToNode(v,dimension + 1)  -- 하위 테이블을 재귀적으로 순회
            elseif type(v) == "table" and getmetatable(v) == Node then
                self.dimension = math.max(self.dimension, dimension)
                self.size[self.dimension] = #table
                table[i] = v -- 원소를 Node로 변경하지 않고 대입
            else
                self.dimension = math.max(self.dimension, dimension)
                self.size[self.dimension] = #table
                table[i] = Node(v) -- 원소를 Node로 변경
            end
        end
    end
    tableToNode(values, 1)

    self.values = values
    return self
end
Tensor = setmetatable(Tensor, {
    __call = function(_, values)
        return Tensor.new(values)
    end
})

--#region Tensor 메타메소드 구현부

-- Tensor[][][] 식으로 values 내부의 값 호출가능
--[[
Tensor.__index = function(self, index)
    -- tensor[i1][i2][i3] 형태의 접근을 지원
    local value = self.values[index]
    if type(value) == "table" and getmetatable(value) ~= Node then
        return setmetatable({values = value} , getmetatable(self))
    elseif type(value) == "table" and getmetatable(value) == Node then
        return value.value
    else
        return Tensor -- 아무것도 아니면 Tensor.__index = Tensor
    end
end

--Tensor.__newindex = function()
end
--]]

-- __tostring 메타메소드 구현 : Tensor 출력
Tensor.__tostring = function(self)
    local str = "[Tensor]\n"
    local function tostring(table)
        for _, v in pairs(table) do
            if getmetatable(v) ~= Node and type(v) == "table" then
                tostring(v)  -- 하위 테이블을 재귀적으로 순회
                str = str.."\n"
            else
                str = str..string.format("%9.9s", v.value).." "
            end
        end
    end
    tostring(self.values)

    str = str..string.format("Dimension: %d Size: ", self.dimension)
    for _, value in ipairs(self.size) do
        str = str..string.format("%d ", value)
    end
    return str
end
-- __len 메타메소드 구현 : Tensor의 전체 원소 개수 
Tensor.__len = function(self)
    local length = 1
    for _, s in pairs(self.size) do
        length = length * s
    end
    return length
end

-- __add 메타메소드 구현 : 행렬 합(서로 크기가 같아야만 합 가능)(스칼라 합 가능)
Tensor.__add = function(t1, t2)
    if getmetatable(t1) ~= Tensor and getmetatable(t2) == Tensor then--m1이 스칼라고 m2가 텐서라면
        t1 = Tensor.deepcopy(t2):fill(t1)
    elseif getmetatable(t1) == Tensor and getmetatable(t2) ~= Tensor then--m2이 스칼라고 m1이 텐서라면
        t2 = Tensor.deepcopy(t1):fill(t2)
    end

    local function add(table1, table2, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = add(table1[i], table2[i], dimensions, depth + 1)
            else
                -- 테이블의 요소의 합을 할당
                subTable[i] = table1[i] + table2[i]
            end
        end
        return subTable
    end
    local t3 = add(t1.values, t2.values, t1.size, 1)
    return Tensor(t3)
end

-- __sub 메타메소드 구현 : 행렬 차(서로 크기가 같아야만 합 가능)(스칼라 차 가능)
Tensor.__sub = function(t1, t2)
    if getmetatable(t1) ~= Tensor and getmetatable(t2) == Tensor then--m1이 스칼라고 m2가 텐서라면
        t1 = Tensor.deepcopy(t2):fill(t1)
    elseif getmetatable(t1) == Tensor and getmetatable(t2) ~= Tensor then--m2이 스칼라고 m1이 텐서라면
        t2 = Tensor.deepcopy(t1):fill(t2)
    end

    local function sub(table1, table2, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = sub(table1[i], table2[i], dimensions, depth + 1)
            else
                -- 테이블의 요소의 차를 할당
                subTable[i] = table1[i] - table2[i]
            end
        end
        return subTable
    end
    local t3 = sub(t1.values, t2.values, t1.size, 1)
    return Tensor(t3)
end

-- __mul 메타메소드 구현 : Tensor의 요소 곱(스칼라 곱 가능)
Tensor.__mul = function(t1, t2)
    if getmetatable(t1) ~= Tensor and getmetatable(t2) == Tensor then--m1이 스칼라고 m2가 텐서라면
        t1 = Tensor.deepcopy(t2):fill(t1)
    elseif getmetatable(t1) == Tensor and getmetatable(t2) ~= Tensor then--m2이 스칼라고 m1이 텐서라면
        t2 = Tensor.deepcopy(t1):fill(t2)
    end

    local function mul(table1, table2, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = mul(table1[i], table2[i], dimensions, depth + 1)
            else
                -- 테이블의 요소의 곱을 할당
                subTable[i] = table1[i] * table2[i]
            end
        end
        return subTable
    end
    local t3 = mul(t1.values, t2.values, t1.size, 1)
    return Tensor(t3)
end
--#endregion
--#region Tensor 행렬 연산 구현부

-- Tensor가 행렬인지 검사
function Tensor.isMatrix(t1)
    if t1.dimension == 2 then
        return true
    else
        error("Tensor isn't Matrix",2)
        return false
    end
end

-- Tensor의 행렬곱(Tensor를 행렬화 후 계산 필요)
function Tensor.dot(t1, t2)
    if not Tensor.isMatrix(t1) and not Tensor.isMatrix(t2) then
        return
    end

    local m3 = {}
    for i = 1, #t1.values do
        m3[i] = {}
        for j = 1, #t2.values[#t1.values[i]] do
            local sum = 0
            for k = 1, #t1.values[i] do
                sum = sum + t1.values[i][k] * t2.values[k][j]
            end
            m3[i][j] = sum
        end
    end
    return Tensor(m3)
end

-- 전치함수
function Tensor:transpose()
    if not Tensor.isMatrix(self) then
        return
    end

    local transposed = {}
    for i = 1, #self.values[1] do      
        transposed[i] = {}
        for j = 1, #self.values do
            transposed[i][j] = self.values[j][i]
        end
    end
    return Tensor(transposed)
end

-- 소행렬을 반환하는 함수
function Tensor:minor(i, j)
    if not Tensor.isMatrix(self) then
        return
    end

    local minor = {}
    for row = 1, #self.values do
        if row ~= i then
            local minor_row = {}
            for col = 1, #self.values[row] do
                if col ~= j then
                    table.insert(minor_row, self.values[row][col])
                end
            end
            table.insert(minor, minor_row)
        end
    end
    return Tensor(minor)
end

-- 행렬식을 계산하는 함수(정사각행렬만 가능)
function Tensor:determinant()
    if not Tensor.isMatrix(self) then
        return
    end

    local tempTensor = Tensor.deepcopy(self)--깊은 복사를 통한 Node 연산 추적에 영향을 주지 않기
    if #tempTensor.values == 2 then
        return (tempTensor.values[1][1] * tempTensor.values[2][2] - tempTensor.values[1][2] * tempTensor.values[2][1])
    elseif #tempTensor.values == 1 and type(tempTensor.values[1]) == "table" and getmetatable(tempTensor.values[1]) ~= Node then
        return tempTensor.values[1][1]
    elseif #tempTensor.values == 1 and type(tempTensor.values[1]) == "table" and getmetatable(tempTensor.values[1]) == Node then
        return tempTensor.values[1]
    else
        local det = 0
        for j = 1, #tempTensor.values do
            det = det + (-1) ^ (1 + j) * tempTensor.values[1][j] * tempTensor:minor(1, j):determinant()
        end
        return det
    end
end

-- 수반 행렬(Adjugate Matrix) 계산(정사각행렬만 가능)
function Tensor:adjugate()
    if not Tensor.isMatrix(self) then
        return
    end

    local adjugateMatrix = Tensor.deepcopy(self)
    local n = #adjugateMatrix.values
    local adjugate = {}

    for i = 1, n do
        adjugate[i] = {}
        for j = 1, n do
            local minor = adjugateMatrix:minor(i, j)
            local cofactor = minor:determinant()
            adjugate[i][j] = cofactor * (-1)^(i + j)
        end
    end

    return Tensor(adjugate):transpose()  -- 여인수 행렬의 전치 행렬을 반환
end
-- 역행렬 계산(정사각 행렬만 가능, 행렬식 0 아니여야함)
function Tensor:inverse()
    if not Tensor.isMatrix(self) then
        return
    end

    local inverseMatrix = Tensor.deepcopy(self)
    local det = inverseMatrix:determinant()--Node

    if #inverseMatrix.values ~= 1 then
        local adjugate = inverseMatrix:adjugate()--Tensor
        return adjugate * (1 / det.value.real)
    else
        return (1 / det.value.real)
    end
end

--#endregion
--#region Tensor 신경망 관련 기능 구현부

-- Tensor 각 요소에 함수 적용
function Tensor.apply(tensor, func)
    local function apply(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = apply(table[i], dimensions, depth + 1)
            else
                subTable[i] = func(table[i])
            end
        end
        return subTable
    end
    return Tensor(apply(tensor.values, tensor.size, 1))
end

-- Tensor에 padding을 하여 크기 성장
function Tensor.padding(tensor, padding)
    local flatTensor = tensor:reshape(1, #tensor).values[1]
    local paddingTensorSize = {}
    for i = 1, #tensor.size do
       paddingTensorSize[i] = tensor.size[i] + 2 * padding
    end
    local index = {}-- 크기가 패딩텐서 차원과 같으면서 요소가 전부 1인 테이블
    for i = 1, #paddingTensorSize do
        index[i] = 1
    end
    local function create(dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = create(dimensions, depth + 1)
            else
                local function is_index_in_padding()-- 인덱스가 패딩 구간안에 존재하는지
                    local bool = false
                    for j = 1, #index do
                        if index[j] <= padding or index[j] > dimensions[j] - padding then-- 인덱스가 패딩구간이면
                            bool = true
                        end
                    end
                    return bool
                end
                if is_index_in_padding() then
                    subTable[i] = Node(0) -- 제로 패딩
                else-- 인덱스가 패딩구간이 아니라면 기존 텐서 값 대입
                    subTable[i] = table.remove(flatTensor, 1) -- 1차원화 된 텐서 요소 하나씩 대입
                end
                
                if index[#index] == dim then-- 맨 끝 인덱스가 현 차원 끝이면
                    for j = 1, #index do
                        if index[#index - j] ~= dimensions[depth - (j)] then-- 인덱스 중에 끝부터 검사해서  최대차원 아닌거 1 증가
                            index[#index - j] = index[#index - j] + 1
                            index[#index] = 1-- 맨 끝 인덱스 초기화
                            break-- 인덱스 증가했으므로 탈출
                        elseif index[#index - j] == dimensions[depth - (j)] then
                            index[#index - j] = 1 --혹시 최대값 걸리면 초기화
                        end
                    end
                else
                    index[#index] = index[#index] + 1-- 맨 끝 인덱스가 현차원 끝이 아니면 1증가
                end
            end
        end
        return subTable
    end
    return Tensor(create(paddingTensorSize, 1))
end

-- 시작인덱스부터 끝 인덱스까지 텐서의 일부를 추출
function Tensor:slice(startIndex, endIndex)
    if #startIndex ~= #endIndex then
        error("startIndex's len and endIndex's len are not same")
    end
    local beginIndex = {}-- 모든 요소가 finishIndex보다 작은 출발점 (인덱스 비교 용이를 위함)
    local finishIndex = {}-- 모든 요소가 beginIndex보다 큰 종말점 (인덱스 비교 용이를 위함)

    local sliceTensor = {}-- 추출할 텐서
    local sliceTensorShape = {}-- 추출할 텐서 크기
    for i = 1, #startIndex do
        sliceTensorShape[i] = math.abs(endIndex[i] - startIndex[i]) + 1
        beginIndex[i] = math.min(endIndex[i] ,startIndex[i])
        finishIndex[i] = math.max(endIndex[i] ,startIndex[i])
    end

    local index = {}
    for i = 1, self.dimension do-- 인덱스 초기화
        index[i] = 1
    end
    local count = 1-- 1부터 시작해야하기 때문
    local function create(table, dimensions, depth)
        local dim = dimensions[depth]
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                create(table[i], dimensions, depth + 1)
            else
                local function is_index_in()-- 인덱스가 추출 구간안에 존재하는지
                    local bool = true
                    for j = 1, #index do
                        if index[j] < beginIndex[j] or index[j] > finishIndex[j] then-- 인덱스가 추출구간이 한번이라도 아니라면
                            bool = false
                        end
                    end
                    return bool
                end
                if is_index_in() then-- 범위 안에 있다면 요소 삽입
                    sliceTensor[count] = table[i]
                    count = count + 1
                end
                
                if index[#index] == dim then-- 맨 끝 인덱스가 현 차원 끝이면
                    for j = 1, #index do
                        if index[#index - j] ~= dimensions[depth - (j)] then-- 인덱스 중에 끝부터 검사해서  최대차원 아닌거 1 증가
                            index[#index - j] = index[#index - j] + 1
                            index[#index] = 1-- 맨 끝 인덱스 초기화
                            break-- 인덱스 증가했으므로 탈출
                        elseif index[#index - j] == dimensions[depth - (j)] then
                            index[#index - j] = 1 --혹시 최대값 걸리면 초기화
                        end
                    end
                else
                    index[#index] = index[#index] + 1-- 맨 끝 인덱스가 현차원 끝이 아니면 1증가
                end
            end
        end
        return sliceTensor
    end
    create(self.values, self.size, 1)
    return Tensor(sliceTensor):reshape(sliceTensorShape)
end

-- Tensor 요소 총 합
function Tensor:sum()
    local result = 0
    local function sum(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = sum(table[i], dimensions, depth + 1)
            else
                result = result + table[i]
            end
        end
        return subTable
    end
    sum(self.values, self.size, 1)
    return result
end

-- Tensor SoftMax 구현 --maybe TODO (분모 연산을 테이블 분리해서 해야할수도)
function Tensor.SoftMax(tensor)
    local e = math.exp(1)
    local denominator = 0
    local function denominator_SoftMax(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = denominator_SoftMax(table[i], dimensions, depth + 1)
            else
                denominator = denominator + e^table[i]
            end
        end
        return subTable
    end
    denominator_SoftMax(tensor.values, tensor.size, 1)
    local function SoftMax(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = SoftMax(table[i], dimensions, depth + 1)
            else
                subTable[i] = e^table[i] / denominator
            end
        end
        return subTable
    end
    return Tensor(SoftMax(tensor.values, tensor.size, 1))
end
--#endregion
--#region Tensor 비연산 메서드 구현부

-- Tensor의 형태 변경 Tensor:reshape({0,0,0}) or Tensor:reshape(0,0,0)
function Tensor:reshape(...)
        -- Tensor 평탄화
    local function flatten()
        local tensor1D = {}
        local length = 0
        local function _flatten(table)
            for _, v in pairs(table) do
                if getmetatable(v) ~= Node and type(v) == "table" then
                    _flatten(v)  -- 하위 테이블을 재귀적으로 순회
                elseif getmetatable(v) == Node and type(v) == "table" then
                    length = length + 1
                    tensor1D[length] = v
                end
            end
        end
        _flatten(self.values)
        return tensor1D
    end

    local shape = {...}
    local flattenTensor = flatten()

    local function _reshape(flatTable, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = _reshape(flatTable, dimensions, depth + 1)
            else
                -- 1차원 테이블의 요소를 할당
                subTable[i] = table.remove(flatTable, 1)
            end
        end
        return subTable
    end

    local reshapedTensor
    if type(shape[1]) == "table" then
        reshapedTensor = _reshape(flattenTensor, shape[1], 1)
    else
        reshapedTensor = _reshape(flattenTensor, shape, 1)
    end
    return Tensor(reshapedTensor)
end

-- 상위 차원으로 합침 (여러개 순차적으로 하고싶으면 table.unpack해서 넣을 것)
function Tensor.stack(...)
    local tensors = {...}
    local stackedTensor = {}
    for _, tensor in pairs(tensors) do
        table.insert(stackedTensor, 1, tensor.values)
    end
    return Tensor(stackedTensor)
end

function Tensor.concat(t1, t2, axis)
    local function concat(tensor1, tensor2, size, depth)
        local dimensions = size[depth]
        local tempTensor = {}
        for i = 1, dimensions do
            if depth ~= axis then
                tempTensor[i] = concat(tensor1[i], tensor2[i], size, depth + 1)
            else
                for _, value in pairs(tensor1) do
                    table.insert(tempTensor, value)
                end
                for _, value in pairs(tensor2) do
                    table.insert(tempTensor, value)
                end
                break
            end
        end
        return tempTensor
    end
    return Tensor(concat(t1.values, t2.values, t1.size, 1))
end

-- Tensor 값을 하나로 통일하여 할당(연산 도중에 쓰는 것은 지양)
function Tensor:fill(value)
    local function fill(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = fill(table[i], dimensions, depth + 1)
            else
                -- 테이블의 요소 할당
                if getmetatable(value) == Node then
                    subTable[i] = value
                else
                    subTable[i] = Node(value)
                end
            end
        end
        return subTable
    end
    local t3 = fill(self.values, self.size, 1)
    return Tensor(t3)
end

-- Tensor에서 Node의 연산에 영향을 주지 않기 위해 데이터 가공에서 발생하는 연산은 깊은 복사를 진행하여 한다.
function Tensor.deepcopy(t1)
    local function copy(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = copy(table[i], dimensions, depth + 1)
            else
                -- 테이블의 요소의 합을 할당
                subTable[i] = Node(table[i].value.real)
            end
        end
        return subTable
    end
    local t2 = copy(t1.values, t1.size, 1)
    return Tensor(t2)
end

-- 원하는 크기의 빈 Tensor 생성
function Tensor.emptyTensor(size, initial)
    initial = initial or 0
    local function create(dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = create(dimensions, depth + 1)
            else
                subTable[i] = Node(initial)
            end
        end
        return subTable
    end
    return Tensor(create(size, 1))
end

-- 특정 Tensor 기준으로 자동 미분
function Tensor:backward(dz_real, dz_imag)
    local dz_real = dz_real or 1
    local dz_imag = dz_imag or 1
    local function traverse(table)
        for _, v in pairs(table) do
            if getmetatable(v) ~= Node and type(v) == "table" then
                traverse(v)  -- 하위 테이블을 재귀적으로 순회
            else
                v:backward(dz_real, dz_imag)
            end
        end
    end
    traverse(self.values)
    return true
end

-- Tensor 전체 요소의 기울기를 출력 backward후에 기울기 반영
function Tensor:grad()
    local function grad(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = grad(table[i], dimensions, depth + 1)
            else
                -- 테이블의 요소의 합을 할당
                subTable[i] = Node(table[i].grad.real)
                table[i] = Node(table[i].value.real)-- 노드 초기화 (따라서 기울기는 한번 쓰면 날아간다)
            end
        end
        return subTable
    end
    local gradiantTensor = grad(self.values, self.size, 1)
    return Tensor(gradiantTensor)
end
--#endregion

return Tensor