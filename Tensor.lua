Node = require("Node")

Tensor = {}
--Tensor.__index = Tensor

-- Tensor 생성
function Tensor.new(values)
    local self = setmetatable({}, Tensor)
    self.values = {}--Tensor의 값

    self.dimension = 0--Tensor의 차원
    self.size = {}--Tensor의 크기
    self.size = setmetatable(self.size, {
        __eq = function(a, b) -- 동등
            if #a == #b then
                for i = 1, #a do
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

--#region Tensor 메타메소드 구현부

-- Tensor.__call = function
-- setmetatable(Tensor, Tensor) 와 동일
Tensor = setmetatable(Tensor, {
    __call = function(_, values)
        return Tensor.new(values)
    end
})

Tensor.__index = function(self, index)
    -- 먼저 Tensor 클래스의 메서드(함수)가 있는지 확인
    local value = rawget(Tensor, index)
    if value then
        return value -- 클래스 함수 반환 (예: reshape, dot 등)
    end
    -- 인덱스가 숫자인 경우, `values`에서 값을 가져옴
    if type(index) == "number" then
        local values = self.values[index]
        if getmetatable(values) == Node then
            return Tensor({values})
        else
            return Tensor(values)
        end
    end
    -- 기본 __index 동작 수행
    return rawget(self, index)
end

Tensor.__newindex = function(self, index, value)
    -- 값이 Tensor 객체라면 내부 values를 저장
    if type(index) == "number" then
        if getmetatable(value) == Tensor then
            self.values[index] = value.values -- Tensor 내부 값만 저장
        elseif getmetatable(value) == Node then
            self.values[index] = value -- Node 객체 직접 저장
        elseif type(value) == "number" then
            self.values[index] = Node(value) -- 일반 숫자는 노드로 변환
        else
            self.values[index] = value --  그대로 저장
        end
    else
        -- 기본 동작 수행 (Tensor 자체 속성을 변경할 경우)
        rawset(self, index, value)
    end
end
-- __tostring 메타메소드 구현 : Tensor 출력
Tensor.__tostring = function(self)
    local str = "[Tensor]\n"
    local function tostring(table)
        for _, v in pairs(table) do
            if getmetatable(v) ~= Node and type(v) == "table" then
                tostring(v)  -- 하위 테이블을 재귀적으로 순회
                str = str.."\n"
            else
                str = str..string.format("%9.18s", v.value).." "
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

-- __len 메타메소드 구현 : Tensor의 전체 원소 개수 5.1에 없음
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

-- __mul 메타메소드 구현 : Tensor의 요소 나눗셈(스칼라 나눗셈 가능)
Tensor.__div = function(t1, t2)
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
                -- 테이블의 요소의 나눗셈을 할당
                subTable[i] = table1[i] / table2[i]
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
            -- 복소수 전치시 허수부 부호 변환
            transposed[i][j].value.imag = transposed[i][j].value.imag * -1
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

-- N*N크기의 이산 푸리에 변환 행렬 생성(1차원 이산 푸리에 변환)
function Tensor.DFT(N)
    local matrixDFT = {}
    local twoPi = 2*math.pi
    for i = 1, N do
        matrixDFT[i] = {}
        for j = 1, N do
            local real = math.cos((twoPi * ((i-1)*(j-1)/N)))
            local imag = -math.sin((twoPi * ((i-1)*(j-1)/N)))
            matrixDFT[i][j] = Node(real, imag)
        end
    end
    return Tensor(matrixDFT)
end

--#endregion
--#region Tensor 신경망 관련 기능 구현부

-- Tensor 각 요소에 함수 적용
function Tensor.apply(tensor, func)
    local function applyIn(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = applyIn(table[i], dimensions, depth + 1)
            else
                subTable[i] = func(table[i])
            end
        end
        return subTable
    end
    return Tensor(applyIn(tensor.values, tensor.size, 1))
end

-- 텐서 패딩 (텐서, {각 축에 대한 패딩 수치}, 패딩 값(기본 0))
function Tensor.padding(tensor, paddingToDimension, initial)
    initial = initial or 0
    local paddedTable = tensor
    for i, value in ipairs(paddingToDimension) do
        if value <= 0 then
            goto continue
        end
        local padTensorSize = paddedTable.size
        padTensorSize[i] = 1
        local padTensor = Tensor.emptyTensor(padTensorSize, initial)

        for j = 1, math.floor(value/2+(2/3)) do
            paddedTable = Tensor.concat(paddedTable, padTensor, i)
        end
        for j = 1, math.floor(value/2) do
            paddedTable = Tensor.concat(padTensor, paddedTable, i)
        end
        ::continue::
    end
    return Tensor(paddedTable.values)
end

function Tensor:slice(startIndexTable, endIndexTable)
    local dimension = self.dimension

    local function slice(tensor, dim)
        local startIndex = startIndexTable[dim]
        local endIndex = endIndexTable[dim]
        local result = tensor

        local slicedTable = {}
        for j = startIndex, endIndex do
            if dim < dimension then
                table.insert(slicedTable, slice(result[j], dim+1))
            else
                table.insert(slicedTable, result[j])
            end
        end
        return slicedTable
    end

    return Tensor(slice(self.values, 1))
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

-- 활성화 함수 모음
function Tensor.activation(name)
    local activationFunction = { -- 활성화 함수 모음
        ReLU = function(tensor) return Tensor.apply(tensor ,function(x)
            return (x > Node(0)) and (x) or (x * 0.01)
        end) end,
        Tanh = function(tensor) return Tensor.apply(tensor, function(x)
            return ( math.exp(1)^(2*x) - 1 ) / (math.exp(1)^(2*x) + 1)
        end) end,
        Sigmoid = function(tensor) return Tensor.apply(tensor, function(x)
            return ( 1 ) / (1 + math.exp(1)^(-x))
        end) end,
        SoftMax = function(tensor) local max = Tensor.max(tensor) local sum = (Tensor.apply(tensor, function(x) return math.exp(1)^(x-max) end):sum()) 
            return Tensor.apply(tensor, function(x)
                return (math.exp(1)^(x-max) / sum)
        end) end,
        Linear = function(tensor) return Tensor.apply(tensor, function(x)
            return x-- Node
        end) end
    }
    return activationFunction[name]
end

--#endregion
--#region Tensor 비연산 메서드 구현부

-- 반환값 Node
function Tensor.max(tensor)
    local max = nil
    local function findMax(table, dimensions, depth)
        local dim = dimensions[depth]
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀
                findMax(table[i], dimensions, depth + 1)
            else
                if max then
                    max = table[i] > max and table[i] or max
                else
                    max = table[i]
                end
            end
        end
    end
    findMax(tensor.values, tensor.size, 1)
    return max
end

function Tensor.toTable(tensor)
    local function _toTable(table, dimensions, depth)
        local dim = dimensions[depth]
        local subTable = {}
        for i = 1, dim do
            if depth < #dimensions then
                -- 하위 차원으로 재귀적으로 테이블 생성
                subTable[i] = _toTable(table[i], dimensions, depth + 1)
            else
                subTable[i] = table[i].value.real
            end
        end
        return subTable
    end
    return _toTable(tensor.values, tensor.size, 1)
end

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

-- Tensor 차원 순서 변경 (permute)
function Tensor:permute(...)
    local order = {...}
    if #order ~= self.dimension then
         error("Dimension mismatch: Expected " .. self.dimension .. " but got " .. #order, 2)
    end

    -- 새 텐서의 크기 계산: 새로운 각 차원의 크기는 기존 텐서의 해당 차원 크기
    local new_shape = {}
    for k = 1, self.dimension do
         new_shape[k] = self.size[order[k]]
    end

    -- 인덱스 리스트를 통해 self.values에서 값을 가져오는 헬퍼 함수
    local function get_value(t, indices)
         local ref = t
         for i = 1, #indices do
              ref = ref[indices[i]]
         end
         return ref
    end

    -- 재귀적으로 새 텐서 값을 채움.
    -- new_indices는 새 텐서의 다중 인덱스 (i₁, i₂, …, iₙ)를 담는 배열입니다.
    local function recursive_fill(new_indices, depth)
         if depth > self.dimension then
              -- new_indices는 새 텐서의 인덱스가 모두 채워진 상태.
              -- 원본 인덱스 배열 orig는 다음과 같이 구함:
              --   각 k (1~n)에 대해 orig[ order[k] ] = new_indices[k]
              local orig = {}
              for k = 1, self.dimension do
                   orig[ order[k] ] = new_indices[k]
              end
              -- 원본 인덱스는 순서대로 정렬되어야 하므로
              local ordered_orig = {}
              for i = 1, self.dimension do
                   ordered_orig[i] = orig[i]
              end
              return get_value(self.values, ordered_orig)
         else
              local arr = {}
              for i = 1, new_shape[depth] do
                   new_indices[depth] = i
                   arr[i] = recursive_fill(new_indices, depth+1)
              end
              return arr
         end
    end

    local new_values = recursive_fill({}, 1)
    return Tensor(new_values)
end

-- 상위 차원으로 합침 (여러개 순차적으로 하고싶으면 table.unpack해서 넣을 것)
function Tensor.stack(...)
    local tensors = {...}
    local stackedTensor = {}
    for _, tensor in pairs(tensors) do
        table.insert(stackedTensor, tensor.values)
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
                --v:backward_iterative(dz_real, dz_imag)
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