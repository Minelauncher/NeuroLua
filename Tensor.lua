Node = require("Node")

Tensor = {}

local size_mt = {
    __eq = function(a, b)
    if #a ~= #b then
        return false
    end
    for i = 1, #a do
        if a[i] ~= b[i] then
        return false
        end
    end
    return true
    end,

    __tostring = function(tbl)
    -- tbl[i]들을 문자열로 모아서 "[1, 2, 3]" 같은 형식으로
    local parts = {}
    for i = 1, #tbl do
        parts[#parts + 1] = tostring(tbl[i])
    end
    return "[" .. table.concat(parts, ", ") .. "]"
    end
}

-- Tensor 생성
function Tensor.new(values)
    local self = setmetatable({}, Tensor)
    self.values = {}--Tensor의 값

    self.dimension = 0--Tensor의 차원
    self.size = {}--Tensor의 크기
    self.size = setmetatable(self.size, size_mt)

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

-- LU 분해(부분 피벗 포함)를 수행하여 LU 결합 행렬과 피벗 순열을 반환
function Tensor:luDecompose()
    assert(self.dimension == 2, "Matrix must be 2D")
    local n = #self.values
    -- 원본 변경 방지를 위해 깊은 복사
    local luTensor = Tensor.deepcopy(self)
    local lu = luTensor.values
    -- 피벗 순열 초기화
    local P = {}
    for i = 1, n do P[i] = i end

    for k = 1, n do
        -- 피벗 선택: k번째 열에서 절댓값 최대 행 탐색
        local maxRow, maxVal = k, math.abs(lu[k][k].value and lu[k][k].value.real or lu[k][k])
        for i = k + 1, n do
            local v = math.abs(lu[i][k].value and lu[i][k].value.real or lu[i][k])
            if v > maxVal then maxVal, maxRow = v, i end
        end
        assert(maxVal > 0, "Singular matrix")
        -- 행 교환
        if maxRow ~= k then
            lu[k], lu[maxRow] = lu[maxRow], lu[k]
            P[k], P[maxRow] = P[maxRow], P[k]
        end
        -- L, U 계산
        for i = k + 1, n do
            local factor = lu[i][k] / lu[k][k]
            lu[i][k] = factor
            for j = k + 1, n do
                lu[i][j] = lu[i][j] - factor * lu[k][j]
            end
        end
    end

    return luTensor, P
end

-- 내부 헬퍼: LU 해 풀기(forward/backward substitution)
local function solveLU(luTensor, P, b)
    local n = #luTensor.values
    local lu = luTensor.values
    -- 순열 적용
    local pb = {}
    for i = 1, n do pb[i] = b[P[i]] end
    -- 전진 대입 L*y = pb
    local y = {}
    for i = 1, n do
        local sum = Node(0)
        for j = 1, i - 1 do sum = sum + lu[i][j] * y[j] end
        y[i] = pb[i] - sum
    end
    -- 후진 대입 U*x = y
    local x = {}
    for i = n, 1, -1 do
        local sum = Node(0)
        for j = i + 1, n do sum = sum + lu[i][j] * x[j] end
        x[i] = (y[i] - sum) / lu[i][i]
    end
    return x
end

-- LU 분해를 이용한 역행렬 계산
function Tensor:inverse()
    assert(self.dimension == 2, "Matrix must be 2D")
    local n = #self.values
    local luTensor, P = self:luDecompose()
    local inv = Tensor.emptyTensor({n, n})
    for j = 1, n do
        -- 단위벡터 생성
        local b = {}
        for i = 1, n do b[i] = (i == j) and Node(1) or Node(0) end
        local x = solveLU(luTensor, P, b)
        for i = 1, n do inv.values[i][j] = x[i] end
    end
    return inv
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

function Tensor:blockHankel(window_size)
    Tensor.isMatrix(self) -- 2차원인지 확인

    local T = self.size[1] -- 총 시점 수
    local d = self.size[2] -- 변수 수
    local block_rows = window_size * d
    local block_cols = T - window_size + 1

    local hankel = {}
    for i = 1, block_rows do
        hankel[i] = {}
        for j = 1, block_cols do
            hankel[i][j] = Node(0)
        end
    end

    for col = 1, block_cols do
        for t = 1, window_size do
            local row_offset = (t - 1) * d
            for var = 1, d do
                local row = row_offset + var
                hankel[row][col] = self.values[col + t - 1][var]
            end
        end
    end

    -- 열은 윈도우 크기 시간만큼의 특성차원 묶음
    return Tensor(hankel)
end

-- 정규화된 leastSquares 함수 예시
function Tensor.leastSquares(Y, Phi, lambda)
    Tensor.isMatrix(Y) -- size: (n × N)
    Tensor.isMatrix(Phi) -- size: ((n + m) × N)
    lambda = lambda or 1e-6
    Y = Y:detach()
    Phi = Phi:detach()

    local Phi_T = Phi:transpose()
    local Phi_PhiT = Tensor.dot(Phi, Phi_T)
    local I = Tensor.emptyTensor(Phi_PhiT.size):fill(0)
    for i = 1, #I.values do
        I.values[i][i] = Node(lambda)
    end
    local regularized = Phi_PhiT + I
    local inv = regularized:inverse()
    return Tensor.dot(Tensor.dot(Y, Phi_T), inv)
end

function Tensor.subspaceID(U, Y, r, window)
    Tensor.isMatrix(U)
    Tensor.isMatrix(Y)

    -- 블록 행렬 생성
    local Y_p = Y:blockHankel(window)
    local Y_f = Y:blockHankel(window):slice({r+1,1}, {Y:blockHankel(window).size[1], Y:blockHankel(window).size[2]})
    local U_p = U:blockHankel(window)
    
    -- 상태 X_t 를 Y_p 의 앞 r행만 사용 (SVD 생략 버전)
    local X_t = Y_p:slice({1,1}, {r, Y_p.size[2]})            -- (r × N)
    local X_tp1 = Y_f:slice({1,1}, {r, Y_f.size[2]})           -- (r × N)
    local U_cut = U_p:slice({1,1}, {U_p.size[1], X_t.size[2]}) -- (m×window) × N

    -- XU = [X_t; U_cut]
    local XU = Tensor.concat(X_t, U_cut, 1)

    -- AB = [A | B]
    local AB = Tensor.leastSquares(X_tp1, XU)

    -- CD = [C | D]
    local CD = Tensor.leastSquares(Y_p, XU)

    return AB, CD
end

-- Jacobi 고유값 분해 (대칭행렬용)
-- 입력: self = 대칭 Tensor (n x n)
-- 출력: eigenvalues (1D Tensor), eigenvectors (n x n Tensor)

function Tensor:eigJacobi(eps, max_iter)
    eps = eps or 1e-10
    max_iter = max_iter or 100
    local n = #self.values

    -- 초기화
    local A = Tensor.deepcopy(self)             -- A: 작업용 복사본
    local V = Tensor.emptyTensor({n, n}, 0)     -- 고유벡터 초기값: 단위행렬
    for i = 1, n do V.values[i][i] = Node(1) end

    for iter = 1, max_iter do
        -- 최대 오프대각 원소 찾기
        local max_val, p, q = 0, 1, 2
        for i = 1, n-1 do
            for j = i+1, n do
                local aij = math.abs(A.values[i][j].value.real)
                if aij > max_val then
                    max_val = aij
                    p, q = i, j
                end
            end
        end

        if max_val < eps then break end  -- 수렴 조건

        local app = A.values[p][p].value.real
        local aqq = A.values[q][q].value.real
        local apq = A.values[p][q].value.real

        local theta = 0.5 * math.atan2(2 * apq, aqq - app)
        local c = math.cos(theta)
        local s = math.sin(theta)

        -- 회전 적용
        for i = 1, n do
            local aip = A.values[i][p]
            local aiq = A.values[i][q]
            A.values[i][p] = c * aip - s * aiq
            A.values[i][q] = s * aip + c * aiq
        end
        for i = 1, n do
            local api = A.values[p][i]
            local aqi = A.values[q][i]
            A.values[p][i] = c * api - s * aqi
            A.values[q][i] = s * api + c * aqi
        end

        -- 대각 원소 갱신
        local new_app = c*c*app - 2*s*c*apq + s*s*aqq
        local new_aqq = s*s*app + 2*s*c*apq + c*c*aqq
        A.values[p][p] = Node(new_app)
        A.values[q][q] = Node(new_aqq)
        A.values[p][q] = Node(0)
        A.values[q][p] = Node(0)

        -- 고유벡터 갱신
        for i = 1, n do
            local vip = V.values[i][p]
            local viq = V.values[i][q]
            V.values[i][p] = c * vip - s * viq
            V.values[i][q] = s * vip + c * viq
        end
    end

    -- 고유값 추출 (대각 성분)
    local eigvals = {}
    for i = 1, n do
        eigvals[i] = A.values[i][i]
    end
    return Tensor({eigvals}), V
end

-- SVD 기반 상태 추정 포함 서브스페이스 식별
function Tensor.subspaceID_SVD(U, Y, r, window)
    -- Step 1: blockHankel
    local Y_f = Y:blockHankel(window)

    -- Step 2: A = Yf^T * Yf
    local Y_f_T = Y_f:transpose()
    local A = Tensor.dot(Y_f_T, Y_f)

    -- Step 3: eig(A) → V, Lambda
    local S_squared, V = A:eigJacobi(1e-8, 100)

    -- Step 4: Sigma = sqrt(S_squared), keep top r
    local sigma_values = {}
    for i = 1, r do
        local raw_val = S_squared.values[1][i].value.real
        if raw_val < 1e-8 then raw_val = 1e-8 end
        sigma_values[i] = Node(math.sqrt(raw_val))
    end
    local Sigma_r = Tensor.emptyTensor({r, r}, 0)
    for i = 1, r do
        Sigma_r.values[i][i] = sigma_values[i]  -- Node 타입
    end

    -- Step 5: X_t = Sigma_r * V_r^T
    local V_r = {}
    for i = 1, #V.values do
        V_r[i] = {}
        for j = 1, r do
            V_r[i][j] = V.values[i][j]
        end
    end
    local V_r_T = Tensor(V_r):transpose()

    local X_t = Tensor.dot(Sigma_r, V_r_T)

    -- Step 6: 미래 상태용 X_t+1 = X_t[:,2:] 등 필요시 자를 것
    local X_tp1 = X_t:slice({1,2}, {X_t.size[1], X_t.size[2]})
    local X_now = X_t:slice({1,1}, {X_t.size[1], X_t.size[2]-1})

    -- Step 7: blockHankel for U, Y (맞춰 자르기)
    local U_p = U:blockHankel(window):slice({1,1}, {U:blockHankel(window).size[1], X_now.size[2]})
    local Y_p = Y:blockHankel(window):slice({1,1}, {Y:blockHankel(window).size[1], X_now.size[2]})

    local XU = Tensor.concat(X_now, U_p, 1)

    local AB = Tensor.leastSquares(X_tp1, XU)
    local CD = Tensor.leastSquares(Y_p, XU)

    -- AB, CD 크기 분해 계산
    local AB_cols = AB.size[2]
    local CD_cols = CD.size[2]

    -- AB 쪼개기
    local A = AB:slice({1,1}, {r, r})
    local B = AB:slice({1, r+1}, {r, AB_cols})

    -- CD 쪼개기
    local C = CD:slice({1,1}, {CD.size[1], r})
    local D = CD:slice({1, r+1}, {CD.size[1], CD_cols})

    return A, B, C, D
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

function Tensor:backwardTensor(gradiantTensor)
    -- 현재 텐서와 크기가 같다고 가정하는 그라디언트 텐서
    local gradiantTensor = gradiantTensor or Tensor.emptyTensor(self.size, 1)

    local function traverse(table, gradiantTable)
        for t, gt in zip(table, gradiantTable) do
            local t_isNotNode = (getmetatable(t) ~= Node and type(t) == "table")
            local gt_isNotNode = (getmetatable(gt) ~= Node and type(gt) == "table")
            if t_isNotNode and gt_isNotNode then
                traverse(t, gt)  -- 하위 테이블을 재귀적으로 순회
            else
                t:backward(gt.value.real, gt.value.imag)
                --t:backward_iterative(gt.value.real, gt.value.imag)
            end
        end
    end
    traverse(self.values, gradiantTensor.values)
    return true
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

-- 새 텐서(leaf 노드 복사본) 반환
function Tensor:detach()
    local function cloneDet(t)
        local out = {}
        for k,v in pairs(t) do
            if getmetatable(v) == Node then           
                out[k] = v:detach()
            elseif type(v) == "table" then            
                out[k] = cloneDet(v)
            else                                      
                out[k] = v
            end
        end
        return out
    end
    return Tensor(cloneDet(self.values))
end

-- 제자리(in-place) detach
function Tensor:detach_()
    local function inplaceDet(t)
        for k,v in pairs(t) do
            if getmetatable(v) == Node then           
                v:detach_()
            elseif type(v) == "table" then            
                inplaceDet(v)
            end
        end
    end
    inplaceDet(self.values)
    return self
end
--#endregion

return Tensor