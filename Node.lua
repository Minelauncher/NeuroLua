require("functions")

local valueMT = {
    __tostring = function(self) -- value 출력
        local str = ""
        str = str..string.format("(%6.3f)", self.real)
        return str
    end,
    __eq = function(a, b) -- 동등
        return a.real == b.real
    end,
    __lt = function(a, b) -- 미만
        return a.real < b.real
    end,
    __le = function(a, b) -- 이하
        return a.real <= b.real
    end
}

local gradMT = {
    __tostring = function(self) 
        local str = ""
        str = str..string.format("(%6.3f)", self.real)
        return str
    end,
    __eq = function(a, b) -- 동등
        return a.real == b.real
    end,
    __lt = function(a, b) -- 미만
        return a.real < b.real
    end,
    __le = function(a, b) -- 이하
        return a.real <= b.real
    end
}

local Node = {}
Node.__index = Node

function Node.new(real, imag, grad_fn)
    local self = setmetatable({}, Node)

    self.value = {real = real or 0, imag = imag or 0} -- 노드의 실수 출력 값, 노드의 허수 출력 값
    self.value = setmetatable(self.value, valueMT)

    self.grad = {real = 0, imag = 0} -- 그래디언트 
    self.grad = setmetatable(self.grad, gradMT)

    self.grad_fn = grad_fn or nil-- 역전파 함수

    self.parents = {}    -- 부모 노드 (입력)
    self.add_parent = function(parent) -- 노드에 부모 추가
        table.insert(self.parents, parent)
    end

    return self
end

--#region Node 메타테이블 구현부
setmetatable(Node, {
    __call = function(self, real, imag, grad_fn)-- 선언 방식 변경
        return Node.new(real, imag, grad_fn)
    end
})

Node.__index = function(table, index)
    local value = Node[index]
    if type(value) == 'function' then
        -- function table:func 처리
        return value
    else
        -- 기본 __index 동작
        return rawget(table, index)
    end
end

Node.__newindex = function(table, index, value)
    -- 기본 __newindex 동작
    rawset(table, index, value)
end

Node.__tostring = function(self) -- value 출력
    local str = ""
    str = str..string.format("[Node]\n v: [%s] g: [%s]", self.value, self.grad)
    return str
end
Node.__eq = function(a, b) -- 동등
    a, b = Node.convertToNode(a, b)
    return a.value == b.value
end
Node.__lt = function(a, b) -- 미만
    a, b = Node.convertToNode(a, b)
    return a.value < b.value
end
Node.__le = function(a, b) -- 이하
    a, b = Node.convertToNode(a, b)
    return a.value <= b.value
end
--#endregion

--#region Node 연산 구현부

-- 덧셈 노드 생성 x+y
Node.__add = function(x, y)
    x, y = Node.convertToNode(x, y)
    local a = x.value.real
    local b = x.value.imag
    local c = y.value.real
    local d = y.value.imag

    local real = a + c
    local imag = b + d
    local z = Node(real, imag)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real, dz_imag)
        y:backward(dz_real, dz_imag)
    end
    return z
end

-- 뺄셈 노드 생성 x-y
Node.__sub = function (x, y)
    x, y = Node.convertToNode(x, y)
    local a = x.value.real
    local b = x.value.imag
    local c = y.value.real
    local d = y.value.imag

    local real = a - c
    local imag = b - d
    local z = Node(real, imag)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real, dz_imag)
        y:backward(-dz_real, -dz_imag)
    end
    return z
end

-- 곱셈 노드 생성 x*y
Node.__mul = function(x, y)
    x, y = Node.convertToNode(x, y)
    local a = x.value.real
    local b = x.value.imag
    local c = y.value.real
    local d = y.value.imag

    local real = (a * c - b * d)
    local imag = (a * d + b * c)
    local z = Node(real, imag)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real * c, dz_imag * d)
        y:backward(dz_real * a, dz_imag * b)
    end
    return z
end

-- 나눗셈 노드 생성 x/y
Node.__div = function(x, y)
    x, y = Node.convertToNode(x, y)
    local a = x.value.real
    local b = x.value.imag
    local c = y.value.real
    local d = y.value.imag

    local real = (a * c + b * d) / (c^2 + d^2)
    local imag = (b * c - a * d) / (c^2 + d^2)
    local z = Node(real, imag) 
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real * (c) / (c^2 + d^2), dz_imag * -(d) / (c^2 + d^2))
        y:backward(dz_real * -a * (c^2 - d^2) / ((c^2 - d^2)^2 + 4 * c^2 * d^2), dz_imag * -(b * (c^2 - d^2)) / ((c^2 - d^2)^2 + 4 * c^2 * d^2))
    end
    return z
end

Node.__unm = function(self)
    return -1 * self
end

-- 로그 노드 생성 log_y_(x)
function Node.log(x, y)
    x, y = Node.convertToNode(x, y)

    local real = math.log(x.value.real ,y.value.real)

    local z = Node(real, 0)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real * (1 / (x.value.real * math.log(y.value.real))))
        y:backward(dz_real * -(math.log(x.value.real) / (y.value.real * math.log(y.value.real)^2)))
    end
    return z
end

-- 지수 노드 생성: x^y
Node.__pow = function(x, y)
    x, y = Node.convertToNode(x, y)

    local real = x.value.real^y.value.real
    local z = Node(real, 0)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real * y.value.real * x.value.real^(y.value.real - 1))
        y:backward(dz_real * x.value.real^y.value.real * math.log(x.value.real))
    end
    return z
end

--#endregion

--#region Node 비연산 구현부

-- 역전파 함수
function Node:backward(dz_real, dz_imag)
    local dz_real = dz_real or 1
    local dz_imag = dz_imag or 1
    self.grad.real = self.grad.real + dz_real
    self.grad.imag = self.grad.imag + dz_imag
    if self.grad_fn then
        self.grad_fn(dz_real, dz_imag)
    end
end

-- 숫자라면 Node로 변경
function Node.convertToNode(...)
    local args = {...}
    for key, arg in pairs(args) do
        if type(arg) == "number" then 
            args[key] = Node(arg) 
        end
    end
    return unpack(args)
end

--#endregion

return Node