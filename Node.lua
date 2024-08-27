local Node = {}
Node.__index = Node

function Node.new(real, imag, grad_fn)
    local self = setmetatable({}, Node)

    self.value = {real = real or 0, imag = imag or 0} -- 노드의 실수 출력 값, 노드의 허수 출력 값
    self.value = setmetatable(self.value, {
        __tostring = function() -- value 출력
            local str = ""
            str = str..string.format("(%6.3f)", self.value.real)
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
    })

    self.grad = {real = 0, imag = 0} -- 그래디언트 
    self.grad = setmetatable(self.grad, {
        __tostring = function() 
            local str = ""
            str = str..string.format("(%6.3f)", self.grad.real)
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
    })

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
Node.__tostring = function(self) -- value 출력
    local str = ""
    str = str..string.format("[Node]\n v: (%6.3f) g: (%6.3f)", self.value.real, self.grad.real)
    return str
end
Node.__eq = function(a, b) -- 동등
    a, b = Node.if_isNode_then_change(a, b)
    return a.value == b.value
end
Node.__lt = function(a, b) -- 미만
    a, b = Node.if_isNode_then_change(a, b)
    return a.value < b.value
end
Node.__le = function(a, b) -- 이하
    a, b = Node.if_isNode_then_change(a, b)
    return a.value <= b.value
end
--#endregion

--#region Node 연산 구현부

-- 덧셈 노드 생성 x+y
Node.__add = function(x, y)
    x, y = Node.if_isNode_then_change(x, y)

    local real = x.value.real + y.value.real
    local z = Node(real, imag)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real)
        y:backward(dz_real)
    end
    return z
end

-- 뺄셈 노드 생성 x-y
Node.__sub = function (x, y)
    x, y = Node.if_isNode_then_change(x, y)

    local real = x.value.real - y.value.real
    local z = Node(real, imag)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real)
        y:backward(-dz_real)
    end
    return z
end

-- 곱셈 노드 생성 x*y
Node.__mul = function(x, y)
    x, y = Node.if_isNode_then_change(x, y)

    local z = Node(x.value.real * y.value.real)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real * y.value.real)
        y:backward(dz_real * x.value.real)
    end
    return z
end

-- 나눗셈 노드 생성 x/y
Node.__div = function(x, y)
    x, y = Node.if_isNode_then_change(x, y)

    local real = x.value.real / y.value.real
    local z = Node(real, imag) 
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real * 1 / y.value.real)
        y:backward(dz_real * x.value.real * -1 * y.value.real^-2)
    end
    return z
end

Node.__unm = function(self)
    return -1 * self
end

-- 로그 노드 생성 log_y_(x)
function Node.log(x, y)
    x, y = Node.if_isNode_then_change(x, y)

    local real = math.log(x.value.real ,y.value.real)

    local z = Node(real, imag)
    z.grad_fn = function(dz_real, dz_imag)
        x:backward(dz_real * (1 / (x.value.real * math.log(y.value.real))))
        y:backward(dz_real * -(math.log(x.value.real) / y.value.real * math.log(y.value.real)^2))
    end
    return z
end

-- 지수 노드 생성: x^y
Node.__pow = function(x, y)
    x, y = Node.if_isNode_then_change(x, y)

    local real = x.value.real^y.value.real
    local z = Node(real, imag)
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
    local dz_imag = dz_imag or 0
    self.grad.real = self.grad.real + dz_real
    if self.grad_fn then
        self.grad_fn(dz_real)
    end
end

--Node가 아니라면 Node로 변경
function Node.if_isNode_then_change(x, y)
    if getmetatable(x) ~= Node then x = Node(x) end
    if getmetatable(y) ~= Node then y = Node(y) end
    return x, y
end

--#endregion

return Node