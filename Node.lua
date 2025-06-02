--[[
--]]
require("functions")

-- value 및 grad 출력, 비교를 위한 메타테이블
local valueMT = {
    __tostring = function(self)
        return string.format("(%6.3f)", self.real)
    end,
    __eq = function(a, b)
        return a.real == b.real
    end,
    __lt = function(a, b)
        return a.real < b.real
    end,
    __le = function(a, b)
        return a.real <= b.real
    end
}

local gradMT = {
    __tostring = function(self)
        return string.format("(%6.3f)", self.real)
    end,
    __eq = function(a, b)
        return a.real == b.real
    end,
    __lt = function(a, b)
        return a.real < b.real
    end,
    __le = function(a, b)
        return a.real <= b.real
    end
}

local Node = {}
Node.__index = Node

-- Node 생성자: 기본 값, 허수부, 기울기 함수(옵션) 설정
function Node.new(real, imag, grad_fn, requires_grad)
    local self = setmetatable({}, Node)
    
    self.value = {real = real or 0, imag = imag or 0}
    self.value = setmetatable(self.value, valueMT)
    
    self.grad = {real = 0, imag = 0}
    self.grad = setmetatable(self.grad, gradMT)
    
    self.grad_fn = grad_fn or nil  -- 역전파 시 호출할 기울기 함수
    self.parents = {}               -- 부모 노드 리스트 (필요 시 사용)
    self.requires_grad = (requires_grad ~= false)  -- nil 이나 true → 추적
    
    return self
end

-- Node 호출 시 Node.new를 대신 호출 (객체 선언 방식 변경)
setmetatable(Node, {
    __call = function(self, real, imag, grad_fn, requires_grad)
        return Node.new(real, imag, grad_fn, requires_grad)
    end
})

-- Node의 문자열 표현
Node.__tostring = function(self)
    return string.format("[Node]\n v: [%s] g: [%s]", tostring(self.value), tostring(self.grad))
end

-- 동등 비교 등 (기본형은 그대로 유지)
Node.__eq = function(a, b)
    a, b = Node.convertToNode(a, b)
    return a.value == b.value
end

Node.__lt = function(a, b)
    a, b = Node.convertToNode(a, b)
    return a.value < b.value
end

Node.__le = function(a, b)
    a, b = Node.convertToNode(a, b)
    return a.value <= b.value
end

-- backward 메소드: 역전파 시 기울기 함수 호출
function Node:backward(dz_real, dz_imag)
    if not self.requires_grad then return end  -- detach 된 노드는 스킵
    local dz_real = dz_real or 1
    local dz_imag = dz_imag or 1
    self.grad.real = self.grad.real + dz_real
    self.grad.imag = self.grad.imag + dz_imag
    if self.grad_fn then
        -- 미리 정의된 기울기 함수에 현재 노드와 미분값을 전달
        self.grad_fn(self, dz_real, dz_imag)
    end
end

-- 분리 메서드 추가
function Node:detach()
    -- leaf 복사본을 만들어 반환
    return Node(self.value.real, self.value.imag, nil, false)
end

function Node:detach_()
    -- 제자리 분리: 이후 이 노드로부터 gradient 전파 금지
    self.grad_fn       = nil
    self.parents       = {}
    self.requires_grad = false
    return self
end

-- convertToNode: 인자가 Node가 아니면 즉석에서 변환
function Node.convertToNode(x, y)
    if getmetatable(x) ~= Node then
        x = Node(x)
    end
    if getmetatable(y) ~= Node then
        y = Node(y)
    end
    return x, y
end

-------------------------------------------------------------------
-- 미리 정의된 제너릭 기울기 함수들 (클로저 대신 재사용)
-------------------------------------------------------------------

local function generic_add_grad(node, dz_real, dz_imag)
    -- 덧셈: d/dx = 1, d/dy = 1
    node.left:backward(dz_real, dz_imag)
    node.right:backward(dz_real, dz_imag)
end

local function generic_sub_grad(node, dz_real, dz_imag)
    -- 뺄셈: d/dx = 1, d/dy = -1
    node.left:backward(dz_real, dz_imag)
    node.right:backward(-dz_real, -dz_imag)
end

local function generic_mul_grad(node, dz_real, dz_imag)
    -- 곱셈: d/dx = y, d/dy = x
    node.left:backward(dz_real * node.right.value.real, dz_imag * node.right.value.imag)
    node.right:backward(dz_real * node.left.value.real, dz_imag * node.left.value.imag)
end

local function generic_div_grad(node, dz_real, dz_imag)
    -- 나눗셈: x/y 미분 (단순화된 형태)
    local left = node.left
    local right = node.right
    local a, b = left.value.real, left.value.imag
    local c, d = right.value.real, right.value.imag
    local denom = c^2 + d^2
    left:backward(dz_real * c / denom, dz_imag * -d / denom)
    
    -- 우변에 대한 미분은 복잡하므로 단순화한 예시
    local denom2 = (c^2 - d^2)^2 + 4 * c^2 * d^2
    right:backward(dz_real * -a * (c^2 - d^2) / denom2, dz_imag * -b * (c^2 - d^2) / denom2)
end

-- 지수 함수: y = exp(x), dy/dx = exp(x)
local function generic_exp_grad(node, dz_real, dz_imag)
    node.child:backward(dz_real * node.value.real, dz_imag * node.value.real)
end

-- 로그 함수: y = log(x), dy/dx = 1/x
local function generic_log_grad(node, dz_real, dz_imag)
    local left = node.left
    local right = node.right

    left:backward(dz_real * (1 / (left.value.real * math.log(right.value.real))))
    right:backward(dz_real * -(math.log(left.value.real) / (right.value.real * math.log(right.value.real)^2)))
end

-- 거듭제곱 함수: y = x^p, dy/dx = p * x^(p-1)
local function generic_pow_grad(node, dz_real, dz_imag)
    local child = node.child
    local exponent = node.exponent

    child:backward(dz_real * exponent.value.real * child.value.real^(exponent.value.real - 1))
    exponent:backward(dz_real * child.value.real^exponent.value.real * math.log(child.value.real))
end

-- 사인 함수: y = sin(x), dy/dx = cos(x)
local function generic_sin_grad(node, dz_real, dz_imag)
    node.child:backward(dz_real * math.cos(node.child.value.real), dz_imag * math.cos(node.child.value.real))
end

-- 코사인 함수: y = cos(x), dy/dx = -sin(x)
local function generic_cos_grad(node, dz_real, dz_imag)
    node.child:backward(-dz_real * math.sin(node.child.value.real), -dz_imag * math.sin(node.child.value.real))
end

-------------------------------------------------------------------
-- 연산자 오버로딩: 클로저 대신 제너릭 기울기 함수를 사용
-------------------------------------------------------------------

-- 덧셈
Node.__add = function(x, y)
    x, y = Node.convertToNode(x, y)
    local sum = x.value.real + y.value.real  -- 허수부는 단순히 0으로 처리 (필요에 따라 확장 가능)
    local z = Node(sum, 0)
    
    if x.requires_grad or y.requires_grad then
        z.grad_fn = generic_add_grad
        z.left, z.right = x, y
        z.requires_grad = true
    else
        -- 둘 다 require_grad=false 면 grad_fn 자체를 남기지 않음
        z.requires_grad = false
    end
    return z
end

-- 뺄셈
Node.__sub = function(x, y)
    x, y = Node.convertToNode(x, y)
    local diff = x.value.real - y.value.real
    local z = Node(diff, 0)

    if x.requires_grad or y.requires_grad then
        z.grad_fn = generic_sub_grad
        z.left = x
        z.right = y
    else
        -- 둘 다 require_grad=false 면 grad_fn 자체를 남기지 않음
        z.requires_grad = false
    end
    return z
end

-- 곱셈
Node.__mul = function(x, y)
    x, y = Node.convertToNode(x, y)
    local a, b = x.value.real, x.value.imag
    local c, d = y.value.real, y.value.imag
    local real = a * c - b * d
    local imag = a * d + b * c
    local z = Node(real, imag)
    
    if x.requires_grad or y.requires_grad then
        z.grad_fn = generic_mul_grad
        z.left = x
        z.right = y
    else
        -- 둘 다 require_grad=false 면 grad_fn 자체를 남기지 않음
        z.requires_grad = false
    end
    return z
end

-- 나눗셈
Node.__div = function(x, y)
    x, y = Node.convertToNode(x, y)
    local a, b = x.value.real, x.value.imag
    local c, d = y.value.real, y.value.imag
    local denom = c^2 + d^2
    local real = (a * c + b * d) / denom
    local imag = (b * c - a * d) / denom
    local z = Node(real, imag)
    
    if x.requires_grad or y.requires_grad then
        z.grad_fn = generic_div_grad
        z.left = x
        z.right = y
    else
        -- 둘 다 require_grad=false 면 grad_fn 자체를 남기지 않음
        z.requires_grad = false
    end
    return z
end

-- 단항 마이너스 (음수)
Node.__unm = function(x)
    return -1 * x
end

-- 지수: x^y
Node.__pow = function(x, y)
    x, y = Node.convertToNode(x, y)

    local real = x.value.real^y.value.real
    local z = Node(real, 0)

    if x.requires_grad or y.requires_grad then
        z.grad_fn = generic_pow_grad
        z.child = x
        z.exponent = y
    else
        -- 둘 다 require_grad=false 면 grad_fn 자체를 남기지 않음
        z.requires_grad = false
    end
    return z
end

-- 로그: log_y_(x)
function Node.log(x, y)
    y = y or Node(math.exp(1))
    x, y = Node.convertToNode(x, y)

    local real = math.log(x.value.real, y.value.real)

    local z = Node(real, 0)

    if x.requires_grad or y.requires_grad then
        z.grad_fn = generic_log_grad
        z.left = x
        z.right = y
    else
        -- 둘 다 require_grad=false 면 grad_fn 자체를 남기지 않음
        z.requires_grad = false
    end
    return z
end

-- 지수: y = exp(x)
function Node.exp(x)
    if getmetatable(x) ~= Node then
        x = Node(x)
    end
    local out = Node(math.exp(x.value.real), 0)

    if x.requires_grad then
        out.grad_fn = generic_exp_grad
        out.child = x
    else
        -- require_grad=false 면 grad_fn 자체를 남기지 않음
        out.requires_grad = false
    end
    return out
end

-- 사인: y = sin(x)
function Node.sin(x)
    if getmetatable(x) ~= Node then
        x = Node(x)
    end
    local out = Node(math.sin(x.value.real), 0)

    if x.requires_grad then
        out.grad_fn = generic_sin_grad
        out.child = x
    else
        -- require_grad=false 면 grad_fn 자체를 남기지 않음
        out.requires_grad = false
    end
    return out
end

-- 코사인: y = cos(x)
function Node.cos(x)
    if getmetatable(x) ~= Node then
        x = Node(x)
    end
    local out = Node(math.cos(x.value.real), 0)

    if x.requires_grad then
        out.grad_fn = generic_cos_grad
        out.child = x
    else
        -- require_grad=false 면 grad_fn 자체를 남기지 않음
        out.requires_grad = false
    end
    return out
end

-- 노드의 value와 grad를 보기 좋게 출력하는 함수
local function print_node(n, label)
    print(string.format("%s -> value: %s, grad: %s", label, tostring(n.value), tostring(n.grad)))
end

-- 반복문 기반 역전파 함수: backward_iterative
function Node:backward_iterative(dz_real, dz_imag)
    dz_real = dz_real or 1
    dz_imag = dz_imag or 1
    local stack = {}
    -- 초기 노드와 미분값을 스택에 넣음
    table.insert(stack, {node = self, dz_real = dz_real, dz_imag = dz_imag})
    
    while #stack > 0 do
        local item = table.remove(stack)
        local node = item.node
        local dzr = item.dz_real
        local dzi = item.dz_imag
        
        -- 현재 노드의 기울기를 업데이트
        node.grad.real = node.grad.real + dzr
        node.grad.imag = node.grad.imag + dzi
        
        -- 기울기 함수가 있다면, 해당 노드의 종류에 따라 자식 노드에 미분값을 전파
        if node.grad_fn then
            if node.grad_fn == generic_add_grad then
                -- 덧셈: 미분값을 그대로 양쪽에 전파
                table.insert(stack, {node = node.left, dz_real = dzr, dz_imag = dzi})
                table.insert(stack, {node = node.right, dz_real = dzr, dz_imag = dzi})
            elseif node.grad_fn == generic_sub_grad then
                -- 뺄셈: 오른쪽에는 음수를 전파
                table.insert(stack, {node = node.left, dz_real = dzr, dz_imag = dzi})
                table.insert(stack, {node = node.right, dz_real = -dzr, dz_imag = -dzi})
            elseif node.grad_fn == generic_mul_grad then
                -- 곱셈: 각각 상대 노드의 값으로 곱해진 미분값을 전파
                table.insert(stack, {node = node.left, dz_real = dzr * node.right.value.real, dz_imag = dzi * node.right.value.imag})
                table.insert(stack, {node = node.right, dz_real = dzr * node.left.value.real, dz_imag = dzi * node.left.value.imag})
            elseif node.grad_fn == generic_div_grad then
                -- 나눗셈: 단순화된 형태의 미분 계산
                local left = node.left
                local right = node.right
                local a, b = left.value.real, left.value.imag
                local c, d = right.value.real, right.value.imag
                local denom = c^2 + d^2
                table.insert(stack, {node = left, dz_real = dzr * c / denom, dz_imag = dzi * -d / denom})
                local denom2 = (c^2 - d^2)^2 + 4 * c^2 * d^2
                table.insert(stack, {node = right, dz_real = dzr * -a * (c^2 - d^2) / denom2, dz_imag = dzi * -b * (c^2 - d^2) / denom2})
            elseif node.grad_fn == generic_exp_grad then
                -- 지수 함수: exp(x)의 미분은 exp(x)
                table.insert(stack, {node = node.child, dz_real = dzr * node.value.real, dz_imag = dzi * node.value.real})
            elseif node.grad_fn == generic_log_grad then
                -- 로그 함수: 단순화된 형태
                local left = node.left
                local right = node.right
                table.insert(stack, {node = left, dz_real = dzr * (1 / (left.value.real * math.log(right.value.real))), dz_imag = 0})
                table.insert(stack, {node = right, dz_real = dzr * -(math.log(left.value.real) / (right.value.real * math.log(right.value.real)^2)), dz_imag = 0})
            elseif node.grad_fn == generic_pow_grad then
                -- 거듭제곱 함수: 미분값 전파
                local child = node.child
                local exponent = node.exponent
                table.insert(stack, {node = child, dz_real = dzr * exponent.value.real * child.value.real^(exponent.value.real - 1), dz_imag = dzi})
                table.insert(stack, {node = exponent, dz_real = dzr * child.value.real^exponent.value.real * math.log(child.value.real), dz_imag = 0})
            elseif node.grad_fn == generic_sin_grad then
                -- 사인 함수: 미분은 cos(x)
                table.insert(stack, {node = node.child, dz_real = dzr * math.cos(node.child.value.real), dz_imag = dzi * math.cos(node.child.value.real)})
            elseif node.grad_fn == generic_cos_grad then
                -- 코사인 함수: 미분은 -sin(x)
                table.insert(stack, {node = node.child, dz_real = -dzr * math.sin(node.child.value.real), dz_imag = -dzi * math.sin(node.child.value.real)})
            end
        end
    end
end

return Node