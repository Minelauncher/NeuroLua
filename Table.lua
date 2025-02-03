Table = {}

function Table.new(table)
    local self = setmetatable({}, Tensor)
    self.table = {}
    self.size = 0
    return self
end
Table = setmetatable(Table, {
    __call = function(_, table)
        return Table.new(table)
    end
})

-- Table[][][] 식으로 table 값 호출가능
Table.__index = function(table, index)
    local value = Table[index]
    if type(value) == 'function' then
        -- function table:func 처리
        return value
    else
        if type(index) == "number" or type(index) == "string" then
            return table.values[index]
        else
            -- 기본 __index 동작
            return rawget(table, index)
        end
    end
end

Table.__newindex = function(table, index, value)
    -- 기본 __newindex 동작
    rawset(table, index, value)
end

-- __tostring 메타메소드 구현 : Tensor 출력
Table.__tostring = function(self)
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