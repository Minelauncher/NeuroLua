DataManager = {}
DataManager.__index = DataManager

function DataManager.new()
    local self = setmetatable({}, DataManager)
    return self
end

function DataManager.saveTableToFile(table, filename)
    -- 테이블을 문자열로 직렬화하는 함수
    local function serializeTable(t, indent)
        indent = indent or ""
        local result = "{\n"
        for key, value in pairs(t) do
            if type(key) == "number" then
                result = result .. indent .. "\t"
            else
                local formattedKey = type(key) == "string" and string.format('["%s"]', key) or key
                result = result .. indent .. "\t" .. formattedKey .. " = "
            end
            if type(value) == "table" then
                result = result .. serializeTable(value, indent .. "\t")
            elseif type(value) == "string" then
                result = result .. string.format('"%s"', value)
            elseif type(value) == "number" then
                result = result .. string.format("%.5f",value)
            else
                result = result .. tostring(value)
            end
            if key == #t then
                result = result .. "\n"
            else
                local n = type(value) == "table" and "\n" or ""
                result = result .. "," .. n
            end
        end
        return result .. indent .. "}"
    end
    local file = io.open(filename, "w")
    if file then
        file:write("return\n" .. serializeTable(table))
        file:close()
        print("Table was saved at " .. filename .. " !.")
    else
        print("Can't open the file!.")
    end
end

function DataManager.loadTableFromFile(filename)
    local f = loadfile(filename)
    if f then
        return f()  -- 테이블 반환
    else
        print("Can't load the file!.")
        return nil
    end
end
--[[
-- 원본 테이블
local data = {
    user = "Alice",
    age = 25,
    preferences = {
        theme = "dark",
        language = "Lua",
        plugins = {"autocomplete", "linter"}
    }
}

local data2 = {{{1,2},{3,4}}, {{1,2},{3,4}}}

-- 테이블 저장
DataManager.saveTableToFile(data, "settings.lua")

-- 저장된 테이블 읽기
local loadedSettings = DataManager.loadTableFromFile("settings.lua")

-- 읽은 데이터 출력
if loadedSettings then
    print("USER:", loadedSettings.user)
    print("THEME:", loadedSettings.preferences.theme)
    print("PULGINS LIST:")
    for _, plugin in ipairs(loadedSettings.preferences.plugins) do
        print("-"..plugin)
    end
end
--]]
return DataManager