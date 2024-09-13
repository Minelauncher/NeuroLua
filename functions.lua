--#region 함수 모음

-- Box-Muller 변환을 사용하여 하나의 표준 정규분포 난수 생성(mean==평균, stddev==표준편차)
function GenerateStandardNormal(mean, stddev)
    if stddev < 0 then
        error("stddev's value is under zero")
    end
    local u1 = math.random()
    local u2 = math.random()

    local z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    local z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2) -- 필요 없다면 생략

    return mean + z0 * stddev
end

function Gaussian_Distribution_Probability_Density(mean, stddev, value)
    return (1 / math.sqrt(2*math.pi * stddev^2)) * math.exp(1)^(-((value - mean)/(2 * stddev^2)))
end

-- 테이블 요소 곱
function mulTable(table)
    local sum = 1
    for _, value in pairs(table) do
        sum = sum * value
    end
    return sum
end

-- 테이블 원하는 크기로 하나 생성 초기값 지정 가능
function initializeTable(size, value)
    local t = {}
    for i = 1, size do
        t[i] = value
    end
    return t
end

-- 테이블을 깊은 복사
function tableCopy(original)
    -- 원본이 테이블이 아니면 원본을 그대로 반환 (기본형 데이터의 경우)
    if type(original) ~= "table" then
        return original
    end

    -- 이미 복사된 테이블을 추적하는 테이블 (순환 참조 방지용)
    local copy = {}
    for key, value in pairs(original) do
        -- 재귀적으로 깊은 복사를 수행
        copy[key] = tableCopy(value)
    end
    return copy
end

-- ANSI 이스케이프 코드로 RGB 색상을 정의하는 함수
function rgb(r, g, b)
    return string.format("\27[38;2;%d;%d;%dm", r, g, b)
end
local reset = "\27[0m"

function Sleep(ms)
    local t = os.clock()
    ms = ms / 1000
    while os.clock() - t <= ms do
    end
end
--#endregion