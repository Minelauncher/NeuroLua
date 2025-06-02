local Loader = {}

-- flat 1D 테이블을 shape에 맞춘 다차원 테이블로 바꿔주는 재귀 함수
local function reshape(flat, shape, dim, offset)
    dim    = dim    or 1
    offset = offset or 1

    local size = shape[dim]
    local t    = {}

    if dim == #shape then
        -- 마지막 차원: flat[offset], flat[offset+1], … flat[offset+size-1]
        for i = 1, size do
            t[i] = flat[offset + i - 1]
        end
        return t, offset + size
    else
        -- 그 외 차원: subtables를 재귀 호출로 생성
        for i = 1, size do
            local subt, new_off = reshape(flat, shape, dim + 1, offset)
            t[i] = subt
            offset = new_off
        end
        return t, offset
    end
end

-- 4바이트를 읽어서 big-endian 32bit 정수로 변환
local function readInt32BE(f)
    local bytes = f:read(4)
    assert(#bytes == 4, "Unexpected EOF when reading 4 bytes")
    local b1, b2, b3, b4 = bytes:byte(1,4)
    -- 2^24 = 16777216, 2^16 = 65536, 2^8 = 256
    return b1 * 16777216 + b2 * 65536 + b3 * 256 + b4
end

function Loader.readIDX(path, reshape_to_tensor)
    local f = assert(io.open(path, "rb"))
    -- magic number + dims 읽기
    local magic = f:read(4)
    assert(#magic == 4, "Invalid IDX file")
    local dims = magic:byte(4)

    -- shape 읽기
    local shape = {}
    for i = 1, dims do
        shape[i] = readInt32BE(f)
    end

    -- flat 데이터 읽기
    local total = 1
    for _, v in ipairs(shape) do total = total * v end

    local raw = f:read(total)
    assert(#raw == total, "Unexpected EOF when reading data")
    f:close()

    local flat = {}
    for i = 1, total do
        flat[i] = raw:byte(i)
    end

    if reshape_to_tensor then
        local nested = reshape(flat, shape)
        return shape, nested
    else
        return shape, flat
    end
end

function Loader.getImage2D(flat_data, index, rows, cols)
    local image = {}
    for r = 1, rows do
        image[r] = {}
        for c = 1, cols do
            -- 1-based linear 인덱스
            local idx = (index - 1) * rows * cols + (r - 1) * cols + c
            image[r][c] = flat_data[idx] / 255
        end
    end
    return image
end
--[[
local shape, trainImage = Loader.readIDX("MNIST/train-images.idx3-ubyte", true)
print("이미지 수:", shape[1])     -- 60000
print("이미지 크기:", shape[2], "x", shape[3])  -- 28 x 28

local shape, trainLabel = Loader.readIDX("MNIST/train-labels.idx1-ubyte", true)
print("이미지 수:", shape[1])     -- 60000
print("이미지 크기:", shape[2], "x", shape[3])  -- 0 x 0

for _, v1 in pairs(trainImage[1]) do
    local str = ""
    for _, v2 in pairs(v1) do
        str = str..((v2/255)<0.5 and '0' or '1')
    end
    print(str)
end
print(trainLabel[1])
--]]
return Loader

