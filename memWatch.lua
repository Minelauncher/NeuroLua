--------------------------------------------------------------------
-- 초경량 메모리 워처 (LuaJIT / Lua 5.x 겸용)
--------------------------------------------------------------------
local memWatch = {}

-- ① 초기화 ────────────────────────────────────────────────────────
--    interval : 몇 step 마다 로그를 찍을지 (기본 100)
function memWatch.start(interval)
    memWatch.interval  = interval or 100
    memWatch.baseStep  = 0
    collectgarbage()                     -- 기준점 전 GC
    memWatch.baseKB   = collectgarbage("count")   -- Lua heap KB
    print(("memWatch ▶ 시작  | Lua %.1f KB"):format(memWatch.baseKB))
end

-- ② 루프 중 호출 ─────────────────────────────────────────────────
function memWatch.step(step)
    if not memWatch.interval then return end
    if step % memWatch.interval ~= 0 then return end

    collectgarbage()                     -- 강제 GC(0~1ms)
    local now = collectgarbage("count")
    local dKB = now - memWatch.baseKB
    print(("step %d | Lua Δ %+7.1f KB | heap %.1f KB")
          :format(step, dKB, now))
end

return memWatch