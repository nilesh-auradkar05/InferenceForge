"""The load driver.
 
Two modes:
 
  closed-loop  -- `concurrency` slots. A new request is submitted only when one
                  completes. Load self-throttles, so the server can never be
                  observed in overload. Good for clean apples-to-apples latency.
 
  open-loop    -- requests are submitted at their scheduled arrival time no
                  matter what. Queue depth grows without bound past capacity.
                  This is the ONLY mode that finds the knee (post item 10).
"""

from __future__ import annotations
 
import asyncio
import threading
import time
 
from configs.config import LoadMode, RunConfig
from src.utils.engine import Engine, Event
from src.utils.metrics import GpuTelemetry, RequestRecord, aggregate
from src.utils.trace import Request
 
_SENTINEL = object()
 
 
def _cuda():
    try:
        import torch
 
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None
 
 
def _cuda_memory_summary(torch_module) -> dict[str, float]:
    """Report both allocator use and the reservation-based capacity limit."""
    cuda = torch_module.cuda
    gib = 2**30
    allocated = cuda.max_memory_allocated() / gib
    reserved = cuda.max_memory_reserved() / gib
    total = cuda.get_device_properties(cuda.current_device()).total_memory / gib
    allocator_slack = max(0.0, reserved - allocated)
    return {
        "gpu_total_gb": total,
        "peak_alloc_gb": allocated,
        "peak_reserved_gb": reserved,
        # Kept for existing result consumers. This is allocator slack (cached
        # free blocks plus unusable fragments), not a direct fragmentation
        # measurement.
        "fragmentation_gb": allocator_slack,
        "allocator_slack_gb": allocator_slack,
        # This, not gpu_total - peak_alloc, is the safe basis for increasing
        # --kv-budget-gb under the measured workload.
        "reserved_headroom_gb": max(0.0, total - reserved),
    }


def _drain_to_loop(engine: Engine, req: Request, loop, aq: "asyncio.Queue") -> None:
    """Runs on a worker thread. Never raises into the caller."""
 
    def push(item):
        loop.call_soon_threadsafe(aq.put_nowait, item)
 
    gen = None
    try:
        gen = engine.stream(req)
        for ev in gen:
            push(ev)
    except Exception as exc:  # noqa: BLE001
        push(Event(kind="error", t=time.perf_counter(), message=repr(exc)))
    finally:
        # Close the generator explicitly so its `finally` blocks run NOW --
        # releasing the engine lock and dropping tensor references -- rather
        # than whenever the GC gets around to it.
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass
        push(_SENTINEL)
 
 
async def _run_one(engine: Engine, req: Request, submit_t: float) -> RequestRecord:
    """Bridge the engine's blocking generator into asyncio.
 
    Events cross the thread boundary via `call_soon_threadsafe`. The obvious
    alternative -- `await asyncio.to_thread(q.get)` on a `queue.Queue` -- burns
    one ThreadPoolExecutor worker per in-flight request. The default executor
    holds min(32, cpu_count + 4) workers, so past that concurrency the LOAD
    DRIVER throttles itself and you measure the harness instead of the engine.
    In open-loop mode it also delays submissions, silently flattening the
    Poisson arrival process you were careful to construct.
    """
    rec = RequestRecord(
        req_id=req.req_id,
        prompt_len=req.prompt_len,
        output_len_requested=req.output_len,
        arrival_t=submit_t,
        schedule_t=submit_t,
    )
    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()
    threading.Thread(target=_drain_to_loop, args=(engine, req, loop, aq), daemon=True).start()
 
    last_t = None
    while True:
        item = await aq.get()
        if item is _SENTINEL:
            break
        ev: Event = item
        if ev.kind == "scheduled":
            rec.schedule_t = ev.t
        elif ev.kind == "token":
            if rec.output_len_actual == 0:
                rec.first_token_t = ev.t
            else:
                rec.itls.append(ev.t - last_t)
            last_t = ev.t
            rec.output_len_actual += 1
        elif ev.kind == "done":
            rec.end_t = ev.t
        elif ev.kind == "error":
            rec.error = ev.message
 
    if rec.end_t == 0.0:
        rec.end_t = last_t if last_t is not None else submit_t
    return rec
 
 
async def _drive_closed(engine: Engine, reqs: list[Request], concurrency: int) -> list[RequestRecord]:
    sem = asyncio.Semaphore(concurrency)
 
    async def worker(r: Request) -> RequestRecord:
        async with sem:
            return await _run_one(engine, r, time.perf_counter())
 
    return list(await asyncio.gather(*(worker(r) for r in reqs)))
 
 
async def _drive_open(engine: Engine, reqs: list[Request]) -> list[RequestRecord]:
    t0 = time.perf_counter()
    tasks: list[asyncio.Task] = []
    lateness: list[float] = []
    for r in reqs:
        delay = r.arrival_offset_s - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            lateness.append(-delay)
        tasks.append(asyncio.create_task(_run_one(engine, r, time.perf_counter())))
    if lateness and max(lateness) * 1e3 > 5.0:
        print(
            f"[warn] driver fell behind its own schedule by up to "
            f"{max(lateness) * 1e3:.1f} ms on {len(lateness)} request(s). "
            f"Arrivals are no longer Poisson -- treat this point as invalid."
        )
    return list(await asyncio.gather(*tasks))
 
 
async def run_benchmark(
    engine: Engine,
    requests: list[Request],
    cfg: RunConfig,
) -> tuple[dict, list[RequestRecord]]:
    warm = [r for r in requests if r.is_warmup]
    measured = [r for r in requests if not r.is_warmup]
    torch = _cuda()
 
    if warm:
        # Warmup is discarded entirely: it pays for lazy CUDA context creation,
        # cuBLAS autotuning, allocator growth, and clock ramp.
        await _drive_closed(engine, warm, max(1, cfg.load.concurrency))

    # Always, including when warmup is empty: the engine object is reused
    # across sweep points, so without this each point reports itself plus
    # every prior point.
    engine.reset_stats()

    if torch is not None:
        # Reset AFTER warmup so the peak reflects steady state, not load-time
        # transients.
        torch.cuda.reset_peak_memory_stats()
 
    telem = GpuTelemetry(cfg.telemetry_interval_s)
    telem.start()
 
    # Re-base open-loop arrival offsets so the first measured request is t=0.
    if cfg.load.mode is LoadMode.open and measured:
        base = measured[0].arrival_offset_s
        for r in measured:
            r.arrival_offset_s -= base
 
    wall_start = time.perf_counter()
    try:
        if cfg.load.mode is LoadMode.open:
            records = await _drive_open(engine, measured)
        else:
            records = await _drive_closed(engine, measured, cfg.load.concurrency)
    finally:
        wall_end = time.perf_counter()
        telem.stop()
 
    summary = aggregate(records, wall_start=wall_start, wall_end=wall_end, slo=cfg.slo)
    summary.update(telem.summary())
    summary.update({f"engine/{k}": v for k, v in engine.stats().items()})
    if torch is not None:
        summary.update(_cuda_memory_summary(torch))
    return summary, records
 