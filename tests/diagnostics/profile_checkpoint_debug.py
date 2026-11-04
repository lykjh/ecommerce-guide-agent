import asyncio
from contextlib import aclosing
import cProfile
import io
from pathlib import Path
import pstats
import sys
import tempfile
import time
from unittest.mock import AsyncMock, patch
sys.path.insert(0, r"D:/desktop/PythonProject")
from tests.v1.g.test_concurrency_g import collect, route_model
from agent import main_agent as main

async def probe():
    barrier = asyncio.Barrier(10)
    async def answer(messages, **kwargs):
        await barrier.wait()
        return await route_model().ainvoke(messages, config=kwargs.get("config"))
    model = AsyncMock()
    model.ainvoke.side_effect = answer
    with tempfile.TemporaryDirectory(prefix="agent-profile-") as directory, patch.object(main,"DB_PATH",str(Path(directory)/"probe.db")), patch.object(main.router,"client",model):
        replies = await asyncio.wait_for(asyncio.gather(*[collect("turn:0:0",f"profile-{i}") for i in range(10)]), 15)
        assert len(replies) == 10
        assert model.ainvoke.await_count == 10

for debug in (True, False):
    profiler = cProfile.Profile()
    wall,cpu=time.perf_counter(),time.process_time()
    with asyncio.Runner(debug=debug) as runner:
        profiler.enable()
        runner.run(probe())
        profiler.disable()
    print("PROFILE",debug,"wall",round(time.perf_counter()-wall,3),"cpu",round(time.process_time()-cpu,3))
    pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(14)
