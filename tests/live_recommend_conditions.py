import asyncio

from langchain_core.messages import HumanMessage

from agent.budget import invoke_model
from agent.deps import get_extract_client, INTERNAL_TAG
from agent.memory import memory_manager
from agent.tool.schemas import FilterParams
from agent.prompts import TOTAL_EXTRACT_SYSTEM

CASES = [
    {
        "name": "修改预算",
        "query": "预算提高到500元",
        "old_price": 300,
        "expected_price": 500,
        "expected_color": "黑色",
    },
    {
        "name": "取消颜色",
        "query": "颜色不限",
        "old_price": 500,
        "expected_price": 500,
        "expected_color": None,
    },
    {
        "name": "修改颜色",
        "query": "换成白色",
        "old_price": 500,
        "expected_price": 500,
        "expected_color": "白色",
    },
]
async def main():
    # 模拟上一轮保存的条件，以及本轮用户的新要求

    for case in CASES:
        state = {
             "messages": [
                HumanMessage(content=case["query"]),
            ],
            "intent": "商品推荐",
            "task_context": {
                "intent": "商品推荐",
                 "filters": {
                    "category": "耳机",
                    "min_price": None,
                    "max_price": case["old_price"],
                    "color": "黑色",
                },
             },
         }

        structured_llm = (
            get_extract_client()
            .with_structured_output(FilterParams)
            .with_config({"tags": [INTERNAL_TAG]})
        )

        messages = memory_manager.build_messages(
            state,
            TOTAL_EXTRACT_SYSTEM,
           purpose="extract",
        )

    # 没有使用替身：这里真正请求模型
        params = await invoke_model(
            structured_llm,
            messages,
            schema=FilterParams,
        )

        print(case["name"], "模型实际返回：", params.model_dump())

        assert params.category == "耳机", f"{case['name']}品类没有保留"
        assert params.max_price == case["expected_price"], f"{case['name']}预算不符合预期"
        assert params.color == case["expected_color"], f"{case['name']}颜色不符合预期"
        assert params.min_price is None, f"{case['name']}模型添加了最低价"

        print(case["name"], "通过测试")


if __name__ == "__main__":
    asyncio.run(main())