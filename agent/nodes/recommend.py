"""推荐与概览共用分类导航、条件提取和同 SKU 查询。"""
from decimal import Decimal
from openai import APIError
from pydantic import ValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from agent.budget import BudgetStop, invoke_model, invoke_query
from agent.deps import get_extract_client, INTERNAL_TAG
from agent.memory import memory_manager
from agent.prompts import TOTAL_EXTRACT_SYSTEM, CATEGORY_QUERY_FAILED, mall_error_reply
from agent.router import SIGNAL_NONE
from agent.state import AgentState
from agent.tool import mall
from agent.tool.product import find_products
from agent.tool.schemas import CategoryRequest, DiscoveryRequest, ProductListRequest, FilterParams, MallError
from agent.utils.log_tool import get_loger

logger = get_loger("agent.nodes.recommend")


async def navigate_categories(category: dict, query: str | None, action: dict) -> tuple[dict, str | None]:
    """明确关键词直接查商品；显式浏览才读分类，不猜分类ID。"""
    reply = None
    parent = None
    if not action and query is not None and not category.get("selected"):
        return {"query": query, "mode": "search", "selected": None, "pending": False}, None
    if action.get("kind") == "browse_root":
        category = {}
    elif action.get("kind") == "keyword":
        category = {"query": query, "mode": "search", "selected": None, "pending": False}
    elif action:
        index = action["index"]
        if index >= len(category["candidates"]):
            reply = "该序号不在最近展示的分类候选中，请按已展示的序号选择。"
        else:
            chosen = category["candidates"][index]
            if action["kind"] == "browse" and chosen["hasChildren"]:
                parent = chosen
            elif action["kind"] == "browse":
                reply = "该分类没有可展开的子分类；可选择该分类作为范围。"
            else:
                category = {"query": query, "selected": {**chosen, "source": "user"},
                            "pending": False, "mode": "list" if query is None or query == chosen["name"] else "search"}
    if reply is None and (parent is not None or action.get("kind") == "browse_root"):
        rows = await invoke_query(mall.list_categories, CategoryRequest(parentId=parent["id"] if parent else None))
        path = parent["path"] if parent else []
        candidates = [{**row.model_dump(), "path": path + [{"id": row.id, "name": row.name}],
                       "parent_id": parent["id"] if parent else None} for row in rows]
        if candidates:
            category = {"query": query, "pending": True, "candidates": candidates, "selected": None}
        elif parent:
            reply = "本次查询该分类的可见子分类为空；仍可选择此前展示的分类范围，或按关键词搜索。"
        else:
            category = {"query": query, "pending": False, "selected": None, "mode": "search"}
    if reply is None and category.get("pending"):
        lines = [f'{i}. ' + " / ".join(item["name"] for item in candidate["path"])
                 + ("（可展开子分类）" if candidate["hasChildren"] else "")
                 for i, candidate in enumerate(category["candidates"], 1)]
        reply = "请确认分类范围；以下是当前层级，不能据此认定全商城分类唯一：\n" + "\n".join(lines)
        reply += "\n回复“第N个”选择该范围（包含可见后代），或“查看第N个子分类”。无法确认可回复“按关键词搜索”。"
        if len(reply) > memory_manager.MAX_HISTORY_CHARS:
            reply = "当前层级分类过多，超出当前对话可容纳范围，分类定位未完成；需要核实分类规模与展示方案。"
            category = {}
    return category, reply


async def product_search_node(state: AgentState, intent: str):
    action = state.get("category_action", {})
    previous_task = state.get("task_context", {})
    if action.get("kind") == "browse_root" and "filters" not in previous_task:
        params = FilterParams()
    elif action:
        # checkpoint 中金额用字符串保存；内部可信状态只还原类型，不再做入口校验。
        saved = dict(previous_task["filters"])
        for name in ("min_price", "max_price"):
            if saved[name] is not None:
                saved[name] = Decimal(saved[name])
        params = FilterParams.model_construct(**saved)
        if action.get("kind") == "next_page":
            params.next_page = True
            action = {}
    else:
        # 显式通过工具参数传递 schema，避免兼容端点对 json_schema 的不同处理。
        structured = get_extract_client().with_structured_output(
            FilterParams, method="function_calling").with_config(tags=[INTERNAL_TAG])
        try:
            params = await invoke_model(structured,
                memory_manager.build_messages(state, TOTAL_EXTRACT_SYSTEM, purpose="extract"), schema=FilterParams)
        except BudgetStop:
            raise
        except (APIError, TimeoutError, ValidationError, OutputParserException) as exc:
            # 只记录字段位置/错误码，不记录模型原文或用户数据。
            fields = ([(error["loc"], error["type"]) for error in exc.errors(
                include_input=False, include_context=False, include_url=False)]
                if isinstance(exc, ValidationError) else [])
            logger.warning("filter_extract_failed error_type=%s fields=%s", type(exc).__name__, fields)
            reply = "筛选条件解析失败，请重试本次请求。"
            if isinstance(exc, ValidationError) and any("最低价格不能超过最高价格" in error["msg"] for error in exc.errors()):
                reply = "最低价格不能超过最高价格，请重新确认价格上下限。"
            return {"messages": [AIMessage(content=reply)], "final_reply": reply}

    filters = params.model_dump(mode="json", exclude={"next_page"})
    for name in ("min_price", "max_price"):
        value = getattr(params, name)
        if value is not None:
            filters[name] = format(value.normalize(), "f")
            # 商城游标区分金额文本；首查与 checkpoint 恢复后的请求必须一致。
            setattr(params, name, Decimal(filters[name]))
    task = {"intent": intent, "filters": filters}
    scope_products = None
    if params.scope != "catalog":
        scope_products = (state.get("comparison_products", []) if params.scope == "comparison"
                          else previous_task.get("scope_products", state.get("candidate_products", [])))
        if not scope_products:
            reply = "尚未确认要保留的商品范围，请先指定商品或重新查询；本轮未扩大到全商城搜索。"
            return {"messages": [AIMessage(content=reply)], "final_reply": reply}
        task["scope_products"] = scope_products
    previous_filters = previous_task.get("filters", {})
    if params.category == previous_filters.get("category") and params.color is not None:
        if params.color in previous_task.get("color_choices", []):
            if params.color not in state["messages"][-1].content:
                reply = "颜色名称不能自动等同；请明确回复您选择的商城实际颜色名称：" + "、".join(previous_task["color_choices"])
                return {"messages": [AIMessage(content=reply)], "final_reply": reply,
                        "product_page": {}, "candidate_products": []}
            task["color_confirmation"] = {"requested": previous_filters["color"], "value": params.color}
        elif params.color == previous_filters.get("color") and "color_confirmation" in previous_task:
            task["color_confirmation"] = previous_task["color_confirmation"]
    category = state.get("category_context", {})
    if category.get("query") != params.category:
        category = {}
    update = {"task_context": task, "category_context": category, "category_action": {},
              "candidate_products": [], "product_page": {},
              "signal_state": {"signal": SIGNAL_NONE, "level": 0}}
    reply = None
    if params.unsupported_specs:
        reply = "目前仅支持价格、颜色和可售状态筛选，尚不支持这些硬性规格条件：" + "、".join(params.unsupported_specs)
        reply += "。本轮未查询符合这些要求的商品；可指定商品咨询规格，或明确取消这些筛选条件。"
    else:
        try:
            if scope_products is not None:
                category = {"mode": "selection", "selected": None, "pending": False}
            else:
                category, reply = await navigate_categories(category, params.category, action)
        except MallError:
            reply = CATEGORY_QUERY_FAILED
        update["category_context"] = category
        if reply is None:
            selected = category.get("selected")
            category_id = selected["id"] if selected else None
            mode = category.get("mode", "list")
            previous = state.get("product_page", {})
            cursor = None
            if (params.next_page and previous.get("filters") == filters
                    and previous.get("mode") == mode and previous.get("category_id") == category_id):
                update["product_page"] = previous
                if not previous["hasMore"]:
                    reply = "这次搜索没有下一页了。可以修改条件重新查询；这不代表全商城商品总数。"
                else:
                    cursor = previous["nextCursor"]
            if reply is None:
                if mode == "search":
                    request = DiscoveryRequest(q=params.category, categoryId=category_id,
                        minPrice=params.min_price, maxPrice=params.max_price, cursor=cursor)
                else:
                    request = ProductListRequest(categoryId=category_id, minPrice=params.min_price,
                        maxPrice=params.max_price, afterId=cursor)
                try:
                    if scope_products is None:
                        selection = await find_products(request, params)
                    else:
                        selection = await find_products(request, params,
                            product_ids=[item["product_id"] for item in scope_products])
                except MallError as exc:
                    reply = mall_error_reply(exc.kind, exc.http_status, exc.code)
                    if mode == "search" and cursor is not None and (
                            exc.http_status in (400, 410) or exc.code in (400, 410)):
                        update["product_page"] = {}
                        reply = "本次无法继续商品搜索翻页，请重新查询；回复‘下一批’将按原条件从第一页重新开始。"
                    return {**update, "messages": [AIMessage(content=reply)], "final_reply": reply}
                candidates, lines = [], []
                for match in selection.matches:
                    product, sku = match.product, match.skus[0]
                    values = {value.id: value.value for option in product.options for value in option.values}
                    specs = "、".join(values[value_id] for value_id in sku.optionValueIds if value_id in values)
                    candidates.append({"product_id": product.id, "name": product.name})
                    lines.append(f"- {product.name}（商品编号 {product.id}，SKU {sku.id}）：{sku.price} 元"
                                 + (f"，规格：{specs}" if specs else "，未提供具体规格") + "，查询时可售。")
                reply = (f"本批核实到 {len(lines)} 款符合已核实条件的商品：\n" + "\n".join(lines)) if lines else "本批核实的候选中没有找到符合条件的商品。"
                if mode == "search":
                    reply = "本次按关键词搜索" + ("，并限定在已选分类范围内。\n" if selected else "，未确认精确分类。\n") + reply
                elif mode == "selection":
                    reply = "本次仅在原商品范围内筛选。\n" + reply
                if selection.unverified_product_ids:
                    reply += "\n部分候选尚未核实，或缺少可确认的规格资料，不能认定符合条件。"
                if params.color is not None and selection.alternative_colors and not lines:
                    task["color_choices"] = selection.alternative_colors
                    reply += "\n已核实候选中还有这些商城实际颜色值：" + "、".join(selection.alternative_colors)
                    reply += "。它们不自动等同于您要求的颜色；如需修改，请明确回复实际颜色名称。"
                reply += "\n搜索还有更多候选，可要求下一批。" if selection.page.hasMore else "\n这次搜索没有更多候选页。"
                reply += "当前批次不代表全商城总数；价格和可售状态以商城下单时为准。"
                update["candidate_products"] = candidates
                update["product_page"] = {"filters": filters, "mode": mode, "category_id": category_id,
                    "hasMore": selection.page.hasMore, "nextCursor": selection.page.nextCursor}
                logger.info("product_reply matches=%d checked=%d unverified=%d has_more=%s",
                    len(lines), len(selection.checked_product_ids), len(selection.unverified_product_ids), selection.page.hasMore)
    return {**update, "messages": [AIMessage(content=reply)], "final_reply": reply}


async def recommend_node(state: AgentState):
    return await product_search_node(state, "商品推荐")
