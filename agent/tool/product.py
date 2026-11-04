"""商城商品编排：候选查找、SKU筛选与对象确认。"""
import re
from agent.budget import invoke_query
from agent.tool import mall
from agent.tool.schemas import (
    CandidateRef, DiscoveryRequest, FilterParams, MallError, ProductListRequest,
    ProductMatch, ProductResolution, ProductSelection,
)
from agent.utils.log_tool import get_loger

logger = get_loger("agent.tool.product")


async def find_products(
    request: DiscoveryRequest | ProductListRequest, filters: FilterParams,
) -> ProductSelection:
    """查询一页候选，并用详情核实最多三个商品的 SKU。"""
    query = mall.search_products if isinstance(request, DiscoveryRequest) else mall.list_products
    page = await invoke_query(query, request)
    matches = []
    checked = []
    unverified = [card.id for card in page.items[3:]]
    alternative_colors = []
    for card in page.items[:3]:
        detail = await invoke_query(mall.get_product, card.id)
        checked.append(card.id)
        color_values = {}
        if filters.color is not None:
            color_options = [option for option in detail.options if option.name == "颜色"]
            if len(color_options) != 1:
                unverified.append(card.id)
                continue
            color_values = {value.id: value.value for value in color_options[0].values}
        selected = []
        unknown_color = False
        for sku in detail.skus:
            if not sku.available:
                continue
            if filters.min_price is not None and sku.price < filters.min_price:
                continue
            if filters.max_price is not None and sku.price > filters.max_price:
                continue
            if filters.color is not None:
                colors = [color_values[value_id] for value_id in sku.optionValueIds if value_id in color_values]
                if len(colors) != 1:
                    unknown_color = True
                    continue
                if colors[0] != filters.color:
                    if colors[0] not in alternative_colors:
                        alternative_colors.append(colors[0])
                    continue
            selected.append(sku)
        if unknown_color:
            unverified.append(card.id)
        if selected:
            # 回答证据只保留匹配 SKU，避免模型借用同商品另一颜色的价格。
            matches.append(ProductMatch(product=detail.model_copy(update={"skus": selected}), skus=selected))
    logger.info("product_selection candidates=%d checked=%d matches=%d unverified=%d has_more=%s",
                len(page.items), len(checked), len(matches), len(unverified), page.hasMore)
    return ProductSelection(matches=matches, page=page, checked_product_ids=checked,
                            unverified_product_ids=unverified, alternative_colors=alternative_colors)


async def resolve_product(subject: str, candidates: list[CandidateRef]) -> ProductResolution:
    """只把可信候选或明确编号解析为商品，不把相似名称当身份。"""
    known = {candidate.product_id: candidate for candidate in candidates
             if subject in (candidate.name, str(candidate.product_id))}
    explicit_id = re.fullmatch(r"(?:商品编号[：:]?\s*)?([1-9][0-9]*)", subject)
    if explicit_id:
        product_id = int(explicit_id.group(1))
    elif len(known) == 1:
        product_id = next(iter(known))
    elif known:
        return ProductResolution(status="choose", candidates=list(known.values()))
    else:
        page = await invoke_query(mall.search_products, DiscoveryRequest(q=subject))
        found = [CandidateRef(product_id=card.id, name=card.name) for card in page.items]
        exact = [candidate for candidate in found if candidate.name == subject]
        if not found:
            return ProductResolution(status="not_found")
        if len(exact) != 1 or page.hasMore:
            return ProductResolution(status="choose", candidates=found)
        product_id = exact[0].product_id
    try:
        product = await invoke_query(mall.get_product, product_id)
    except MallError as exc:
        if exc.http_status == 404 or exc.code == 404:
            logger.info("product_resolution status=not_found")
            return ProductResolution(status="not_found")
        raise
    logger.info("product_resolution status=resolved")
    return ProductResolution(status="resolved", product=product)
