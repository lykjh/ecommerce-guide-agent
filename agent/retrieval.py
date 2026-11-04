"""现有商品文本的保守证据筛选，不将相似片段当作商品身份。"""
import re
from agent.tool.schemas import FilterParams
MAX_EVIDENCE_BYTES = 4500
NO_EVIDENCE = "暂时没有查到这款商品的可靠资料，请提供完整商品名称或商品编号。"
AMBIGUOUS_PRODUCT = "资料中存在多个同名商品，请提供商品编号或完整型号，确认后再查询。"


def select_product_evidence(context: str, subject: str) -> tuple[str, str]:
    """要求片段含G编号和完整名称；混合片段按商品记录拆开后再筛选。"""
    if not isinstance(context, str) or not context.strip():
        return "", "empty"
    matches = []
    seen = set()
    for chunk in context.split("\n\n---\n\n"):
        # 不使用没有商品头的续页，避免把上一个商品的属性接到另一个对象上。
        records = re.split(r"(?=^【G-\d+】)", chunk, flags=re.MULTILINE)
        for record in records:
            header = re.match(r"【(G-\d+)】([^\r\n]+)", record)
            if not header:
                continue
            identifier, title = header.groups()
            name_match = re.search(r"^商品名称[：:]\s*(.+)$", record, re.MULTILINE)
            name = name_match.group(1).strip() if name_match else title.strip()
            if subject not in (identifier, name):
                continue
            normalized = record.strip()
            if normalized not in seen:
                seen.add(normalized)
                matches.append((identifier, normalized))
    if len({identifier for identifier, _ in matches}) > 1:
        return "", "ambiguous"
    selected = []
    size = 0
    for _, record in matches:
        length = len(record.encode("utf-8")) + 2
        if size + length > MAX_EVIDENCE_BYTES:
            continue
        selected.append(record)
        size += length
    return "\n\n".join(selected), "matched" if selected else "no_supported_evidence"


def select_recommendation_evidence(
    context: str,
    filters: FilterParams,
) -> str:
    selected = []

    for chunk in context.split("\n\n---\n\n"):
        records = re.split(
            r"(?=^【G-\d+】)",
            chunk,
            flags=re.MULTILINE,
        )

        for record in records:
            if not re.match(r"【G-\d+】", record):
                continue

            if filters.max_price is not None:
                price_match = re.search(
                    r"^价格：[ \t]*(\d+(?:\.\d+)?)[ \t]*元[ \t]*$",
                    record,
                    flags=re.MULTILINE,
                )

                # 有预算限制但资料没给出明确价格，不能认定它符合预算。
                if price_match is None:
                    continue

                price = float(price_match.group(1))
                if price > filters.max_price:
                    continue
            if filters.color is not None:
                color_match = re.search(
                    r"^规格：[ \t]*([^，,\r\n]+)",
                    record,
                    flags=re.MULTILINE,
                )
                if color_match is None:
                    continue

                color = color_match.group(1).strip()
                if color != filters.color:
                    continue

            if filters.category is not None:
                category_match = re.search(
                    r"^类目：[ \t]*([^\r\n]+)",
                    record,
                    flags=re.MULTILINE,
                )
                if category_match is None:
                    continue

                category = category_match.group(1).strip()
                if filters.category not in category:
                    continue

            selected.append(record.strip())

    return "\n\n".join(selected)

