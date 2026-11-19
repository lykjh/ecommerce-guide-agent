"""Read-only guidance; private order facts never enter the model or its history."""

from agent.rag.policy import load_policy
from agent.nodes.policy import policy_sentences


STEPS = {
    "REQUESTED": "待审核；请在售后记录中查看商家答复。",
    "REJECTED": "审核未通过；请在售后记录查看审核说明，与销售者核对争议。",
    "WAITING_RETURN": "等待寄回；先向销售者确认本次退货地址，再在售后记录填写物流公司和单号。",
    "WAITING_RECEIPT": "等待商家确认收货；保留交寄凭证，可在售后记录查看进展。",
    "REFUNDING": "退款处理中；尚不能确认完成，请查看后续退款结果。",
    "REFUNDED": "售后记录显示退款完成；该状态不能证明真实资金已到账。",
    "CLOSED": "申请已关闭；请在售后记录核对关闭原因。",
}
REFUNDS = {"PENDING": "待处理", "UNKNOWN": "结果未确定", "SUCCESS": "系统记录成功", "FAILED": "失败"}


def guidance(order, page, query):
    lines = [f"订单 {order.id} 的售后："]
    progress_only = any(word in query for word in ("进度", "到哪", "状态", "刷新"))
    if order.status == "UNPAID":
        lines.append("订单待支付，无需申请退款；可到订单详情取消待付款订单。")
    elif order.status == "CANCELLED":
        lines.append("订单已取消，请查看已有售后记录；不能据此认定已退款到账。")
    elif "换货" in query:
        lines.append("当前没有在线换货申请入口。请联系订单对应销售者确认换货方案；这里尚未代办。")
    elif progress_only:
        lines.append("入口：我的订单→该订单→售后服务，可查看完整记录。")
    else:
        kind = "退货退款" if "退货" in query else "仅退款或退货退款"
        lines.append(f"入口：商城「我的订单」→该订单→「售后服务」，选择{kind}、商品、数量并填写原因。")
        lines.append("在线申请须核对可退件数、包裹来源；已确认收货的包裹须在确认后7天内申请，以提交校验为准。")
    if "换货" in query and order.status != "PAID":
        lines.append("当前没有在线换货申请入口。")
    if page.items:
        lines.append("本次查询的售后记录：")
        for item in page.items:
            kind = "仅退款" if item.type == "REFUND_ONLY" else "退货退款"
            lines.append(f"售后 {item.id}（{kind}）：{STEPS[item.status]}")
            if item.refund:
                lines.append(f"退款结果：{REFUNDS[item.refund.status]}。")
        if page.nextBeforeId is not None:
            lines.append("这里只展示最近5条；更多记录请到该订单售后页查看。")
    else:
        lines.append("本次未查到售后申请记录；这里没有提交申请。")
    if order.status == "PAID" and not progress_only:
        try:
            _, clauses = load_policy()
            clause_id = "7.2" if "换货" in query else "7.1" if "质量" in query else "6.1"
            selected = [clause for clause in clauses if clause.id == clause_id]
            sentence = next(iter(policy_sentences(selected).values()))[1]
            lines.append(f"政策条件：{sentence}（商城政策第{clause_id}条）")
            if clause_id == "6.1":
                lines.append("须核对商品完好及法定例外；超出线上窗口不等于失去质量救济。")
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            lines.append("政策资料暂不可用，暂不能核对退换货条件。")
    lines.append("当前商城退款为模拟退款，不退真实资金；客服只查询和引导，不代为提交或撤销。")
    return "\n".join(lines)
