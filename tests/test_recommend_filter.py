import unittest

from agent.tool.schemas import FilterParams
from agent.retrieval import select_recommendation_evidence


class RecommendFilterTests(unittest.TestCase):
    def test_only_keeps_products_matching_price_and_color(self):
        context="""
【G-0001】耳机甲
商品名称：耳机甲
类目：手机数码 / 蓝牙耳机
价格：299 元
规格：黑色，入耳式

【G-0002】耳机乙
商品名称：耳机乙
类目：手机数码 / 蓝牙耳机
价格：399 元
规格：白色，入耳式

【G-0004】鼠标甲
商品名称：鼠标甲
类目：电脑办公 / 鼠标
价格：199 元
规格：黑色，无线

【G-0003】耳机丙
商品名称：耳机丙
类目：手机数码 / 蓝牙耳机
价格：599 元
规格：黑色，入耳式
        """

        filters=FilterParams(
            category="耳机",
            min_price=None,
            max_price=500,
            color="黑色",
        )

        result=select_recommendation_evidence(context,filters)

        print(result)
        self.assertIn("【G-0001】", result)
        self.assertNotIn("【G-0002】", result)
        self.assertNotIn("【G-0003】", result)
        self.assertNotIn("【G-0004】", result)