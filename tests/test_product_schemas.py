import json
import unittest
from decimal import Decimal
from pydantic import ValidationError
from agent.tool.schemas import FilterParams, DiscoveryRequest, ProductListRequest, ProductDetail
from agent.memory import MemoryManager
from langchain_core.messages import HumanMessage

class ProductSchemaTests(unittest.TestCase):
    def test_decimal_roundtrip_and_cancelled_color(self):
        params=FilterParams.model_validate_json('{"min_price":0.10,"max_price":299.99,"color":null}')
        self.assertEqual(params.min_price,Decimal("0.10"))
        self.assertEqual(params.max_price,Decimal("299.99"))
        saved=params.model_dump(mode="json",exclude={"next_page"})
        self.assertIsNone(saved["color"])
        self.assertIsInstance(saved["max_price"],str)
        messages=MemoryManager().build_messages({"messages":[HumanMessage(content="再看看")],"intent":"商品推荐","task_context":{"intent":"商品推荐","filters":saved}},"规则")
        self.assertTrue(any("299.99" in message.content for message in messages))
        self.assertEqual(FilterParams.model_validate(saved).max_price,Decimal("299.99"))

    def test_invalid_ranges_rejected_once_at_schema(self):
        for values in ({"min_price":"500","max_price":"300"},{"max_price":"NaN"},{"max_price":"Infinity"},{"min_price":-1}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                FilterParams(**values)
        with self.assertRaises(ValidationError):
            DiscoveryRequest(minPrice="500",maxPrice="300")

    def test_query_contract_and_cursor_types(self):
        for values in ({"color":"黑色"},{"size":101},{"size":True},{"categoryId":0},{"cursor":12}):
            with self.subTest(values=values),self.assertRaises(ValidationError):
                DiscoveryRequest(**values)
        with self.assertRaises(ValidationError):
            ProductListRequest(afterId="12")
        self.assertEqual(DiscoveryRequest(cursor="opaque").cursor,"opaque")
        self.assertEqual(ProductListRequest(afterId=12).afterId,12)

    def test_nullable_description_is_not_nullable_price_or_availability(self):
        data={"id":101,"name":"耳机","description":None,"options":[],"skus":[{"id":1001,"optionValueIds":[],"price":"299.99","available":True}]}
        self.assertIsNone(ProductDetail.model_validate(data).description)
        for name in ("price","available"):
            invalid={**data,"skus":[{**data["skus"][0],name:None}]}
            with self.subTest(name=name),self.assertRaises(ValidationError):
                ProductDetail.model_validate(invalid)
