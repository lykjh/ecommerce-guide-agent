"""商品业务的结构化输入契约。"""
from decimal import Decimal
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[int, Field(gt=0, strict=True)]

class FilterParams(BaseModel):
    category: Optional[str] = Field(default=None, max_length=80, description="商品品类；结合现有条件，用户取消时填JSON null")
    min_price: Optional[Money] = Field(default=None, description="最低价格；用户取消时填JSON null")
    max_price: Optional[Money] = Field(default=None, description="最高价格；用户取消时填JSON null")
    color: Optional[str] = Field(default=None, max_length=40, description="颜色；用户取消时填JSON null")
    next_page: bool = Field(default=False, description="只有用户明确要求下一批或更多候选时为true；首次查询或修改筛选条件时为false")
    unsupported_specs: list[str] = Field(default_factory=list, description="用户仍要求用于筛选的尺码、容量、版本等非价格/颜色规格；取消后清空。仅咨询参数不填写")

    @model_validator(mode="after")
    def ordered_prices(self):
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("最低价格不能超过最高价格")
        return self


class ProductQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categoryId: Identifier | None = None
    brandId: Identifier | None = None
    shopId: Identifier | None = None
    minPrice: Money | None = None
    maxPrice: Money | None = None
    size: Annotated[int, Field(ge=1, le=100, strict=True)] = 3

    @model_validator(mode="after")
    def ordered_prices(self):
        if self.minPrice is not None and self.maxPrice is not None and self.minPrice > self.maxPrice:
            raise ValueError("最低价格不能超过最高价格")
        return self


class DiscoveryRequest(ProductQuery):
    q: str | None = Field(default=None, max_length=100)
    cursor: str | None = None


class ProductListRequest(ProductQuery):
    afterId: Identifier | None = None


class ProductCard(BaseModel):
    id: Identifier
    name: str
    startingPrice: Money | None
    soldOut: Annotated[bool, Field(strict=True)]


class DiscoveryPage(BaseModel):
    items: list[ProductCard]
    hasMore: Annotated[bool, Field(strict=True)]
    nextCursor: str | None


class ProductPage(BaseModel):
    items: list[ProductCard]
    size: Annotated[int, Field(ge=1, le=100, strict=True)]
    hasMore: Annotated[bool, Field(strict=True)]
    nextCursor: Identifier | None


class OptionValue(BaseModel):
    id: Identifier
    value: str


class ProductOption(BaseModel):
    id: Identifier
    name: str
    values: list[OptionValue]


class ProductSku(BaseModel):
    id: Identifier
    optionValueIds: list[Identifier]
    price: Money
    available: Annotated[bool, Field(strict=True)]


class ProductDetail(BaseModel):
    id: Identifier
    name: str
    description: str | None
    options: list[ProductOption]
    skus: list[ProductSku]


class CandidateRef(BaseModel):
    product_id: Identifier
    name: str


class ProductMatch(BaseModel):
    product: ProductDetail
    skus: list[ProductSku]


class ProductSelection(BaseModel):
    matches: list[ProductMatch]
    page: DiscoveryPage | ProductPage
    checked_product_ids: list[int]
    unverified_product_ids: list[int]
    alternative_colors: list[str] = Field(default_factory=list)


class CategoryRequest(BaseModel):
    parentId: Identifier | None = None


class CategoryNav(BaseModel):
    id: Identifier
    name: str
    level: Annotated[int, Field(strict=True)]
    hasChildren: Annotated[bool, Field(strict=True)]
    sortOrder: Annotated[int, Field(strict=True)]


class ProductResolution(BaseModel):
    status: Literal["resolved", "choose", "not_found"]
    product: ProductDetail | None = None
    candidates: list[CandidateRef] = Field(default_factory=list)


class MallError(Exception):
    """仅携带可记录的错误分类，不携带上游响应正文或凭据。"""
    def __init__(self, kind: str, *, http_status: int | None = None, code: int | None = None):
        super().__init__(kind)
        self.kind = kind
        self.http_status = http_status
        self.code = code

