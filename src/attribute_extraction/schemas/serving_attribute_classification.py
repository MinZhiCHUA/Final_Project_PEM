from typing import List, Optional, Tuple

from pydantic import BaseModel


class LOV(BaseModel):
    lov_code: str
    lov_label: str


class QueryInfo(BaseModel):
    raw_record_id: Optional[int]
    title: str
    description: str
    product_metadata: Optional[List[Tuple[str, str]]]
    attribute_code: str
    lov_values_translated: Optional[List[LOV]]


class Query(BaseModel):
    instances: List[QueryInfo]