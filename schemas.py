# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ProductItem(BaseModel):
    description: Optional[str] = Field(None, description="The name or description of the product/service.")
    quantity: Optional[float] = Field(None, description="The quantity.")
    rate: Optional[float] = Field(None, description="The price per unit.")
    amount: Optional[float] = Field(None, description="The total amount for this line item.")

class Invoice(BaseModel):
    vendor_name: Optional[str] = Field(None, description="The name of the vendor/hotel.")
    invoice_number: Optional[str] = Field(None, description="The invoice or bill number.")
    invoice_date: Optional[str] = Field(None, description="The date of the invoice in YYYY-MM-DD format.")
    buyer_name: Optional[str] = Field(None, description="The name of the guest or company buying the service.")
    total_amount: Optional[float] = Field(None, description="The final net amount or grand total of the invoice.")
    line_items: List[ProductItem] = Field(default_factory=list, description="A list of all products or services on the invoice.")