"""Pydantic schemas for request and response payloads."""
from typing import List, Optional

from pydantic import BaseModel, Field


class Store(BaseModel):
    name: str = Field(..., description="Full resource name of the store")
    display_name: Optional[str] = Field(None, description="Human-friendly store label")
    create_time: Optional[str] = Field(None, description="Creation timestamp from the API")


class StoreListResponse(BaseModel):
    stores: List[Store]


class CreateStoreRequest(BaseModel):
    display_name: Optional[str] = Field(None, description="Display name for the store")


class CreateStoreResponse(BaseModel):
    store: Store


class UploadFileResponse(BaseModel):
    operation_name: Optional[str] = None
    done: bool = False
    message: str


class QueryRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to submit to Gemini")
    model: Optional[str] = Field(None, description="Override the default Gemini model")
    system_prompt: Optional[str] = Field(None, description="System prompt to guide the model's behavior")


class QuerySource(BaseModel):
    title: Optional[str] = None
    uri: Optional[str] = None
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    text: str
    sources: List[QuerySource]


class DeleteStoresResponse(BaseModel):
    deleted: List[str]


class DeleteStoreResponse(BaseModel):
    deleted: str = Field(..., description="Name of the deleted store")
    message: str = Field(..., description="Confirmation message")


class HealthResponse(BaseModel):
    status: str
