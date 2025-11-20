"""API routes for file search management."""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas.file_search import (
    CreateStoreRequest,
    CreateStoreResponse,
    DeleteStoreResponse,
    DeleteStoresResponse,
    QueryRequest,
    QueryResponse,
    StoreListResponse,
    SyncRequest,
    SyncResponse,
    UploadFileResponse,
)
from app.services.file_search_service import FileSearchService

router = APIRouter(prefix="/api", tags=["file-search"])


def get_service() -> FileSearchService:
    return FileSearchService()


UPLOAD_MESSAGES = ("File uploaded and ingested", "File upload in progress")
SYNC_MESSAGES = ("File synced and ingested", "File sync in progress")


def _ingest_response(operation: dict[str, Any], messages: tuple[str, str]) -> UploadFileResponse:
    done = bool(operation.get("done"))
    return UploadFileResponse(
        operation_name=operation.get("name"),
        done=done,
        message=messages[0] if done else messages[1],
    )


@router.get("/stores", response_model=StoreListResponse)
async def list_stores(service: FileSearchService = Depends(get_service)) -> StoreListResponse:
    stores = await run_in_threadpool(service.list_stores)
    return StoreListResponse(stores=stores)


@router.post("/stores", response_model=CreateStoreResponse, status_code=201)
async def create_store(
    payload: CreateStoreRequest,
    service: FileSearchService = Depends(get_service),
) -> CreateStoreResponse:
    result = await run_in_threadpool(service.create_store, payload.display_name)
    ingest = result.get("ingest")
    return CreateStoreResponse(
        store=result["store"],
        ingest=_ingest_response(ingest, UPLOAD_MESSAGES) if ingest else None,
    )


@router.post("/stores/{store_name:path}/files", response_model=UploadFileResponse)
async def upload_file(
    store_name: str,
    file: UploadFile = File(...),
    display_name: Annotated[str | None, Query()] = None,
    service: FileSearchService = Depends(get_service),
) -> UploadFileResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    suffix = os.path.splitext(file.filename)[1]
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name

        result = await run_in_threadpool(
            service.upload_file,
            store_name,
            temp_file_path,
            display_name or file.filename,
        )
        return _ingest_response(result, UPLOAD_MESSAGES)
    finally:
        await file.close()
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/stores/{store_name:path}/sync", response_model=SyncResponse)
async def sync_store(
    store_name: str,
    payload: SyncRequest,
    service: FileSearchService = Depends(get_service),
) -> SyncResponse:
    result = await run_in_threadpool(service.sync_graph_document, store_name, payload.display_name)
    return SyncResponse(store=result["store"], ingest=_ingest_response(result["ingest"], SYNC_MESSAGES))


@router.post("/stores/{store_name:path}/query", response_model=QueryResponse)
async def query_store(
    store_name: str,
    payload: QueryRequest,
    service: FileSearchService = Depends(get_service),
) -> QueryResponse:
    history = [message.model_dump() for message in payload.conversation_history or []]
    result = await run_in_threadpool(
        service.query_store,
        store_name,
        payload.prompt,
        payload.model,
        payload.system_prompt,
        history,
    )
    return QueryResponse(**result)


@router.delete("/stores/{store_name:path}", response_model=DeleteStoreResponse)
async def delete_store(
    store_name: str,
    service: FileSearchService = Depends(get_service),
) -> DeleteStoreResponse:
    deleted = await run_in_threadpool(service.delete_store, store_name)
    return DeleteStoreResponse(deleted=deleted, message=f"Store '{store_name}' deleted successfully")


@router.delete("/stores", response_model=DeleteStoresResponse)
async def delete_stores(service: FileSearchService = Depends(get_service)) -> DeleteStoresResponse:
    deleted = await run_in_threadpool(service.delete_all_stores)
    return DeleteStoresResponse(deleted=deleted)
