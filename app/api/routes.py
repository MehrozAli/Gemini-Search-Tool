"""API routes for file search management."""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from fastapi.concurrency import run_in_threadpool

from app.schemas.file_search import (
    CreateStoreRequest,
    CreateStoreResponse,
    DeleteStoreResponse,
    DeleteStoresResponse,
    QueryRequest,
    QueryResponse,
    StoreListResponse,
    UploadFileResponse,
    SyncRequest,
    SyncResponse,
)
from app.services.file_search_service import FileSearchService

router = APIRouter(prefix="/api", tags=["file-search"])


def get_service() -> FileSearchService:
    return FileSearchService()


@router.get("/stores", response_model=StoreListResponse)
async def list_stores(service: FileSearchService = Depends(get_service)) -> StoreListResponse:
    stores = await run_in_threadpool(service.list_stores)
    return StoreListResponse(stores=stores)


@router.post("/stores", response_model=CreateStoreResponse, status_code=201)
async def create_store(
    payload: CreateStoreRequest,
    service: FileSearchService = Depends(get_service),
) -> CreateStoreResponse:
    try:
        result = await run_in_threadpool(service.create_store, payload.display_name)
        ingest = result.get("ingest") or {}
        message = "File uploaded and ingested" if ingest.get("done") else "File upload in progress"
        ingest_resp = None
        if ingest:
            ingest_resp = UploadFileResponse(
                operation_name=ingest.get("name"),
                done=bool(ingest.get("done")),
                message=message,
            )
        return CreateStoreResponse(store=result["store"], ingest=ingest_resp)
    except Exception as exc:  # pragma: no cover - surface clean error upstream
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stores/{store_name:path}/files", response_model=UploadFileResponse)
async def upload_file(
    store_name: str,
    file: UploadFile = File(...),
    display_name: Annotated[str | None, Query()] = None,
    service: FileSearchService = Depends(get_service),
) -> UploadFileResponse:
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename")

    await file.seek(0)
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
            display_name,
        )
        print("Result: ", result)
        message = "File uploaded and ingested" if result.get("done") else "File upload in progress"
        return UploadFileResponse(
            operation_name=result.get("name"),
            done=bool(result.get("done")),
            message=message,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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
    try:
        result = await run_in_threadpool(
            service.sync_graph_document,
            store_name,
            payload.display_name,
        )
        ingest = result.get("ingest") or {}
        message = "File synced and ingested" if ingest.get("done") else "File sync in progress"
        ingest_resp = UploadFileResponse(
            operation_name=ingest.get("name"),
            done=bool(ingest.get("done")),
            message=message,
        )
        return SyncResponse(store=result["store"], ingest=ingest_resp)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stores/{store_name:path}/query", response_model=QueryResponse)
async def query_store(
    store_name: str,
    payload: QueryRequest,
    service: FileSearchService = Depends(get_service),
) -> QueryResponse:
    try:
        # Convert Pydantic models to dicts for conversation history
        conversation_history = None
        if payload.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in payload.conversation_history
            ]
        
        result = await run_in_threadpool(
            service.query_store,
            store_name,
            payload.prompt,
            payload.model,
            payload.system_prompt,
            conversation_history  # Pass conversation history
        )
        return QueryResponse(**result)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/stores/{store_name:path}", response_model=DeleteStoreResponse)
async def delete_store(
    store_name: str,
    service: FileSearchService = Depends(get_service),
) -> DeleteStoreResponse:
    try:
        deleted = await run_in_threadpool(service.delete_store, store_name)
        return DeleteStoreResponse(
            deleted=deleted,
            message=f"Store '{store_name}' deleted successfully"
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/stores", response_model=DeleteStoresResponse)
async def delete_stores(service: FileSearchService = Depends(get_service)) -> DeleteStoresResponse:
    deleted = await run_in_threadpool(service.delete_all_stores)
    return DeleteStoresResponse(deleted=deleted)
