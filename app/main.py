"""FastAPI application entry point."""

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors

from app.api.routes import router as file_search_router
from app.core.errors import ConfigurationError
from app.schemas.file_search import HealthResponse

app = FastAPI(title="Gemini File Search API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(file_search_router)

# Gemini client errors that mean something to the caller keep their status;
# anything else from upstream is reported as a bad gateway.
PASSTHROUGH_STATUSES = {400, 404, 429}


@app.exception_handler(genai_errors.APIError)
async def gemini_error_handler(_request: Request, exc: genai_errors.APIError) -> JSONResponse:
    status = exc.code if exc.code in PASSTHROUGH_STATUSES else 502
    return JSONResponse(status_code=status, content={"detail": f"Gemini API error: {exc.message}"})


@app.exception_handler(requests.RequestException)
async def notion_error_handler(_request: Request, exc: requests.RequestException) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": f"Notion API error: {exc}"})


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(_request: Request, exc: ConfigurationError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/", response_model=HealthResponse, tags=["health"])
@app.get("/health", response_model=HealthResponse, tags=["health"])
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")
