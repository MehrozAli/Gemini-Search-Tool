# File Search FastAPI Service

FastAPI wrapper for the Gemini File Search Tool functions from `SearchTool.py`. It exposes
REST endpoints for creating and managing file search stores, uploading documents, querying
content, and deleting stores.

## 🎉 NEW: Conversational RAG Support

This API now supports **multi-turn conversations with full context awareness**! 

- ✅ Maintain conversation history across queries
- ✅ Ask follow-up questions naturally
- ✅ AI understands context from previous messages
- ✅ Works seamlessly with File Search/RAG

**Quick Start:** See [`FRONTEND_QUICK_START.md`](./FRONTEND_QUICK_START.md) for implementation guide.

**Full Docs:** See [`CONVERSATION_HISTORY_GUIDE.md`](./CONVERSATION_HISTORY_GUIDE.md) for detailed documentation.

## Requirements

- Python 3.11+
- A valid `GOOGLE_API_KEY`

Install dependencies:

```bash
cd fastapi_file_search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with the required Google API key (and optional overrides):

```
GOOGLE_API_KEY=your-key
DEFAULT_STORE_DISPLAY_NAME=my-file-search-store
UPLOAD_POLL_INTERVAL_SECONDS=5
DEFAULT_MODEL=gemini-2.5-flash
```

## Running the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs live at
`/docs` (Swagger) and `/redoc`.

## Key Endpoints

- `GET /api/stores` – list file search stores
- `POST /api/stores` – create a new store
- `POST /api/stores/{store_name}/files` – upload and ingest a file (multipart upload)
- `POST /api/stores/{store_name}/query` – run a prompt against a store using Gemini with
  file grounding. **NEW:** Accepts optional `conversation_history` for multi-turn conversations.
- `DELETE /api/stores` – delete all stores (force)
- `GET /health` – lightweight readiness probe

Each endpoint mirrors the behaviors found in `SearchTool.py`, but wrapped in a
stateless HTTP interface suitable for automation or UI integrations.

## Example: Multi-Turn Conversation

```bash
# First query (no history)
curl -X POST http://localhost:8000/api/stores/my-store/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What products are covered?"}'

# Follow-up query (with history)
curl -X POST http://localhost:8000/api/stores/my-store/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me more about the first one",
    "conversation_history": [
      {"role": "user", "content": "What products are covered?"},
      {"role": "model", "content": "The manuals cover three products..."}
    ]
  }'
```

## Documentation Files

- **[FRONTEND_QUICK_START.md](./FRONTEND_QUICK_START.md)** - Quick guide for frontend developers
- **[CONVERSATION_HISTORY_GUIDE.md](./CONVERSATION_HISTORY_GUIDE.md)** - Complete documentation
- **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[README_CONVERSATION.md](./README_CONVERSATION.md)** - Visual flow diagrams
- **[test_conversation.py](./test_conversation.py)** - Automated test suite
