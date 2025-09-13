# Muzzle API Layer

This directory contains the FastAPI route handlers and endpoint organization for the Muzzle content moderation API.

## Structure

```
api/
├── __init__.py       # Package initialization
└── endpoints/        # Individual endpoint modules
    ├── __init__.py   # Endpoints package initialization
    ├── text.py       # Text moderation endpoints
    └── image.py      # Image moderation endpoints
```

## Endpoint Organization

Each endpoint file contains:
- **Pydantic models** for request/response validation
- **APIRouter instance** for modular route organization
- **Endpoint functions** with proper async/await patterns

### Text Moderation (`text.py`)

**Endpoint:** `POST /api/moderate/text`

**Request Model:**
```python
class TextModerationRequest(BaseModel):
    text: str
```

**Response Model:**
```python
class TextModerationResponse(BaseModel):
    text: str
    is_toxic: bool
    confidence: float
    labels: list[str]
```

**Example Usage:**
```bash
curl -X POST "http://localhost:8000/api/moderate/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message"}'
```

### Image Moderation (`image.py`)

**Endpoint:** `POST /api/moderate/image`

**Request:** File upload using `UploadFile`

**Response Model:**
```python
class ImageModerationResponse(BaseModel):
    filename: str
    comment: str
    toxicity_score: float
    nsfw_score: float
    is_safe: bool
```

**Example Usage:**
```bash
curl -X POST "http://localhost:8000/api/moderate/image" \
  -F "image=@path/to/your/image.jpg"
```

## Router Integration

Each endpoint file exports a `router` that gets included in the main FastAPI application:

```python
# In main.py
from muzzle.api.endpoints import text, image

app.include_router(text.router, prefix="/api", tags=["text-moderation"])
app.include_router(image.router, prefix="/api", tags=["image-moderation"])
```

## Current Implementation

- **Mock responses** - Currently returns hardcoded responses for testing
- **Basic validation** - Pydantic models ensure proper request/response format
- **Async patterns** - All endpoints use `async def` for non-blocking operations

## Next Steps

1. **ML Model Integration** - Replace mock logic with actual model inference
2. **Error Handling** - Add comprehensive error responses and validation
3. **Batch Endpoints** - Add endpoints for processing multiple items
4. **Authentication** - Add API key or JWT-based authentication
5. **Rate Limiting** - Implement request rate limiting per client
6. **Logging** - Add structured logging for monitoring and debugging

## Adding New Endpoints

To add a new endpoint:

1. Create a new file in `endpoints/` (e.g., `video.py`)
2. Define Pydantic request/response models
3. Create an `APIRouter` instance
4. Implement endpoint functions with proper async patterns
5. Include the router in `main.py`

Example template:
```python
from fastapi import APIRouter
from pydantic import BaseModel

class YourRequest(BaseModel):
    # Define request fields
    pass

class YourResponse(BaseModel):
    # Define response fields
    pass

router = APIRouter()

@router.post("/your-endpoint", response_model=YourResponse)
async def your_endpoint(request: YourRequest):
    # Implement your logic
    return YourResponse(...)
```
