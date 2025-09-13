"""
Text moderation endpoints for Muzzle API.

This module handles all text-related content moderation functionality.
"""

from fastapi import APIRouter
from pydantic import BaseModel

# Create router for text endpoints
router = APIRouter()


# Request model for text moderation
class TextModerationRequest(BaseModel):
    text: str


# Response model for text moderation
class TextModerationResponse(BaseModel):
    text: str
    is_toxic: bool
    confidence: float
    labels: list[str]


@router.post("/moderate/text", response_model=TextModerationResponse)
async def moderate_text(request: TextModerationRequest):
    """
    Analyze text for toxicity and hate speech.

    This endpoint accepts text content and returns toxicity analysis
    including confidence scores and labels.
    """
    # Simple mock logic for now
    is_toxic = "bad" in request.text.lower() or "hate" in request.text.lower()

    return TextModerationResponse(
        text=request.text,
        is_toxic=is_toxic,
        confidence=0.9 if is_toxic else 0.1,
        labels=["toxic"] if is_toxic else ["safe"]
    )
