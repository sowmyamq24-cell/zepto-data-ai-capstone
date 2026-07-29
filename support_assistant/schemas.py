"""schemas.py — Pydantic request/response models for the /ask endpoint."""

from typing import List

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
