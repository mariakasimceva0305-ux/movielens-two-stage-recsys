from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_bundle, recommend_for_user


APP_ARTIFACTS_DIR = Path("artifacts")

app = FastAPI(title="MovieLens Recommendation API", version="0.1.0")

_bundle = None


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="MovieLens user id")
    k: int = Field(default=10, ge=1, le=100)


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    score: Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id: int
    k: int
    recommendations: List[RecommendationItem]


@app.on_event("startup")
def startup_event() -> None:
    global _bundle
    if APP_ARTIFACTS_DIR.exists():
        try:
            _bundle = load_bundle(APP_ARTIFACTS_DIR)
        except Exception:
            _bundle = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifacts_loaded": _bundle is not None}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded. Run training first and place artifacts under ./artifacts.",
        )

    recs = recommend_for_user(_bundle, user_id=request.user_id, k=request.k)
    return RecommendationResponse(user_id=request.user_id, k=request.k, recommendations=recs)
