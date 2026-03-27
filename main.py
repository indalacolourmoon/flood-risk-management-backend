"""
EcoGuard AI - FastAPI Backend
Serves processed flood analysis data from Google Earth Engine CSVs.
"""

import gzip
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
import shutil

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from core.ml_model import FloodMLModel
from core.processor import FloodProcessor
from typing import Optional

# --- CONFIG ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Global session state (simple for MVP)
PROJECT_METADATA = {"name": "Krishna River Basin"}

def get_project_data_dir(project_name: Optional[str] = None) -> str:
    """Returns the sanitized directory path for a specific project's data."""
    name = project_name or PROJECT_METADATA["name"]
    slug = "".join(c if c.isalnum() else "_" for c in name).lower()
    path = os.path.join(DATA_DIR, slug)
    os.makedirs(path, exist_ok=True)
    return path

def get_year_paths(project_name: Optional[str] = None):
    project_dir = get_project_data_dir(project_name)
    return os.path.join(project_dir, "year1.csv"), os.path.join(project_dir, "year2.csv")

# ---------------------------------------------------------------------------
# Lifespan — runs once at startup
# ---------------------------------------------------------------------------
ml_model = FloodMLModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load saved ML model on startup (if it exists)."""
    loaded = ml_model.load_model()
    if loaded:
        print("✅ ML model loaded from disk.")
    else:
        print("ℹ️  No saved model found. Call /api/train to create one.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EcoGuard AI — Flood Risk API",
    description="Multi-temporal geospatial analysis engine backed by GEE data.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    threshold: float = Field(default=19.0, ge=0, le=200, description="Flood elevation threshold in metres")


class TrainResponse(BaseModel):
    accuracy: float
    precision_flooded: float
    recall_flooded: float
    f1_flooded: float
    training_samples: int
    test_samples: int
    model_status: str


# ---------------------------------------------------------------------------
# Helper — compressed JSON response
# ---------------------------------------------------------------------------
def _gzip_json(data) -> Response:
    payload = json.dumps(data, default=str).encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=6)
    return Response(
        content=compressed,
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


def _get_processor(project_name: str = None) -> FloodProcessor:
    y1, y2 = get_year_paths(project_name)
    if not os.path.exists(y1) or not os.path.exists(y2):
        raise HTTPException(
            status_code=503,
            detail="Dataset files not found. Please upload historical and current data first."
        )
    return FloodProcessor(y1, y2)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    return {"status": "online", "service": "Integrated Flood Risk Management API", "version": "1.1.0"}


@app.get("/api/status", tags=["Data"])
async def get_status(project_name: str = Query(None)):
    """Returns the availability of baseline and current stage data for a project."""
    name = project_name or PROJECT_METADATA["name"]
    y1, y2 = get_year_paths(name)
    return {
        "baseline_present": os.path.exists(y1),
        "current_present": os.path.exists(y2),
        "project_name": name
    }


@app.post("/api/upload", tags=["Data"])
async def upload_data(
    project_name: str = Form("Krishna River Basin"),
    year1: UploadFile = File(None),
    year2: UploadFile = File(...)
):
    """Uploads Yearly CSV files into a project-specific workspace."""
    try:
        PROJECT_METADATA["name"] = project_name
        y1_path, y2_path = get_year_paths(project_name)
        
        # Handle Year 1 (Baseline)
        if year1:
            with open(y1_path, "wb") as buffer:
                shutil.copyfileobj(year1.file, buffer)
        elif not os.path.exists(y1_path):
            raise HTTPException(status_code=400, detail="No baseline (Year 1) found. Please upload it.")

        # Handle Year 2 (Current)
        with open(y2_path, "wb") as buffer:
            shutil.copyfileobj(year2.file, buffer)
        
        # Validation
        processor = FloodProcessor(y1_path, y2_path)
        df = processor.load_and_merge()
        
        return {
            "status": "success",
            "message": f"Project '{project_name}' ready for analysis.",
            "points_count": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data processing failed: {str(e)}")


@app.get("/api/metadata", tags=["Data"])
async def get_metadata():
    return PROJECT_METADATA


@app.get("/health", tags=["Health"])
async def health_check():
    y1, y2 = get_year_paths()
    return {
        "api": "healthy",
        "baseline_ready": os.path.exists(y1),
        "current_ready": os.path.exists(y2),
        "ml_model_loaded": ml_model.model is not None,
    }


@app.get("/api/analysis", tags=["Analysis"])
async def get_analysis(threshold: float = Query(default=19.0, ge=0, le=200)):
    processor = _get_processor()
    processed_df = processor.classify_and_compare(threshold)
    stats = processor.get_summary_stats(processed_df)
    points = processed_df.to_dict(orient="records")

    payload = {
        "threshold": threshold,
        "summary": stats,
        "points": points,
    }
    return _gzip_json(payload)


@app.post("/api/train", tags=["ML"], response_model=TrainResponse)
async def train_model(body: AnalysisRequest):
    processor = _get_processor()
    processed_df = processor.classify_and_compare(body.threshold)

    try:
        metrics = ml_model.train(processed_df, body.threshold)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return metrics


@app.get("/api/predict", tags=["ML"])
async def predict_flood_risk(threshold: float = Query(default=19.0, ge=0, le=200)):
    if ml_model.model is None:
        raise HTTPException(status_code=428, detail="Model not trained. POST /api/train first.")

    processor = _get_processor()
    processed_df = processor.classify_and_compare(threshold)
    result_df = ml_model.predict_probabilities_fast(processed_df)

    points = result_df[[
        "lat", "lng", "elevation_y2", "status_y2",
        "change_analysis", "flood_probability", "risk_level"
    ]].to_dict(orient="records")

    return _gzip_json({"threshold": threshold, "points": points})


@app.get("/api/stats", tags=["Analysis"])
async def get_stats(threshold: float = Query(default=19.0, ge=0, le=200)):
    processor = _get_processor()
    processed_df = processor.classify_and_compare(threshold)
    stats = processor.get_summary_stats(processed_df)
    return {"threshold": threshold, **stats}
