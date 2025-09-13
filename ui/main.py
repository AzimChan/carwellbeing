from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Car Wellbeing Analyzer API",
    description="API for analyzing car cleanness and integrity from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Car Wellbeing Analyzer API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/analyze")
async def analyze_car(file: UploadFile = File(...)):
    """
    Analyze car image for cleanness and integrity
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Validate file size (max 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size must be less than 10MB"
            )
        
        # Read image data
        image_data = await file.read()
        
        # TODO: Add your analysis logic here
        # For now, return mock data
        results = {
            "cleanness": {
                "score": 85,
                "description": "Very good condition with minor cleaning needed."
            },
            "integrity": {
                "score": 92,
                "description": "Excellent structural integrity, no visible damage."
            },
            "tips": [
                "Consider a professional car wash and interior detailing",
                "Regular washing prevents paint damage and maintains value"
            ],
            "metadata": {
                "filename": file.filename,
                "file_size": len(image_data),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Analysis completed for {file.filename}")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )

@app.post("/api/analyze-base64")
async def analyze_car_base64(data: dict):
    """
    Analyze car image from base64 encoded data
    """
    try:
        base64_data = data.get("image")
        if not base64_data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # TODO: Add your base64 analysis logic here
        # For now, return mock data
        results = {
            "cleanness": {
                "score": 78,
                "description": "Good condition, but could use a thorough cleaning."
            },
            "integrity": {
                "score": 88,
                "description": "Very good condition with minor cosmetic issues."
            },
            "tips": [
                "Schedule a professional inspection for any visible damage",
                "Address minor issues before they become major problems"
            ],
            "metadata": {
                "file_size": len(base64_data),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during base64 analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )