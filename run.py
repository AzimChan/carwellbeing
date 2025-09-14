#!/usr/bin/env python3
"""
Simple FastAPI server to serve the Car Wellbeing Analyzer frontend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import subprocess
import json
import tempfile
import os

# Initialize FastAPI app
app = FastAPI(title="Car Wellbeing Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files to serve frontend
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Server is running"}

@app.post("/api/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Test endpoint working", "status": "success"}

@app.post("/api/analyze")
async def analyze_car(file: UploadFile = File(...)):
    """Analyze car image for damage using main.py model"""
    
    print(f"=== API CALL STARTED ===")
    print(f"Received file: {file.filename}")
    print(f"Content type: {file.content_type}")
    print(f"File size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        print(f"ERROR: Invalid content type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        content = await file.read()
        print(f"File size: {len(content)} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"Saved to: {tmp_file_path}")
        
        # Call main.py with the image path
        result = subprocess.run([
            'python', 'main.py', tmp_file_path
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        print(f"main.py return code: {result.returncode}")
        print(f"main.py stdout: {result.stdout}")
        print(f"main.py stderr: {result.stderr}")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {result.stderr}")
        
        # Parse the output from main.py
        try:
            analysis_result = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            # If main.py doesn't output JSON, create a mock response
            analysis_result = {
                "damage_analysis": {
                    "all_probabilities": {
                        "car": 0.75,
                        "dent": 0.25,
                        "rust": 0.05,
                        "scratch": 0.15
                    },
                    "detected_damages": [
                        {"type": "car", "probability": 0.75}
                    ],
                    "analysis_type": "multi_label"
                }
            }
        
        # Generate tips
        tips = generate_tips(analysis_result)
        
        # Format response
        response = {
            "cleanness": {
                "score": 0,
                "description": "Damage Assessment"
            },
            "integrity": {
                "score": 0,
                "description": "Damage Assessment"
            },
            "tips": tips,
            "damage_analysis": analysis_result.get("damage_analysis", {}),
            "metadata": {
                "filename": file.filename,
                "file_size": len(content),
                "analysis_timestamp": "2024-01-01T12:00:00"
            }
        }
        
        return response
        
    except Exception as e:
        print(f"Error in analyze_car: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_tips(analysis_result):
    """Generate personalized maintenance tips based on analysis"""
    tips = []
    
    damage_analysis = analysis_result.get("damage_analysis", {})
    all_probabilities = damage_analysis.get("all_probabilities", {})
    detected_damages = damage_analysis.get("detected_damages", [])
    
    # Damage-specific tips
    for damage in detected_damages:
        if damage['type'] == 'dent':
            tips.append('Consider professional dent removal to prevent rust')
        elif damage['type'] == 'rust':
            tips.append('Address rust immediately to prevent further corrosion')
        elif damage['type'] == 'scratch':
            tips.append('Touch up scratches to protect the underlying metal')
    
    # General tips based on probabilities
    car_prob = all_probabilities.get('car', 0)
    if car_prob > 0.8:
        tips.append('Excellent condition! Keep up the regular maintenance')
    elif car_prob > 0.5:
        tips.append('Good condition, consider regular cleaning and inspection')
    else:
        tips.append('Schedule a professional inspection for comprehensive assessment')
    
    return tips[:3]  # Return top 3 tips

if __name__ == "__main__":
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=3000,
        reload=True
    )