# Car Wellbeing Analyzer

A modern web application for analyzing car cleanness and integrity from uploaded images.

## Features

- **Drag & Drop Image Upload**: Easy file upload with drag-and-drop functionality
- **Real-time Image Preview**: See your uploaded image before analysis
- **Dual Analysis**: Get both cleanness and integrity scores for your car
- **Visual Progress Bars**: Clear visual representation of analysis results
- **Personalized Tips**: Get maintenance recommendations based on analysis
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Built with Tailwind CSS and Shadcn/UI components
- **FastAPI Backend**: Robust Python API with computer vision analysis

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Styling**: Tailwind CSS with Shadcn/UI design system
- **Backend**: Python FastAPI with OpenCV
- **Computer Vision**: OpenCV for image analysis

## Project Structure

```
carwellbeing/
├── index.html          # Main HTML file with UI layout
├── script.js           # Frontend JavaScript logic
├── main.py             # FastAPI backend application
├── run.py              # Simple server runner script
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── LICENSE             # License file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Backend

```bash
python main.py
# or
python run.py
```

The API will be available at `http://localhost:8000`

### 3. Open the Frontend

Open `index.html` in your web browser or serve it with a local server:

```bash
# Using Python's built-in server
python -m http.server 3000

# Then open http://localhost:3000
```

### 4. Test the Application

1. Upload a car image using drag-and-drop or click to browse
2. Click "Analyze Car" to get real analysis results
3. View the scores, progress bars, and personalized tips

## API Endpoints

### POST `/api/analyze`
Analyze car image for cleanness and integrity

**Request:**
- `image`: Image file (multipart/form-data)

**Response:**
```json
{
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
        "filename": "car.jpg",
        "file_size": 1024000,
        "image_dimensions": [480, 640],
        "analysis_timestamp": "2024-01-01T12:00:00"
    }
}
```

### POST `/api/analyze-base64`
Analyze car image from base64 encoded data

**Request:**
```json
{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

### GET `/health`
Health check endpoint

### GET `/`
API information and status

## Analysis Algorithm

The system uses computer vision techniques to analyze:

### Cleanness Analysis:
- **Dirt Level**: Analyzes image darkness and texture variation
- **Scratches**: Uses edge detection to identify surface scratches
- **Paint Condition**: Evaluates color consistency across regions
- **Interior Cleanliness**: Basic interior area analysis

### Integrity Analysis:
- **Structural Damage**: Contour analysis to detect major damage
- **Dents**: Surface variation analysis using Gaussian blur
- **Rust**: Color range detection in HSV space
- **Glass Condition**: Basic glass area assessment

## Configuration

### CORS Settings
The API is configured to allow all origins for development. In production, update the CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Analysis Weights
You can adjust the analysis weights in the `CarAnalyzer` class:

```python
self.cleanness_weights = {
    'dirt_level': 0.4,
    'scratches': 0.3,
    'paint_condition': 0.2,
    'interior_cleanliness': 0.1
}
```

## Development

### Running in Development Mode
```bash
# Backend with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (if serving from server)
python -m http.server 3000
```

### API Documentation
Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Production Deployment

### Using Docker (Recommended)
Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## License

See LICENSE file for details.