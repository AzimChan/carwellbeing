# Car Wellbeing Analyzer

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Backend
```bash
python run.py
```
The API will be available at `http://localhost:3000`

### 3. Open the Frontend

Open `http://localhost:3000/` in your web browser

## API Endpoints

### POST `/api/analyze`
Analyze car image for damage using multi-label classification

**Request:**
- `image`: Image file (multipart/form-data)

**Response:**
```json
{
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
    },
    "tips": [
        "Good condition, consider regular cleaning and inspection"
    ],
    "metadata": {
        "filename": "car.jpg",
        "file_size": 1024000,
        "analysis_timestamp": "2024-01-01T12:00:00"
    }
}
```

## Training Your Own Model

To train the model on your own dataset:
```bash
python model_training.py
```
The training script follows a 5-step process:
1) Turn Dataset images into tensors (There is 300 images in roboflow dataset)
2) Use pretrained model (Resnet-50, augmentation)
3) Check if there is any mistakes 
4) Retrain model based on the mistake 
5) Retrain model on the same dataset again to get better results

## Tech Stack

- **Frontend**: Tailwind CSS
- **Backend**: Python FastAPI with PyTorch
- **AI Model**: ResNet-50 for multi-label car damage classification
- **Computer Vision**: PyTorch and OpenCV for image analysis

