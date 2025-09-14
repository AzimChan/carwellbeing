# Car Wellbeing Analyzer
## Tech Stack

- **Frontend**: Tailwind CSS
- **Backend**: Python FastAPI with PyTorch
- **AI Model**: ResNet-50 for multi-label car damage classification
- **Computer Vision**: PyTorch and OpenCV for image analysis

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

## AI Model Details

### Multi-label Damage Classification
The system uses a trained ResNet-50 model to classify car images into 4 categories simultaneously:

- **Car** (Class 0): Clean car condition
- **Dent** (Class 1): Dents and dings
- **Rust** (Class 2): Rust and corrosion
- **Scratch** (Class 3): Scratches and paint damage

### Multi-label Features
- **Sigmoid activation** for independent probabilities
- **Threshold-based detection** (0.5 threshold)
- **Multiple damage detection** per image
- **Real damage percentages** for each type

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
1. Convert dataset images to tensors
2. Use pretrained ResNet-50 model
3. Check for mistakes in validation
4. Retrain model based on mistakes
5. Retrain on same dataset for better results
