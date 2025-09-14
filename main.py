import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
from typing import Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarAnalysisModel(nn.Module):
    """ResNet-50 based model for car damage analysis"""
    def __init__(self, num_classes: int = 4):  # 4 classes: car, dent, rust, scratch
        super(CarAnalysisModel, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=False)  # We'll load our trained weights
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Initialize model with the same architecture as training
model = CarAnalysisModel(num_classes=4)

# Load saved weights
model.load_state_dict(torch.load("trained_car_damage_model.pth", map_location=torch.device('cpu')))
model.eval()  # put model in inference mode

# Define transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # match training size
    transforms.ToTensor(),
    transforms.Normalize(            # use ImageNet normalization
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

import sys
import json

def analyze_image(image_path):
    """Analyze a single image and return results as JSON"""
    try:
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # add batch dimension

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs[0])  # Use sigmoid for multi-label

        # Map class indices to damage types
        class_names = ['car', 'dent', 'rust', 'scratch']

        # Create probability dictionary
        all_probabilities = {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }

        # Determine detected damages (threshold > 0.5)
        detected_damages = []
        for i, class_name in enumerate(class_names):
            if probs[i] > 0.5:
                detected_damages.append({
                    'type': class_name,
                    'probability': float(probs[i])
                })

        # Return JSON result
        result = {
            "damage_analysis": {
                "all_probabilities": all_probabilities,
                "detected_damages": detected_damages,
                "analysis_type": "multi_label"
            }
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode - analyze provided image
        image_path = sys.argv[1]
        result = analyze_image(image_path)
        print(json.dumps(result))
    else:
        # Test mode - use default image
        image_path = "dataset/test/images.jpg"
        result = analyze_image(image_path)
        print(json.dumps(result))
