# Vibecoded

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarDataset(Dataset):
    """Dataset class for car images with multi-label support"""
    def __init__(self, image_paths: List[str], labels: List[List[int]], transform=None):
        self.image_paths = image_paths
        self.labels = labels  # Now each label is a list of 0s and 1s
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Multi-label tensor
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CarAnalysisModel(nn.Module):
    """ResNet-50 based model for car damage analysis"""
    def __init__(self, num_classes: int = 4):  # 4 classes: car, dent, rust, scratch
        super(CarAnalysisModel, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers (optional - you can unfreeze for fine-tuning)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Enable gradients for the new layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)

class ModelTrainer:
    """Main class implementing your 5-step guide"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.transform = None
        
        logger.info(f"Using device: {self.device}")
    
    def step1_convert_images_to_tensors(self, dataset_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Step 1: Turn Dataset images into tensors
        """
        logger.info("Step 1: Converting dataset images to tensors...")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        train_image_paths, train_labels = self._load_dataset(dataset_path, 'train')
        val_image_paths, val_labels = self._load_dataset(dataset_path, 'valid')
        
        # Create datasets
        train_dataset = CarDataset(train_image_paths, train_labels, self.transform)
        val_dataset = CarDataset(val_image_paths, val_labels, self.transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Created train loader with {len(train_dataset)} samples")
        logger.info(f"Created val loader with {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def step2_use_pretrained_model(self, pretrained_model_path: str = None):
        """
        Step 2: Use pretrained model (ResNet-50)
        """
        logger.info("Step 2: Loading ResNet-50 pretrained model...")
        
        # Initialize ResNet-50 model
        self.model = CarAnalysisModel()
        
        # Load your custom pretrained weights if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
            logger.info(f"Loaded custom pretrained weights from {pretrained_model_path}")
        else:
            logger.info("Using ImageNet pretrained ResNet-50 weights")
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer and criterion
        # Only optimize the new fully connected layer initially
        self.optimizer = optim.Adam(self.model.backbone.fc.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()  # Better for multi-label classification
        
        logger.info("ResNet-50 model setup completed")
    
    def step3_check_mistakes(self, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Step 3: Check if there is any mistakes
        """
        logger.info("Step 3: Checking for mistakes in validation...")
        
        self.model.eval()
        mistakes = []
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # For multi-label: use sigmoid and threshold
                predictions = torch.sigmoid(outputs) > 0.5
                correct = (predictions == labels).all(dim=1).sum().item()
                correct_predictions += correct
                total_predictions += labels.size(0)
                
                # Identify specific mistakes
                mistake_indices = (predictions != labels).any(dim=1).nonzero(as_tuple=True)[0]
                for idx in mistake_indices:
                    mistakes.append({
                        'batch_idx': batch_idx,
                        'sample_idx': idx.item(),
                        'predicted': predictions[idx].cpu().numpy().tolist(),
                        'actual': labels[idx].cpu().numpy().tolist(),
                        'loss': loss.item()
                    })
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(val_loader)
        
        mistake_analysis = {
            'total_mistakes': len(mistakes),
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'mistakes': mistakes
        }
        
        logger.info(f"Found {len(mistakes)} mistakes out of {total_predictions} samples")
        logger.info(f"Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}")
        
        return mistake_analysis
    
    def step4_retrain_based_on_mistakes(self, train_loader: DataLoader, mistake_analysis: Dict[str, Any]):
        """
        Step 4: Retrain model based on the mistake
        """
        logger.info("Step 4: Retraining model based on mistakes...")
        
        # Focus training on mistake-prone samples
        # This is a simplified approach - you might want to implement more sophisticated methods
        
        self.model.train()
        epochs = 10  # Adjust based on your needs
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("Step 4 completed: Model retrained based on mistakes")
    
    def step5_retrain_on_same_dataset(self, train_loader: DataLoader):
        """
        Step 5: Retrain model on the same dataset again to get better results
        """
        logger.info("Step 5: Retraining model on same dataset for better results...")
        
        self.model.train()
        epochs = 5  # Additional training epochs
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Additional Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("Step 5 completed: Model retrained on same dataset")
    
    def _load_dataset(self, dataset_path: str, split: str) -> Tuple[List[str], List[List[int]]]:
        """
        Load dataset from CSV file for multi-label car damage analysis
        """
        # Load CSV file
        csv_path = os.path.join(dataset_path, split, "_classes.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded CSV with {len(df)} samples")
        
        image_paths = []
        labels = []
        
        # Process each row in the CSV
        for _, row in df.iterrows():
            filename = row['filename']
            image_path = os.path.join(dataset_path, split, filename)
            
            # Check if image file exists
            if os.path.exists(image_path):
                image_paths.append(image_path)
                
                # Create multi-label encoding: [car, dent, rust, scratch]
                # Each can be 0 or 1, allowing multiple labels per image
                multi_label = [
                    int(row['car']),      # car (clean)
                    int(row['dunt']),     # dent (note: 'dunt' in your CSV)
                    int(row['rust']),     # rust
                    int(row['scracth'])   # scratch (note: 'scracth' in your CSV)
                ]
                labels.append(multi_label)
            else:
                logger.warning(f"Image not found: {image_path}")
        
        logger.info(f"Loaded {len(image_paths)} images for {split} split")
        
        # Log label distribution
        if labels:
            labels_array = np.array(labels)
            logger.info(f"Label distribution:")
            logger.info(f"  Car (clean): {labels_array[:, 0].sum()}")
            logger.info(f"  Dent: {labels_array[:, 1].sum()}")
            logger.info(f"  Rust: {labels_array[:, 2].sum()}")
            logger.info(f"  Scratch: {labels_array[:, 3].sum()}")
        
        return image_paths, labels
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a saved model"""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        logger.info(f"Model loaded from {load_path}")

def main():
    """
    Main function implementing your 5-step guide
    """
    # Set your dataset path
    dataset_path = "dataset"  # Your dataset path
    
    # Set your pretrained model path (optional)
    pretrained_model_path = None  # No pretrained model for now
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        # Step 1: Convert images to tensors
        train_loader, val_loader = trainer.step1_convert_images_to_tensors(dataset_path)
        
        # Step 2: Use pretrained model
        trainer.step2_use_pretrained_model(pretrained_model_path)
        
        # Step 3: Check for mistakes
        mistake_analysis = trainer.step3_check_mistakes(val_loader)
        
        # Step 4: Retrain based on mistakes
        trainer.step4_retrain_based_on_mistakes(train_loader, mistake_analysis)
        
        # Step 5: Retrain on same dataset
        trainer.step5_retrain_on_same_dataset(train_loader)
        
        # Save the final model
        trainer.save_model("trained_car_damage_model.pth")
        
        logger.info("All 5 steps completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
