import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import shutil

class CustomPersonDetector:
    def __init__(self):
        self.model = None
        
    def prepare_dataset(self, image_dir, label_dir, output_dir):
        """
        Prepare dataset for training
        Expected format: YOLO format (class x_center y_center width height)
        Class 0 = person
        """
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        
        for dir_path in [train_dir, val_dir]:
            os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)
        
        images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        np.random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        for img in train_images:
            shutil.copy(os.path.join(image_dir, img), 
                       os.path.join(train_dir, 'images', img))
            
            label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(os.path.join(label_dir, label_file)):
                shutil.copy(os.path.join(label_dir, label_file),
                           os.path.join(train_dir, 'labels', label_file))
        
        for img in val_images:
            shutil.copy(os.path.join(image_dir, img),
                       os.path.join(val_dir, 'images', img))
            
            label_file = img.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(os.path.join(label_dir, label_file)):
                shutil.copy(os.path.join(label_dir, label_file),
                           os.path.join(val_dir, 'labels', label_file))
        
        yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images

nc: 1
names: ['person']
"""
        with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)
        
        return os.path.join(output_dir, 'dataset.yaml')
    
    def train(self, dataset_yaml, epochs=50, imgsz=640, batch_size=16):
        """
        Train custom person detection model
        """
        # Load pre-trained model
        self.model = YOLO('yolov8n.pt')
        
        # Train the model
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            workers=8,
            name='person_detection'
        )
        
        # Save best model
        self.best_model_path = 'runs/detect/person_detection/weights/best.pt'
        print(f"Model saved at: {self.best_model_path}")
        
        return results
    
    def evaluate(self, test_image_path):
        """Test the trained model on new images"""
        if self.model is None:
            self.model = YOLO(self.best_model_path)
        
        results = self.model(test_image_path, conf=0.5)
        return results

class ManualLabeler:
    def __init__(self):
        self.points = []
        
    def draw_rectangle(self, event, x, y, flags, param):
        """Mouse callback function for drawing rectangles"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.points.append((x, y))
            cv2.rectangle(param['image'], self.points[0], self.points[1], (0, 255, 0), 2)
            cv2.imshow('Labeling', param['image'])
    
    def label_image(self, image_path):
        """
        Manually label persons in an image
        Click and drag to draw bounding boxes
        Press 's' to save, 'n' for next, 'q' to quit
        """
        image = cv2.imread(image_path)
        original = image.copy()
        
        bboxes = []
        current_bbox = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_bbox
            if event == cv2.EVENT_LBUTTONDOWN:
                current_bbox = [x, y, x, y]
            elif event == cv2.EVENT_LBUTTONUP:
                current_bbox[2] = x
                current_bbox[3] = y
                bboxes.append(current_bbox)
                cv2.rectangle(image, (current_bbox[0], current_bbox[1]), 
                            (current_bbox[2], current_bbox[3]), (0, 255, 0), 2)
                cv2.imshow('Labeling', image)
                current_bbox = None
            elif event == cv2.EVENT_MOUSEMOVE and current_bbox is not None:
                temp_image = original.copy()
                cv2.rectangle(temp_image, (current_bbox[0], current_bbox[1]), (x, y), (0, 255, 0), 2)
                cv2.imshow('Labeling', temp_image)
        
        cv2.imshow('Labeling', image)
        cv2.setMouseCallback('Labeling', mouse_callback)
        
        print("Instructions:")
        print("- Click and drag to draw bounding boxes around persons")
        print("- Press 's' to save labels")
        print("- Press 'r' to reset")
        print("- Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                label_file = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
                with open(label_file, 'w') as f:
                    h, w = original.shape[:2]
                    for bbox in bboxes:
                        x_center = (bbox[0] + bbox[2]) / (2 * w)
                        y_center = (bbox[1] + bbox[3]) / (2 * h)
                        width = (bbox[2] - bbox[0]) / w
                        height = (bbox[3] - bbox[1]) / h
                        f.write(f"0 {x_center} {y_center} {width} {height}\n")
                print(f"Labels saved to {label_file}")
                break
            elif key == ord('r'):
                # Reset
                image = original.copy()
                bboxes = []
                cv2.imshow('Labeling', image)
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return bboxes
