import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

class PersonDetector:
    def detect(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.model(img)
        
        person_count = 0
        for r in results:
            boxes = r.boxes
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    if self.model.names[cls] == 'person':
                        person_count += 1
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                        cv2.putText(img, 'person', (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        cv2.putText(img, f'Persons: {person_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        return img, person_count
