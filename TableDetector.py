from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch

class TableDetector:
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
        self.model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")

        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        return results