from label_studio_ml.model import LabelStudioMLBase
import requests, os
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

LS_URL = os.environ['LABEL_STUDIO_BASEURL']
LS_API_TOKEN = os.environ['LABEL_STUDIO_API_TOKEN']

class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = [    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',     'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',     'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',     'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',     'toothbrush'] 
        self.model = YOLO("yolov8n.pt")

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns 
            the list of predictions based on input list of tasks 
        """
        task = tasks[0]

        predictions = []
        score = 0

        header = {
            "Authorization": "Token " + LS_API_TOKEN}
        image = Image.open(BytesIO(requests.get(
            LS_URL + task['data']['image'], headers=header).content))
        original_width, original_height = image.size
        results = self.model.predict(image)

        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append({
                    "id": str(i),
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "score": prediction.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": xyxy[0] / original_width * 100, 
                        "y": xyxy[1] / original_height * 100,
                        "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                        "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                        "rectanglelabels": [self.labels[int(prediction.cls.item())]]
                    }
                })
                score += prediction.conf.item()
            
        return [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8n",  # all predictions will be differentiated by model version
        }]
