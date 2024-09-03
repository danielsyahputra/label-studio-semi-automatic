import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".env"],
    pythonpath=True,
    dotenv=True,
)

import os

import cv2
import hydra
from hydra import compose, initialize
from label_studio_ml.model import LabelStudioMLBase  # type: ignore
from PIL import Image, ImageOps

from src.engine.ocr.ocr_recognition import OCREngineRecognizer

LS_URL = os.environ["LABEL_STUDIO_BASEURL"]
LS_API_TOKEN = os.environ["LABEL_STUDIO_API_TOKEN"]

import warnings

warnings.filterwarnings("ignore")


class OCRModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(OCRModel, self).__init__(**kwargs)

        self.init_model()

    def init_model(self):
        self.MODEL_DIR = os.environ["MODEL_DIR"]
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # type: ignore
        initialize(version_base=None, config_path=f"../configs")
        cfg = compose(config_name="main")
        self.ocr_recognizer = OCREngineRecognizer(cfg=cfg)

    def load_image(self, img_path_url, task_id):
        cache_dir = os.path.join(self.MODEL_DIR, ".file-cache")
        os.makedirs(cache_dir, exist_ok=True)
        filepath = self.get_local_path(
            img_path_url,
            cache_dir=cache_dir,
            ls_access_token=LS_API_TOKEN,
            ls_host=LS_URL,
            task_id=task_id,
        )
        # image = Image.open(filepath)
        # image = ImageOps.exif_transpose(image)
        image = cv2.imread(filepath)
        return image

    def predict(self, tasks, **kwargs):
        """This is where inference happens: model returns
        the list of predictions based on input list of tasks
        """
        from_name, to_name, value = self.label_interface.get_first_tag_occurence(
            "TextArea", "Image"
        )
        task = tasks[0]

        img_path_url = task["data"][value]

        context = kwargs.get("context")
        if context:
            if not context["result"]:
                return []

            image = self.load_image(img_path_url, task.get("id"))

            cv2.imwrite("tmp/image.jpg", image)

            result = context.get("result")[-1]
            meta = self._extract_meta({**task, **result})
            x = meta["x"] * meta["original_width"] / 100
            y = meta["y"] * meta["original_height"] / 100
            w = meta["width"] * meta["original_width"] / 100
            h = meta["height"] * meta["original_height"] / 100

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            image = image[y : y + h, x : x + w]
            texts, confs = self.ocr_recognizer([image])

            result_text = texts[0].strip()

            meta["text"] = result_text
            temp = {
                "original_width": meta["original_width"],
                "original_height": meta["original_height"],
                "image_rotation": 0,
                "value": {
                    "x": x / meta["original_width"] * 100,
                    "y": y / meta["original_height"] * 100,
                    "width": w / meta["original_width"] * 100,
                    "height": h / meta["original_height"] * 100,
                    "rotation": 0,
                    "text": [meta["text"]],
                },
                "id": meta["id"],
                "from_name": from_name,
                "to_name": meta["to_name"],
                "type": "textarea",
                "origin": "manual",
            }

            return [
                {
                    "result": [temp, result],
                    "score": 0,
                    "model_version": "ocr-v1",
                }
            ]
        else:
            return []

    @staticmethod
    def _extract_meta(task):
        meta = dict()
        if task:
            meta["id"] = task["id"]
            meta["from_name"] = task["from_name"]
            meta["to_name"] = task["to_name"]
            meta["type"] = task["type"]
            meta["x"] = task["value"]["x"]
            meta["y"] = task["value"]["y"]
            meta["width"] = task["value"]["width"]
            meta["height"] = task["value"]["height"]
            meta["original_width"] = task["original_width"]
            meta["original_height"] = task["original_height"]
        return meta
