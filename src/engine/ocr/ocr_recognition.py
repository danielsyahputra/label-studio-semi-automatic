import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".env", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import math
from typing import List, Tuple, Union

import cv2
import numpy as np
from omegaconf import DictConfig

from src.engine.ocr import ocr_utils
from src.engine.ocr.ocr_base import OCREngineBase


class OCREngineRecognizer(OCREngineBase):
    def __init__(self, cfg: DictConfig, model_type: str = "rec") -> None:
        super().__init__(cfg, model_type)
        self.model_type = model_type
        self.input_shape = [3, 48, 320]
        self.ctc_decoder = ocr_utils.CTCDecoder()

    def resize(self, image, max_wh_ratio):
        input_h, input_w = self.input_shape[1], self.input_shape[2]

        assert self.input_shape[0] == image.shape[2]
        input_w = int((input_h * max_wh_ratio))
        w = self.inputs.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            input_w = w
        h, w = image.shape[:2]
        ratio = w / float(h)
        if math.ceil(input_h * ratio) > input_w:
            resized_w = input_w
        else:
            resized_w = int(math.ceil(input_h * ratio))

        resized_image = cv2.resize(image, (resized_w, input_h))
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padded_image = np.zeros(
            (self.input_shape[0], input_h, input_w), dtype=np.float32
        )
        padded_image[:, :, 0:resized_w] = resized_image
        return padded_image

    def __call__(self, images: Union[np.ndarray, List]) -> Tuple:
        batch_size = 6
        num_images = len(images)

        results = [["", 0.0]] * num_images
        confidences = [["", 0.0]] * num_images
        indices = np.argsort(np.array([x.shape[1] / x.shape[0] for x in images]))

        for index in range(0, num_images, batch_size):
            input_h, input_w = self.input_shape[1], self.input_shape[2]
            max_wh_ratio = input_w / input_h
            norm_images = []
            for i in range(index, min(num_images, index + batch_size)):
                h, w = images[indices[i]].shape[0:2]
                max_wh_ratio = max(max_wh_ratio, w * 1.0 / h)
            for i in range(index, min(num_images, index + batch_size)):
                norm_image = self.resize(images[indices[i]], max_wh_ratio)
                norm_image = norm_image[np.newaxis, :]
                norm_images.append(norm_image)
            norm_images = np.concatenate(norm_images)

            outputs = self.session.run(None, {self.inputs.name: norm_images})
            result, confidence = self.ctc_decoder(outputs[0])
            for i in range(len(result)):
                results[indices[index + i]] = result[i]
                confidences[indices[index + i]] = confidence[i]
        return results, confidences


if __name__ == "__main__":
    """Debugging."""

    from pprint import pprint

    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        image = cv2.imread("tmp/images.png")
        ocr_engine_recognizer = OCREngineRecognizer(cfg=cfg)
        result = ocr_engine_recognizer([image])
        print(result)

    main()
