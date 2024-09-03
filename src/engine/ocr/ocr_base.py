import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", ".env", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

from abc import ABC
from typing import List, Tuple, Union

import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig


class OCREngineBase(ABC):
    def __init__(self, cfg: DictConfig, model_type: str = "det") -> None:
        super().__init__()
        self.cfg = cfg.engine.ocr
        self.model_type = model_type
        self.setup()

    def setup(self) -> None:
        device = self.cfg.device
        providers = ["CPUExecutionProvider"]
        if device != "cpu":
            providers.append("CUDAExecutionProvider")

        model_path = (
            self.cfg.det_model_path
            if self.model_type == "det"
            else self.cfg.rec_model_path
        )
        self.session = ort.InferenceSession(f"{ROOT}/{model_path}", providers=providers)
        self.inputs = self.session.get_inputs()[0]

    def __call__(self, x: Union[np.ndarray, List]) -> Union[np.ndarray, Tuple]:
        raise NotImplementedError


if __name__ == "__main__":
    """Debugging."""

    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        ocr_engine = OCREngineBase(cfg=cfg)

    main()
