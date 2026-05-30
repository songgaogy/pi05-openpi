import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ARX_ACTION_DIM = 14


def make_arx_example() -> dict:
    """Creates a random input example for the ARX A5 policy."""
    return {
        "observation/state": np.random.rand(ARX_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "put_shrimp_in_pot",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Maps ARX A5 LeRobot samples to the model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported ARX model type: {self.model_type}")

        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Maps model output actions back to the ARX A5 14D action space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ARX_ACTION_DIM])}
