import re
import torch
from .simple_models import Size
from torchvision.models.densenet import DenseNet as TVDenseNet
from torchvision.models.densenet import model_urls
from torchvision.models.utils import load_state_dict_from_url
from typing import Optional


class DenseNet(TVDenseNet):

    input_size = (3, 224, 224)

    def __init__(
        self,
        arch: str,
        input_size: Size,
        output_size: Size,
        pretrained: bool,
        ckpt_dir: Optional[str] = None,
    ):
        assert tuple(input_size) == self.input_size

        if arch == "densenet121":
            growth_rate = 32
            block_config = (6, 12, 24, 16)
            num_init_features = 64
        else:
            raise ValueError(f"{arch} is not a supported type of ResNet!")

        super().__init__(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
        )

        if pretrained:
            _load_state_dict(self, model_urls[arch], model_dir=ckpt_dir, progress=True)

    @property
    def linear_output(self):
        return self.classifier


def _load_state_dict(
    model: torch.nn.Module, model_url: str, model_dir: str, progress: bool
) -> None:
    # based on https://github.com/pytorch/vision/blob/7b87af25ba03c2cd2579a66b6c2945624b25a277/torchvision/models/densenet.py#L224
    # That version doesn't allow changing model_dir

    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(
        model_url, model_dir=model_dir, progress=progress
    )
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
