
from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .mask2former import Mask2Former

model_dict = {
    "Unet":Unet,
    "UnetPlusPlus": UnetPlusPlus,
    "Mask2Former": Mask2Former,
}


def build_vision_module(vision, device):
    return model_dict[vision['name']](vision['params'], device)
