from .carts import *
from .vision import *
from .evaluation import *
from .loss import *

model_dict = {
                "CaRTS": CaRTS,
                "Unet": Unet,
                "UnetPlusPlus": UnetPlusPlus,
                "Mask2Former": Mask2Former,
            }

def build_model(model, device):
    return model_dict[model['name']](model['params'], device)
