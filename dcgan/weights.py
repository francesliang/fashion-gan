
import torch.nn as nn


def weights_init(m):
    """
    The function takes an initialized model as input and reinitializes all convolutional, convolutional-transpose, and batch normalization layers to meet the criteria that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02 (from DCGAN paper).
    This function is applied to the models immediately after initialization.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

