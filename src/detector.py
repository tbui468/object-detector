import torch
import torch.nn as nn
from typing import List, Union

#these are the steps the authors of the Single Shot Detector (SSD) followed:
    #base model is VGG16 pretrained on ImageNet2012
    #SSD is trained on 300x300 images, and the following steps allow for this
    #3rd pooling layer (features[16]): use ceiling instead of floor
    #5th pooling layer (features[30]): 2x2 kernel, stride=2 -> 3x3 kernel, stride=1.  results in 12x12 output (which is needed to subsample at dilation of 6 later)
    #last fc layer is dropped
    #remaining two fc layers are reworked to conv layers, and the weights and biases are subsampled from the pretrained model

#first parameter is tensor to subsample
#second parameter is list (of equal length to input tensor dim) of subsample stride for each dim(1 or None for no subsampling, 2 for sampling every other parameter, etc)
def subsample(tensor: torch.Tensor, m: List[Union[int, None]]) -> torch.Tensor:
    assert tensor.dim() == len(m)

    for d in range(0, tensor.dim()):
        if m[d] is not None:
            indices = torch.arange(0, tensor.size(d), m[d])
            tensor = torch.index_select(tensor, d, indices) #needs to be list of integers

    return tensor

def load_weights(model: nn.Module) -> nn.Module:
    return model


def init_model():
    model = torch.hub.load('pytorch/vision:v0.9.1', 'vgg16', pretrained=True)
    original_state_dict = model.state_dict() #get original weights and biases

    #modify model
    model.features[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    model.features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)
    model.avgpool = nn.Identity() #don't need avgpool anymore
    model.classifier = nn.Sequential(
        nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        nn.ReLU(inplace=True), 
        nn.Conv2d(1024, 1024, kernel_size=1),
    )

    new_state_dict = model.state_dict() #state dict of modified model - the weights and biases of modified layers are random right now
    new_params = list(new_state_dict.keys())
    for i, param in enumerate(new_params[:-4]):
        new_state_dict[param] = original_state_dict[param]

    #load all weights into modified model except last 4 (weights and biases for last two conv layers)
    conv1_weight = original_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
    conv1_bias = original_state_dict['classifier.0.bias']
    conv2_weight = original_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
    conv2_bias = original_state_dict['classifier.3.bias']

    new_state_dict['classifier.0.weight'] = subsample(conv1_weight, [4, None, 3, 3])
    new_state_dict['classifier.0.bias'] = subsample(conv1_bias, [4])
    new_state_dict['classifier.2.weight'] = subsample(conv2_weight, [4, 4, 3, 3])
    new_state_dict['classifier.2.bias'] = subsample(conv2_bias, [4])

    #reshape and subsample parameters in last two conv layers, and load them up
    model.load_state_dict(new_state_dict)
    return model

model = init_model()
print(model)
#print(model)
#torch.save(model, 'model.pth')
