import torch
import torch.nn as nn
from typing import List, Union

#follow tutorial
    #SSD is trained on 300x300 images, so it needs to be modified
    #3rd pooling layer (features[16]): use ceiling instead of floor
    #5th pooling layer (features[30]): 2x2 kernel, stride=2 -> 3x3 kernel, stride=1.  results in 12x12 output (which is needed to subsample at dilation of 6 later)
    #last fc layer is dropped
    #remaining two fc layers are reworked to conv layers - TODO
#read SSD paper

def subsample(tensor: torch.Tensor, m: List[Union[int, None]]) -> torch.Tensor:
    assert tensor.dim() == len(m)

    for d in range(0, tensor.dim()):
        if m[d] is not None:
            indices = torch.arange(0, tensor.size(d), m[d])
            tensor = torch.index_select(tensor, d, indices) #needs to be list of integers

    return tensor


def init_model():
    model = torch.hub.load('pytorch/vision:v0.9.1', 'vgg16', pretrained=True)
    state_dict = model.state_dict()
    model.features[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    model.features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)

    model.avgpool = nn.Identity() #don't need avgpool anymore
    #index 0 and 3 have weights - need to save those and subsample to fill in new conv layers
    model.classifier = nn.Sequential(
        nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        nn.ReLU(inplace=True), 
        nn.Conv2d(1024, 1024, kernel_size=1)
    )
    return model

model = init_model()
print(model)
torch.save(model, 'model.pth')
