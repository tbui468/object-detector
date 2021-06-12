import torch
import torch.nn as nn

#SSD is trained on 300x300 images, so it needs to be modified
#3rd pooling layer (features[16]): use ceiling instead of floor
#5th pooling layer (features[30]): 2x2 kernel, stride=2 -> 3x3 kernel, stride=1.  results in 12x12 output (which is needed to subsample at dilation of 6 later)
#last fc layer is dropped
#remaining two fc layers are reworked to conv layers
def init_model():
    model = torch.hub.load('pytorch/vision:v0.9.1', 'vgg16', pretrained=True)
    model.features[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    model.features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, dilation=1, ceil_mode=False)

    model.classifier = model.classifier[0:4]
    return model

model = init_model()
print(model)
torch.save(model, 'model.pth')
