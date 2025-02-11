from torchvision import models

model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
for name, layer in model.named_children():
    print(f'Layer name: {name}, Layer: {layer}')