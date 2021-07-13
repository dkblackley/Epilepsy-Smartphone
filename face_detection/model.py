import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.datasets as datasets

num_classes = 2  # background and face

data_set = datasets.WIDERFace(split='train', download=True)

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
