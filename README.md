
# AI With Smartphone Video in Epilepsy Diagnosis

This project is the official repository hosting the code for the results behind the paper I wrote in collaboration 
with the University of Dundee. The paper was titled AI With Smartphone Video in Epilepsy Diagnosis and is for
internal use only

### Requirements

To install libraries:

```setup
pip install -r requirements.txt
```

Sadly all training data is not made publicly available and is only available internally to the University of Dundee. 
We will also not be giving out any information regarding the contents of the dataset
# Project Abstract

A paper showcasing our results on the 15 videos supplied to us from the University of Glasgow. We detail our methods and results, finding body segmentation gets an average accuracy of 53.57\%, face detection achieves 47\% and no segmentation achieves 48.6\%. We conclude that more data is required to train and test a model.
## Segmentation Methods Used

### MTCNN
For face detection we use the Pretrained MTCNN model as supplied by https://github.com/timesler/facenet-pytorch
The paper for MTCNN can be found at: https://arxiv.org/abs/1604.02878
### KeyPoint RCNN
We use a Keypoint RCNN (which is a variant of Mask R-CNN) that has been pretrained on the COCO dataset. 
The model we used is part of the Pytorch torchvision.models package and can be accessed after downloading torch.
See relevant docs here: https://pytorch.org/vision/stable/models.html
The paper for the Mask R-CNN can be found at: https://arxiv.org/pdf/1703.06870v3.pdf
### SORT
Once the bounding boxes have been identified we track them throughout the video by using the Simple Online Real
 time Tracking algorithm (SORT) as implemented in https://github.com/abewley/sort
The paper for SORT can be found at: https://arxiv.org/abs/1602.00763

### Our Implementation
Due to the very few samples in our data-set, we use a very shallow network, consisting of a single 64 
output convolutional layer followed by a 64 output LSTM layer and a final single output classification layer
We use the BCELossWithLogits which combines a sigmoid activation function with cross entropy loss.


## Usage

To change the FPS of the video dataset to 10 FPS, run the shell command change_fps.sh. This command expects all videos
to be in a folder called datasets/

There are several optional parameters that can be passed to the network, to learn more about what purpose each
one serves, call python main.py --help, an example call is shown below:
```train
python3 main.py --LOSO False --epochs 10 --dataset_dir datasets/ --model_dir models/ --segment face
```

All code is availible at https://github.com/dkblackley/Epilepsy-Smartphone/

