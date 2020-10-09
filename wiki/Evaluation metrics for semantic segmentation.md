# Evaluation metrics and loss functions for semantic segmentation



## Datasets  



- **CamVid**: is a road/driving scene understanding database which was originally captured as five video sequences with a 960 × 720 resolution camera mounted on the dashboard of a car. Those sequences were sampled (four of them at 1 fps and one at 15 fps) adding up to 701 frames. Those stills were manually annotated with 32 classes: void, building, wall, tree, vegetation, fence, sidewalk, parking block, column/pole, traffic cone, bridge, sign, miscellaneous text, traffic light, sky, tunnel, archway, road, road shoulder, lane markings (driving), lane markings (non-driving), animal, pedestrian, child, cart luggage, bicyclist, motorcycle, car, SUV/pickup/truck, truck/bus, train, and other moving object. It is important to remark the partition introduced by Sturgess et al. which divided the dataset into 367=100=233 training, validation, and testing images respectively. That partition makes use of a subset of class labels: building, tree, sky, car, sign, road, pedestrian, fence, pole, sidewalk, and bicyclist  

- **Pascal VOC:** For pascal voc, first download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), after extracting the files we'll end up with `VOCtrainval_11-May-2012/VOCdevkit/VOC2012` containing, the image sets, the XML annotation for both object detection and segmentation, and JPEG images.
  The second step is to augment the dataset using the additionnal annotations provided by [Semantic Contours from Inverse Detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf). First download the image sets (`train_aug`, `trainval_aug`, `val_aug` and `test_aug`) from this link: [Aug ImageSets](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=), and add them the rest of the segmentation sets in `/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation`, and then download new annotations [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and add them to the path `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`, now we're set, for training use the path to `VOCtrainval_11-May-2012`
- **CityScapes:** First download the images and the annotations (there is two types of annotations, Fine `gtFine_trainvaltest.zip` and Coarse `gtCoarse.zip` annotations, and the images `leftImg8bit_trainvaltest.zip`) from the official website [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/), extract all of them in the same folder, and use the location of this folder in `config.json` for training.
- **ADE20K:** For ADE20K, simply download the images and their annotations for training and validation from [sceneparsing.csail.mit.edu](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), and for the rest visit the [website](http://sceneparsing.csail.mit.edu/).
- **COCO Stuff:** For COCO, there is two partitions, CocoStuff10k with only 10k that are used for training the evaluation, note that this dataset is outdated, can be used for small scale testing and training, and can be downloaded [here](https://github.com/nightrome/cocostuff10k). For the official dataset with all of the training 164k examples, it can be downloaded from the official [website](http://cocodataset.org/#download).
  Note that when using COCO dataset, 164k version is used per default, if 10k is prefered, this needs to be specified with an additionnal parameter `partition = 'CocoStuff164k'` in the config file with the corresponding path.



## Network Architectures  

- (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper\]](https://arxiv.org/abs/1802.02611)

- (**GCN**) Large Kernel Matter, Improve Semantic Segmentation by Global Convolutional Network [[Paper\]](https://arxiv.org/abs/1703.02719)

- (**UperNet**) Unified Perceptual Parsing for Scene Understanding [[Paper\]](https://arxiv.org/abs/1807.10221)

- (**DUC, HDC**) Understanding Convolution for Semantic Segmentation [[Paper\]](https://arxiv.org/abs/1702.08502)

- (**PSPNet**) Pyramid Scene Parsing Network [[Paper\]](http://jiaya.me/papers/PSPNet_cvpr17.pdf)

- (**ENet**) A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[Paper\]](https://arxiv.org/abs/1606.02147)

- (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015): [[Paper\]](https://arxiv.org/abs/1505.04597)

- (**SegNet**) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016): [[Paper\]](https://arxiv.org/pdf/1511.00561)

- (**FCN**) Fully Convolutional Networks for Semantic Segmentation (2015): [[Paper\]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

  

## Loss functions

- **Dice-Loss**, which measures of overlap between two samples and can be more reflective of the training objective (maximizing the mIoU), but is highly non-convexe and can be hard to optimize.
- **CE Dice loss**, the sum of the Dice loss and CE, CE gives smooth optimization while Dice loss is a good indicator of the quality of the segmentation results.
- **Focal Loss**, an alternative version of the CE, used to avoid class imbalance where the confident predictions are scaled down.
- **Lovasz Softmax** lends it self as a good alternative to the Dice loss, where we can directly optimization for the mean intersection-over-union based on the convex Lovász extension of submodular losses (for more details, check the paper: [The Lovász-Softmax loss](https://arxiv.org/abs/1705.08790)).



## Evaluation metrics

**Pixel Accuracy (Acc)**: it is the simplest metric, simply computing a ratio between the amount of properly classified pixels and the total number of them.  
$$
Acc = \frac{\sum_{i=0}^{k}{p_{ii}}}{\sum_{i=0}^{k}{\sum_{j=0}^{k}{p_{ij}}}}
$$


**Mean Pixel Accuracy (mAcc)**: a slightly improved PA in which the ratio of correct pixels is computed
in a per-class basis and then averaged over the total number of classes.  
$$
mAcc = \frac{1}{k+1}\sum_{i=0}^{k}{\frac{{p_{ii}}}{\sum_{j=0}^{k}{p_{ij}}}}
$$


**Mean Intersection over Union (mIoU)**: this is the standard metric for segmentation purposes. It computes a ratio between the intersection and the union of two sets, in our case the ground truth and our predicted segmentation. That ratio can be reformulated as the number of true positives (intersection) over the sum of true positives, false negatives, and  false positives (union). That IoU is computed on a per-class basis and then averaged.  
$$
mAcc = \frac{1}{k+1}\sum_{i=0}^{k}{\frac{{p_{ii}}}{\sum_{j=0}^{k}{p_{ij}}+\sum_{j=0}^{k}p_{ji}-p_{ii}}}
$$


**Frequency Weighted Intersection over Union (FWIoU)**: it is an improved over the raw MIoU which
weights each class importance depending on their appearance frequency.  


$$
FWIoU = \frac{1}{\sum_{i=0}^{k}\sum_{j=0}^{k}{p_{ij}}}\sum_{i=0}^{k}{\frac{\sum_{j=0}^{k}{p_{ij}p_{ii}}}{\sum_{j=0}^{k}{p_{ij}}+\sum_{j=0}^{k}p_{ji}-p_{ii}}}
$$








## REFERENCES

[1] A survey of loss functions for semantic segmentation, 2020, https://arxiv.org/pdf/2006.14822v1.pdf

[2] A Review on Deep Learning Techniques Applied to Semantic Segmentation, 2017,  https://arxiv.org/pdf/1704.06857.pdf