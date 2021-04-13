# A Simple Guide to Semantic Segmentation

**Semantic  Segmentation：** Given an input image, assign a label to every pixel (e.g., background, bottle, hand, sky, etc.).

## Datasets  

- **CamVid**: is a road/driving scene understanding database with 367 training images and 233 testing images of day and dusk scenes. The challenge is to segment 11 classes such as building, tree, sky, car, sign, road, pedestrian, fence, pole, sidewalk, and bicyclist etc. We usually resize images to 360x480 pixels for training and testing.

  <p align="center"><img width="100%" src="imgs/CamVid_demo.png" /></p>

- **Pascal VOC:** For pascal voc, first download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), after extracting the files we'll end up with `VOCtrainval_11-May-2012/VOCdevkit/VOC2012` containing, the image sets, the XML annotation for both object detection and segmentation, and JPEG images.
  The second step is to augment the dataset using the additionnal annotations provided by [Semantic Contours from Inverse Detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf). First download the image sets (`train_aug`, `trainval_aug`, `val_aug` and `test_aug`) from this link: [Aug ImageSets](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=), and add them the rest of the segmentation sets in `/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation`, and then download new annotations [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and add them to the path `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`, now we're set, for training use the path to `VOCtrainval_11-May-2012`
  
  <p align="center"><img width="100%" src="imgs/VOC_demo.png" /></p>
  
- **CityScapes:** First download the images and the annotations (there is two types of annotations, Fine `gtFine_trainvaltest.zip` and Coarse `gtCoarse.zip` annotations, and the images `leftImg8bit_trainvaltest.zip`) from the official website [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/), extract all of them in the same folder, and use the location of this folder in `config.json` for training.

  <p align="center"><img width="100%" src="imgs/CityScapes_demo.png" /></p>

- **ADE20K:** For ADE20K, simply download the images and their annotations for training and validation from [sceneparsing.csail.mit.edu](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), and for the rest visit the [website](http://sceneparsing.csail.mit.edu/).

  <p align="center"><img width="100%" src="imgs/ADE20K_demo.png" /></p>

- **COCO Stuff:** For COCO, there is two partitions, CocoStuff10k with only 10k that are used for training the evaluation, note that this dataset is outdated, can be used for small scale testing and training, and can be downloaded [here](https://github.com/nightrome/cocostuff10k). For the official dataset with all of the training 164k examples, it can be downloaded from the official [website](http://cocodataset.org/#download).
  Note that when using COCO dataset, 164k version is used per default, if 10k is prefered, this needs to be specified with an additionnal parameter `partition = 'CocoStuff164k'` in the config file with the corresponding path.
  
  <p align="center"><img width="100%" src="imgs/COCO_demo.png" /></p>



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

### Distribution-based Loss 

- **Cross entropy (CE)**, is derived from Kullback-Leibler (KL) divergence, which is a measure of dissimilarity between two distributions.  

  ![image-20201123141351701](https://i.loli.net/2020/11/23/UTsWauxCml1dt8V.png)

  where $g_{i}^{c}$ is binary indicator if class label c is the correct classification for pixel i, and $s^{c}_{i}$ is the corresponding predicted probability, $w_{c}$ is the weight for each class.  Usually, $w_{c}$ is inversely proportional to the class frequencies which can penalize majority classes.  

- **Focal Loss**, used to avoid  extreme foreground-background class imbalance,  where the confident predictions are scaled down.

  ![image-20201120185128292](https://i.loli.net/2020/11/23/3ipVDGacCPxls7A.png)

- **Distance map penalized cross entropy  (DPCE)** loss  weights cross entropy by distance maps which are derived from ground truth masks. It aims to guide the network’s focus towards hard-to segment boundary regions.  

  ![image-20201123150116276](https://i.loli.net/2020/11/23/17vRYDhiJ2EINxt.png)

  where $D$ is the distance penalty term, and $◦$ is the Hadamard product. Specifically, $D$ is generated by computing distance transform of ground truth and then reverting them.  

### Region-based Loss 

- **Dice-Loss**, which measures of overlap between two samples and can be more reflective of the training objective (maximizing the mIoU), but is highly non-convexe and can be hard to optimize. Dice-Loss does not require class re-weighting for imbalanced segmentation tasks.  

  ![image-20201123151945665](https://i.loli.net/2020/11/23/gPE2evIasfkN41S.png)

- **Generalized Dice Loss**, is the multi-class extension of Dice loss where the weight of each class is inversely proportional to the label frequencies.  

  ![image-20201123152603049](https://i.loli.net/2020/11/23/VFhjrqtsJXBLHZ9.png)

  where   ![image-20201123152652110](https://i.loli.net/2020/11/23/UzqobgY2k8T7WRn.png)

- **CE Dice Loss**, the sum of the Dice loss and CE, CE gives smooth optimization while Dice loss is a good indicator of the quality of the segmentation results.

  

- **IoU Loss**, is used to directly optimize the object category segmentation metric.  

  ![image-20201123152409593](https://i.loli.net/2020/11/23/8GCRJIuyxeZ1vrE.png)

- **Tversky Loss** , reshapes Dice loss and emphasizes false negatives, expect to achieve a better trade-off between precision and recall.

  ![image-20201123152456597](https://i.loli.net/2020/11/23/8dFqH1XT9ZYscno.png)

  where α and β are hyper-parameters which control the balance between false negatives and false positives.  

- **Focal Tversky Loss**, applies the concept of focal loss to focus on hard cases with low probabilities.  

  ![image-20201123153232115](https://i.loli.net/2020/11/23/BQ5ZsNmDRylzUdM.png)

  where γ varies in the range [1; 3].  

- **Lovasz Softmax** lends it self as a good alternative to the Dice loss, where we can directly optimization for the mean intersection-over-union based on the convex Lovász extension of submodular losses.

  

### Boundary-based Loss 

- **Boundary (BD) Loss**, To compute the distance $Dist(\partial G; \partial S)$ between two boundaries in a differentiable way, boundary loss uses integrals over the boundary instead of unbalanced
  integrals over regions to mitigate the difficulties of highly unbalanced segmentation.  

![image-20201123163408139](https://i.loli.net/2020/11/23/6U7SZnpLGluxPWJ.png)

​		where s(p) and g(p) are binary indicator function.  

- **Hausdorff Distance (HD) Loss**, Since minimizing HD directly is intractable and could lead to unstable training, However it can be approximated by the distance transforms of ground truth and predicted segmentation. Further more, the network can be trained with following HD loss function to reduce HD:  

![image-20201123161930869](https://i.loli.net/2020/11/23/ws1fpgPUXHIAzVk.png)

​		 where $d_{G}$ and $d_{S}$ are distance transforms of ground truth and segmentation.  

### Compound Loss 

- **Combo Loss**, is the weighted sum between weighted CE and Dice loss.  

  ![image-20201123160345032](https://i.loli.net/2020/11/23/Miw2ISldJzjch53.png)

- **Exponential Logarithmic Loss (ELL)** , 

![image-20201123161031950](https://i.loli.net/2020/11/23/8cNnu5qSWxdk7F2.png)

​		where ![image-20201123161056438](https://i.loli.net/2020/11/23/DAsn5lRmWE1M2pS.png)

## Evaluation metrics

- **Pixel Accuracy (Acc)**: it is the simplest metric, simply computing a ratio between the amount of properly classified pixels and the total number of them.  

![](https://i.loli.net/2020/11/20/V7rzZn5gpDJF6yW.png)

- **Mean Pixel Accuracy (mAcc)**: a slightly improved PA in which the ratio of correct pixels is computed
  in a per-class basis and then averaged over the total number of classes.  

![mAcc](https://i.loli.net/2020/11/20/FEux7jgikOQCRGV.png)

- **Mean Intersection over Union (mIoU)**: this is the standard metric for segmentation purposes. It computes a ratio between the intersection and the union of two sets, in our case the ground truth and our predicted segmentation. That ratio can be reformulated as the number of true positives (intersection) over the sum of true positives, false negatives, and  false positives (union). That IoU is computed on a per-class basis and then averaged.  

![](https://i.loli.net/2020/11/20/tHyiNDkawcK3FxG.png)

- **Frequency Weighted Intersection over Union (FWIoU)**: it is an improved over the raw MIoU which
  weights each class importance depending on their appearance frequency.  

![FWIoU](https://i.loli.net/2020/11/20/Wa89rqezxm3PMg6.png)

## REFERENCES

[1] A survey of loss functions for semantic segmentation, 2020, https://arxiv.org/pdf/2006.14822v1.pdf

[2] A Review on Deep Learning Techniques Applied to Semantic Segmentation, 2017,  https://arxiv.org/pdf/1704.06857.pdf

[3] Segmentation Loss Odyssey, 2020, https://arxiv.org/abs/2005.13449