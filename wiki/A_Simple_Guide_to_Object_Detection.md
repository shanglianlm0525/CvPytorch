# A Simple Guide to Object Detection

**Object Detection**: Given an input image, identify and locate which objects are present (using rectangular coordinates). 

## Datasets  

- **Pascal VOC:** For pascal voc, first download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), after extracting the files we'll end up with `VOCtrainval_11-May-2012/VOCdevkit/VOC2012` containing, the image sets, the XML annotation for both object detection and segmentation, and JPEG images.
  The second step is to augment the dataset using the additionnal annotations provided by [Semantic Contours from Inverse Detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf). First download the image sets (`train_aug`, `trainval_aug`, `val_aug` and `test_aug`) from this link: [Aug ImageSets](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=), and add them the rest of the segmentation sets in `/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation`, and then download new annotations [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and add them to the path `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`, now we're set, for training use the path to `VOCtrainval_11-May-2012`
  
- **COCO:** For COCO, there is two partitions, CocoStuff10k with only 10k that are used for training the evaluation, note that this dataset is outdated, can be used for small scale testing and training, and can be downloaded [here](https://github.com/nightrome/cocostuff10k). For the official dataset with all of the training 164k examples, it can be downloaded from the official [website](http://cocodataset.org/#download).
  Note that when using COCO dataset, 164k version is used per default, if 10k is prefered, this needs to be specified with an additionnal parameter `partition = 'CocoStuff164k'` in the config file with the corresponding path.
- **VisDrone:** containing 10,209 static images (6,471 for training, 548 for validation and 3,190 for testing) captured by drone platforms in different places at different height. There are ten object categories of interest including pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, and tricycle are annotationed . Some rarely occurring special vehicles (*e.g.*, *machineshop truck*, *forklift truck*, and *tanker*) are ignored in evaluation. Furthermore a target is skipped during evaluation if its truncation ratio is larger than 50%. 

## Network Architectures  

- [x] (**Faster-RCNN**) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015) [[Paper\]](https://arxiv.org/pdf/1506.01497.pdf)

- [ ] (**SSD**) SSD: Single Shot MultiBox Detector (2015) [[Paper\]](https://arxiv.org/pdf/1512.02325.pdf)

- [x] (**RetinaNet**) Focal Loss for Dense Object Detection (2017) [[Paper\]](https://arxiv.org/pdf/1708.02002.pdf)

- [x] (**FCOS**) Fully Convolutional One-Stage Object Detection (2020) [[Paper\]](https://arxiv.org/pdf/1904.01355.pdf)

- [x] (**NanoDet**) https://github.com/RangiLyu/nanodet (2021) 

- [ ] (**YOLOv3**) YOLOv3: An Incremental Improvement (2018) [[Paper\]](https://arxiv.org/pdf/1804.02767.pdf)

- [ ] (**YOLOv5**) https://github.com/ultralytics/yolov5 (2020) 


## Loss functions



## Evaluation metrics

- AP: is to calculate the area under the curve (AUC) of the Precision x Recall curve. \ 
In practice AP is the precision averaged across all recall values between 0 and 1. \
Currently, the interpolation performed by PASCAL VOC challenge uses all data points, \
rather than interpolating only 11 equally spaced points. 

11-point interpolation

The 11-point interpolation tries to summarize the shape of the Precision x Recall curve \ 
by averaging the precision at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ... , 1]:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BAP%7D%3D%5Cfrac%7B1%7D%7B11%7D%20%5Csum_%7Br%5Cin%20%5Cleft%20%5C%7B%200%2C%200.1%2C%20...%2C1%20%5Cright%20%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28%20r%20%5Cright%20%29%7D">
</p>

<!---
\text{AP}=\frac{1}{11} \sum_{r\in \left \{ 0, 0.1, ...,1 \right \}}\rho_{\text{interp}\left ( r \right )}
--->

with

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%7D%20%3D%20%5Cmax_%7B%5Ctilde%7Br%7D%3A%5Ctilde%7Br%7D%20%5Cgeq%20r%7D%20%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29">
</p>

<!--- 
\rho_{\text{interp}} = \max_{\tilde{r}:\tilde{r} \geq r} \rho\left ( \tilde{r} \right )
--->

where \rho_{\text{interp}\left ( r \right ) is the measured precision at recall .

Instead of using the precision observed at each point, the AP is obtained by interpolating the precision only at the 11 levels !  taking the **maximum precision whose recall value is greater than ![](http://latex.codecogs.com/gif.latex?r)**.

Interpolating all points

Instead of interpolating only in the 11 equally spaced points, you could interpolate through all points <img src="https://latex.codecogs.com/gif.latex?n"> in such way that:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Bn%3D0%7D%20%5Cleft%20%28%20r_%7Bn&plus;1%7D%20-%20r_%7Bn%7D%20%5Cright%20%29%20%5Crho_%7B%5Ctext%7Binterp%7D%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29">
</p>

<!---
\sum_{n=0} \left ( r_{n+1} - r_{n} \right ) \rho_{\text{interp}}\left ( r_{n+1} \right )
--->

with

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29%20%3D%20%5Cmax_%7B%5Ctilde%7Br%7D%3A%5Ctilde%7Br%7D%20%5Cge%20r_%7Bn&plus;1%7D%7D%20%5Crho%20%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29">
</p>


<!---
\rho_{\text{interp}}\left ( r_{n+1} \right ) = \max_{\tilde{r}:\tilde{r} \ge r_{n+1}} \rho \left ( \tilde{r} \right )
--->


where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

In this case, instead of using the precision observed at only few points, the AP is now obtained by interpolating the precision at **each level**, ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater or equal than ![](http://latex.codecogs.com/gif.latex?r&plus;1)**. This way we calculate the estimated area under the curve.

- mAP: is a metric used to measure the accuracy of object detectors over all classes in a specific database. The mAP is simply the average AP over all classes, that is

## REFERENCES

[1] A Survey on Performance Metrics for Object-Detection Algorithms, 2020

[2] Detection Evaluation, 2020,  https://cocodataset.org/#detection-eval
