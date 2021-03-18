# A Simple Guide to Deep Learning Object Detection

**Object Detection**: Given an input image, identify and locate which objects are present (using rectangular coordinates). 

## Datasets  

- **Pascal VOC:** For pascal voc, first download the [original dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), after extracting the files we'll end up with `VOCtrainval_11-May-2012/VOCdevkit/VOC2012` containing, the image sets, the XML annotation for both object detection and segmentation, and JPEG images.
  The second step is to augment the dataset using the additionnal annotations provided by [Semantic Contours from Inverse Detectors](http://home.bharathh.info/pubs/pdfs/BharathICCV2011.pdf). First download the image sets (`train_aug`, `trainval_aug`, `val_aug` and `test_aug`) from this link: [Aug ImageSets](https://www.dropbox.com/sh/jicjri7hptkcu6i/AACHszvCyYQfINpRI1m5cNyta?dl=0&lst=), and add them the rest of the segmentation sets in `/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation`, and then download new annotations [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) and add them to the path `VOCtrainval_11-May-2012/VOCdevkit/VOC2012`, now we're set, for training use the path to `VOCtrainval_11-May-2012`
  
  <p align="center"><img width="100%" src="imgs/VOC_demo.png" /></p>
  
  
  
- **COCO Stuff:** For COCO, there is two partitions, CocoStuff10k with only 10k that are used for training the evaluation, note that this dataset is outdated, can be used for small scale testing and training, and can be downloaded [here](https://github.com/nightrome/cocostuff10k). For the official dataset with all of the training 164k examples, it can be downloaded from the official [website](http://cocodataset.org/#download).
  Note that when using COCO dataset, 164k version is used per default, if 10k is prefered, this needs to be specified with an additionnal parameter `partition = 'CocoStuff164k'` in the config file with the corresponding path.
  
  <p align="center"><img width="100%" src="imgs/COCO_demo.png" /></p>



## Network Architectures  

- [ ] (**SSD**) SSD: Single Shot MultiBox Detector (2015) [[Paper\]](https://arxiv.org/pdf/1512.02325.pdf)

- [x] (**Faster-RCNN**) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015) [[Paper\]](https://arxiv.org/pdf/1506.01497.pdf)

- [ ] (**SSD**) SSD: Single Shot MultiBox Detector (2015) [[Paper\]](https://arxiv.org/pdf/1512.02325.pdf)

- [x] (**RetinaNet**) Focal Loss for Dense Object Detection (2017) [[Paper\]](https://arxiv.org/pdf/1708.02002.pdf)

- [ ] (**YOLOv3**) YOLOv3: An Incremental Improvement (2018) [[Paper\]](https://arxiv.org/pdf/1804.02767.pdf)

- [ ] (**FCOS**) Fully Convolutional One-Stage Object Detection (2020) [[Paper\]](https://arxiv.org/pdf/1904.01355.pdf)

- [ ] (**YOLOv5**) https://github.com/ultralytics/yolov5 (2020) 

- [ ] (**NanoDet**) https://github.com/RangiLyu/nanodet (2021) 

  


## Loss functions



## Evaluation metrics

- 

## REFERENCES

[1] A survey of loss functions for semantic segmentation, 2020, https://arxiv.org/pdf/2006.14822v1.pdf

[2] A Review on Deep Learning Techniques Applied to Semantic Segmentation, 2017,  https://arxiv.org/pdf/1704.06857.pdf

[3] Segmentation Loss Odyssey, 2020, https://arxiv.org/abs/2005.13449