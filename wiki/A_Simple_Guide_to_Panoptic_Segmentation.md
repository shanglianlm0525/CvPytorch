# A Simple Guide to Panoptic Segmentation

**Panoptic   Segmentation：** Given an input image, assign a label to every pixel (e.g., background, bottle, hand, sky, etc.).

## Datasets  

- **COCO** panoptic is one of the most commonly used datasets for panoptic segmentation. It has
  133 categories (80 “thing” categories with instance-level annotation and 53 “stuff” categories) in
  118k images for training and 5k images for validation. 
  
  <p align="center"><img width="100%" src="imgs/COCO_demo.png" /></p>
  
- **ADE20K** panoptic combines the ADE20K semantic segmentation annotation for semantic
  segmentation from the SceneParse150 challenge and ADE20K instance annotation from the
  COCO+Places challenge. Among the 150 categories, there are 100 “thing” categories with
  instance-level annotation. 
  
  <p align="center"><img width="100%" src="imgs/VOC_demo.png" /></p>

## Network Architectures  

- (**Deeplab V3**) Rethinking Atrous Convolution for Semantic Image Segmentation, 2017 [[Paper\]](https://arxiv.org/abs/1706.05587)

  

  


## Loss functions



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


## Performance 

### Deeplab V3
      |      |      |      |      
 :--------        | :--------: | :--------: |  :----:   |  ------   
      |      |      |      |      
      |      |      |      |      
      |      |      |      |


## REFERENCES

[1] A survey of loss functions for semantic segmentation, 2020, https://arxiv.org/pdf/2006.14822v1.pdf

[2] A Review on Deep Learning Techniques Applied to Semantic Segmentation, 2017,  https://arxiv.org/pdf/1704.06857.pdf

[3] Segmentation Loss Odyssey, 2020, https://arxiv.org/abs/2005.13449