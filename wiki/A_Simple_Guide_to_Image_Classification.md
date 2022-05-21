# A Simple Guide to Image Classification

**Image classification**: Given an input image, predict what object is present in the image.

## Datasets  

- **ImageNet**:  The ILSVRC  2012 is the most popular image classification dataset. It contains 1.2 million images for training, and 50,000 for validation with 1,000 categories. It is also widely used for training a pretrained model for downstream tasks, like object detection or semantic segmentation.  

  <p align="center"><img width="100%" src="imgs/CamVid_demo.png" /></p>



## Network Architectures  

- (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper\]](https://arxiv.org/abs/1802.02611)

  

  


## Loss functions



## Evaluation metrics

- 



## Benchmarks

###  mini-ImageNet:

|               | Model                                                        | Resolution | mAcc | FPS  |      |
| :-----------: | :----------------------------------------------------------- | :--------: | :--- | :--: | ---- |
|    AlexNet    |                                                              |  224*224   |      |      |      |
|      VGG      | vgg11<br/>vgg13<br/>vgg16<br/>vgg19                          |  224*224   |      |      |      |
|    ResNet     | resnet18<br/>resnet34<br/>resnet50<br/>resnet101<br/>resnet152 |  224*224   |      |      |      |
|   DenseNet    | densenet121<br/>densenet161<br/>densenet169<br/>densenet201  |  224*224   |      |      |      |
|    resnext    |                                                              |  224*224   |      |      |      |
|               |                                                              |  224*224   |      |      |      |
| shufflenet_v1 |                                                              |  224*224   |      |      |      |
| shufflenet_v2 |                                                              |  224*224   |      |      |      |
| mobilenet_v1  |                                                              |  224*224   |      |      |      |
| mobilenet_v2  |                                                              |  224*224   |      |      |      |
| mobilenet_v3  |                                                              |  224*224   |      |      |      |
|  squeezenet   |                                                              |  224*224   |      |      |      |



## REFERENCES

https://github.com/wyharveychen/CloserLookFewShot