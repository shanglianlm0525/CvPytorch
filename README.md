CvPytorch

CvPytorch is an open source COMPUTER VISION toolbox based on PyTorch.



## What's New!!!

- [2021.07.23] Release **nanodet-repvgg** models with RepVGG backbone on [COCO](http://mscoco.org/) (**27.16mAP**).
- [2021.07.23] Release **nanodet-g** models with cspnet backbone on [COCO](http://mscoco.org/) (**23.54mAP**).
- [2021.07.23] Release **nanodet-efficientnet_lite** models with efficientnet_lite backbone on [COCO](http://mscoco.org/) (**25.65mAP**).
- [2021.07.22] Release **nanodet-t** models with Transformer neck on [COCO](http://mscoco.org/) (**21.97mAP**).
- [2021.07.20] Release **nanodet-416** models with shufflenetv2 backbone on [COCO](http://mscoco.org/) (**23.30mAP**).
- [2021.07.07] Release **STDC** models with stdc2 backbone on  [Cityscapes](https://www.cityscapes-dataset.com/) (**73.36mIoU**).
- [2021.07.06] Release **STDC** models with stdc1 backbone on  [Cityscapes](https://www.cityscapes-dataset.com/) (**72.89mIoU**).
- [2021.07.05] Release **nanodet-320** models with shufflenetv2 backbone on [COCO](http://mscoco.org/) (**20.54mAP**).
- [2021.07.01] Release **deeplabv3plus** models with resnet50 backbone on [Cityscapes](https://www.cityscapes-dataset.com/) (**72.96mIoU**).
- [2021.06.28] Release **Unet** models on [Cityscapes](https://www.cityscapes-dataset.com/) (**56.90mIoU**).
- [2021.06.20] Release **PSPNet** models with resnet50 backbone on [Cityscapes](https://www.cityscapes-dataset.com/) (**72.59mIoU**).
- [2021.06.15] Release **deeplabv3** models with mobilenet_v2, resnet50 and resnet101 backbone on [Cityscapes](https://www.cityscapes-dataset.com/) (**68.06, 71.53 and 72.83mIoU**).



## Dependencies

- Python 3.8
- PyTorch 1.6.0
- Torchvision 0.7.0
- tensorboardX 2.1 

## Models 

## Image Classification
- [x] (**VGG**) VGG: Very Deep Convolutional Networks for Large-Scale Image Recognition

- [x] (**ResNet**) ResNet: Deep Residual Learning for Image Recognition

- [x] (**DenseNet**) DenseNet: Densely Connected Convolutional Networks

- [x] (**ShuffleNet**) ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

- [x] (**ShuffleNet V2**) ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design

## Object Detection
- [x] (**SSD**) SSD: Single Shot MultiBox Detector

- [ ] (**Faster R-CNN**) Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

- [ ] (**YOLOv3**) YOLOv3: An Incremental Improvement
- [ ] (**YOLOv5**) 

- [ ] (**FPN**) FPN: Feature Pyramid Networks for Object Detection

- [ ] (**FCOS**) FCOS: Fully Convolutional One-Stage Object Detection

## Semantic Segmentation
- [ ] (**FCN**) Fully Convolutional Networks for Semantic Segmentation 

- [x] (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

- [x] (**PSPNet**) Pyramid Scene Parsing Network

- [x] (**ENet**) A Deep Neural Network Architecture for Real-Time Semantic Segmentation

- [x] (**U-Net**) Convolutional Networks for Biomedical Image Segmentation

- [x] (**SegNet**) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation

## Instance Segmentation
- [x] (**Mask-RCNN**) Mask-RCNN

### Datasets

* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [COCO](http://mscoco.org/)
* [PennPed](https://www.cis.upenn.edu/~jshi/ped_html/)
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid)

### Install

#### 



### Training

For this example, we will use hymenoptera dataset with `conf/hymenoptera.yml` . Feel free to use your own custom dataset and configurations.

#### Single GPU:

```bash
$ python trainer.py --setting 'conf/hymenoptera.yml'
```

#### Multiple GPUs:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 trainer.py --setting 'conf/hymenoptera.yml'
```

### Inference



### TODO
- [x] Train Custom Data

- [x] Multi-GPU Training

- [x] Mixed Precision Training

- [x] Warm-Up

- [ ] Model Pruning/Sparsity

- [ ] Quantization

- [ ] TensorRT Deployment

- [ ] ONNX and TorchScript Export

- [ ] Class Activation Mapping (CAM)

- [ ] Test-Time Augmentation (TTA)


## License

MIT License

Copyright (c) 2020 min liu

  
