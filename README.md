CvPytorch

CvPytorch is an open source COMPUTER VISION toolbox based on PyTorch.

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
* [ADE20K]()
* [COCO]()
* [PennPed]()
* [CamVid]()

### Install

#### 



### Training

For this example, we will use [COCO](https://github.com/ultralytics/yolov5/blob/master/data/get_coco2017.sh) dataset with `yolov5l.yaml` . Feel free to use your own custom dataset and configurations.

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

  