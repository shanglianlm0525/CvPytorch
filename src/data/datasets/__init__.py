from .cityscapes import CityscapesSegmentation

datasets = {
    'citys': CityscapesSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
