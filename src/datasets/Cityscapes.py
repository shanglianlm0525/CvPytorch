# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/6 15:53
# @Author : liumin
# @File : Cityscapes.py



class CityscapesDataset():
    def __init__(self, root='/home/lmin/data/cityscapes/cityscapes', transform=None, target_transform=None, stage='train'):
        super(CityscapesDataset, self).__init__()


    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/Flower17/train'
    dataset = CityscapesDataset(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(20))
    print('finished!')