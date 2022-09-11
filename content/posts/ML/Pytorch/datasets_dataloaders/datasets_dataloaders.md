---
title: "[PyTorch] Datasets & DataLoaders"
date: 2022-08-18
tags: []
series: [Machine Learning, PyTorch]
categories: [ML/DL]
---

# Datasets & DataLoaders

PyTorch provides two `data primitives (基本資料型態)` that allow you to use pre-loaded datasets as well as your own data, as below:

- [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

    > Stores the samples and their corresponding labels

    Dataset 定義資料的結構並且將其包起來，利如:

    - 一張影像和一個標籤
    - 一張影像和多個標籤
    - 一張影像和 Bounding box 的座標與長寬等

- [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    > DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

     DataLoader 將資料集 (Dataset) 進行包裝，定義如何讀取資料，以及每一個 batch 要讀取多少資料 (batch size)。

```
一定要先有 Dataset 才可以用 DataLoader 操作
```

- [Built-in Datasets](#built-in-datasets)
- [Custom Dataset](#custom-dataset)

## Built-in Datasets

`Torchvision` provides many built-in datasets in the [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html) module, as well as utility classes for building your own datasets.

內建的 Dataset 如下:

- `Image classification` (只列舉幾個常用的，其他詳見[官網](https://pytorch.org/vision/stable/datasets.html#image-classification)):
  - [MNIST (手寫數字)](http://yann.lecun.com/exdb/mnist/)
  - [Fashion-MNIST (衣著)](https://github.com/zalandoresearch/fashion-mnist)
  - [ImageNet (物件、場景)](https://image-net.org/)
  - [CIFAR10 (物件)](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [CIFAR100 (物件)](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [CelebA (人臉 ID、屬性、特徵點)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [Caltech101 (物件)](https://data.caltech.edu/records/20086)
  - [Caltech256 (物件)](https://data.caltech.edu/records/20087)
  - [GTSRB (交通標誌)](https://benchmark.ini.rub.de/)
  - [Stanford Cars (汽車, fine-grained image recognition)](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

- `Image detection or segmentation`([官網](https://pytorch.org/vision/stable/datasets.html#image-detection-or-segmentation)):
  - [MS COCO](https://cocodataset.org/#detection-2016)
  - [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark)
  - [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
  - [Cityscapes](https://www.cityscapes-dataset.com/)
  - [GTSRB (交通標誌)](https://benchmark.ini.rub.de/)

### Example

Fashion-MNIST，詳細: [built_in_dataset](https://github.com/kaka-lin/ML-Notes/blob/master/Pytorch/datasets_dataloaders/built_in_dataset.py)

```python
# Loading a Dataset
training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
```

其他 Built-in Dataset 使用方式請查看[官網](https://pytorch.org/vision/stable/datasets.html)。

## Custom Dataset

A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.

建立自己的 Dataset 會需要繼承 `torch.utils.data.Dataset`，且需要實現三個function:
  - `__init__(self)`: 初始化，進行資料定義，如: self.data, self.label。

    We initialize the directory containing the images, the annotations file, and both transforms (covered in more detail in the next section).

The labels.csv file looks like:

  - `__len__(self)`: 獲取資料長度。

    The `__len__` function returns the number of samples in our dataset.

  - `__getitem__(self, index)`: 進行`資料前處理(如: Transform)`與相關讀取方式。

    接收一個索引 (index)，然後返回影像資料和相關標簽。
    其中 index 是根據 `__len__`返回值，如:

    ```
    __len__ 返回: 4
    index = 0, 1, 2, 3
    ```

建立完 Dataset 後用 `DataLoader` 進行包裝，以方便我們進行 training。如下所示:

### 1. Creating a Custom Dataset for your files

```python
import os

import numpy as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file,
                 transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### 2. Preparing your data for training with DataLoaders

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
```

### Example: Fashion-MNIST

1. Creating a Custom Dataset for your files，詳細: [custom_dataset.py](https://github.com/kaka-lin/ML-Notes/blob/master/Pytorch/datasets_dataloaders/custom_dataset/custom_dataset.py)

    ```python
    import numpy as np
    from torch.utils.data import Dataset
    from torchvision.io import read_image


    class CustomDatasetFromFile(Dataset):
        def __init__(self, image_file, label_file, transform=None, target_transform=None):
            self.image_file = image_file
            self.label_file = label_file
            self.transform = transform
            self.target_transform = target_transform

            with open(self.label_file, 'rb') as lbpath:
                self.labels = np.fromfile(lbpath, dtype=np.uint8)

            with open(self.image_file, 'rb') as imgpath:
                self.images = np.fromfile(imgpath, dtype=np.uint8).reshape(
                    len(self.labels), 28, 28)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
    ```

2. Preparing your data for training with DataLoaders，詳細: [main.py](https://github.com/kaka-lin/ML-Notes/blob/master/Pytorch/datasets_dataloaders/custom_dataset/main.py)

    ```python
    # Creating a Custom Dataset for your files
    training_data = CustomDatasetFromFile(
        image_file="data/train-images-idx3-ubyte",
        label_file="data/train-labels-idx1-ubyte",
        transform=ToTensor()
    )

    test_data = CustomDatasetFromFile(
        image_file="data/t10k-images-idx3-ubyte",
        label_file="data/t10k-labels-idx1-ubyte",
        transform=ToTensor()
    )

    # Creating DataLoader
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Iterate through the DataLoader
    ```

## Reference
- [Pytorch/Tutorials/Datasets & Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#)
- [PyTorch 自定義資料集 (Custom Dataset)](https://rowantseng.medium.com/pytorch-%E8%87%AA%E5%AE%9A%E7%BE%A9%E8%B3%87%E6%96%99%E9%9B%86-custom-dataset-7f9958a8ff15)
- [Day-19 PyTorch 怎麼讀取資料? Dataset and DataLoader ](https://ithelp.ithome.com.tw/articles/10277163)
- [Pytorch 基礎學習2_Dataset與DateLoader](https://www.wpgdadatong.com/tw/blog/detail?BID=B5207)
