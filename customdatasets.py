import torch
import numpy as np
from skimage.io import imread
from torch.utils import data
from tqdm.notebook import tqdm
import cv2
from scipy import ndimage
from skimage.measure import label as sk_label
from skimage.measure import regionprops


## old custom dataset for training and validation 
class SegmentationDataSet0(data.Dataset):
    #新たに追加したクラス。Numpyを読み込むようにする
    """Most basic image segmentation dataset."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        x = self.inputs[index]
        y = self.targets[index]

        # Load input and target
        #x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y


class SegmentationDataSet1(data.Dataset):
    """Most basic image segmentation dataset."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y


class SegmentationDataSet2(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = imread(str(img_name)), imread(str(tar_name))
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y


class SegmentationDataSet3(data.Dataset):
    """Image segmentation dataset with caching, pretransforms and multiprocessing."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets, repeat(self.pre_transform)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(str(inp)), imread(str(tar))
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar


class SegmentationDataSet4(data.Dataset):
    """Image segmentation dataset with caching, pretransforms and multiprocessing. Output is a dict."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            from multiprocessing import Pool
            from itertools import repeat

            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_images, zip(inputs, targets, repeat(self.pre_transform)))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return {'x': x, 'y': y, 'x_name': self.inputs[index].name, 'y_name': self.targets[index].name}

    @staticmethod
    def read_images(inp, tar, pre_transform):
        inp, tar = imread(str(inp)), imread(str(tar))
        if pre_transform:
            inp, tar = pre_transform(inp, tar)
        return inp, tar


class SegmentationDataSetRandom(data.Dataset):
    """Random image segmentation dataset for testing purposes."""
    def __init__(self,
                 num_samples,
                 size,
                 num_classes=4,
                 inputs_dtype=torch.float32,
                 targets_dtype=torch.long
                 ):
        self.num_samples = num_samples
        self.size = size
        self.num_classes = num_classes
        self.inputs_dtype = inputs_dtype
        self.targets_dtype = targets_dtype
        self.cached_data = []

        # Generate some random input target pairs
        for num in range(self.num_samples):
            inp = torch.from_numpy(np.random.uniform(low=0, high=1, size=size))
            tar = torch.randint(low=0, high=num_classes, size=size[1:])
            self.cached_data.append((inp, tar))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        x, y = self.cached_data[index]

        # Typecasting
        x, y = x.type(self.inputs_dtype), y.type(self.targets_dtype)

        return {'x': x, 'y': y, 'x_name': f'x_name_{index}', 'y_name': f'y_name_{index}'}
    
    
    
    


## new dataset for lazy loading
class SegmentationDataSet5(data.Dataset):
    def __init__(self, feature_paths: list, label_paths:list, feature_num, transform=None):
        self.feature_paths = feature_paths  # list of lists: [[f1, f2, ..., fn], [f1, ..., fn], ...]
        self.label_paths = label_paths      # list of strings
        self.transform = transform
        self.feature_num = feature_num
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, index):
        
        dataset_img =  np.zeros((256, 256, self.feature_num), dtype=np.float32)
        for m, feature_img_path in enumerate(self.feature_paths[index]):
            input_featureimg = cv2.imread(feature_img_path, cv2.IMREAD_GRAYSCALE)
            dataset_img[:, :, m] = input_featureimg
        
        # Load label
        label_path = self.label_paths[index]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)


        x = dataset_img
        y = label  

        if self.transform is not None:
            x, y = self.transform(x, y)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
    
    


    
    ## custom dataset for mask rcnn
class SegmentationDataSetMaskRcnn(data.Dataset):
    def __init__(self,input_paths:list, label_paths:list, transform=None):
        self.input_paths = input_paths
        self.image_paths = label_paths
        self.label_paths = label_paths      # list of strings
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, index):
        # in this case feature paths and label paths are same
        # but keep separate logically
        # for p in self.label_paths:
        #     print(p, type(p))

        # reading multi class mask
        #dataset_img =  np.zeros((256, 256, 1), dtype=np.float32)
        
        #unet based mrcnnn
        #input_img = cv2.imread(self.label_paths[index], cv2.IMREAD_GRAYSCALE)
        
        #pure mrcnn on gry
        input_img = cv2.imread(self.input_paths[index], cv2.IMREAD_GRAYSCALE)
        
        dataset_img = input_img
        # convert it into binary mask, only when unet based mrcnn
        #dataset_img = (dataset_img > 0).astype(np.uint8)
        
        
        # Load label
        label_path = self.label_paths[index]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)


        x = dataset_img
        y = label  
        x = torch.from_numpy(x).float()
        # because rcnn with pretrained requires 3 channels so copying same tensor 3 times
        if x.ndim == 2:  # shape [H, W]
            x = x.unsqueeze(0).repeat(3, 1, 1)  # [1, H, W] → [3, H, W]
        elif x.shape[0] == 1:  # [1, H, W]
            x = x.repeat(3, 1, 1)
        y = yToRcnn(y)
        
        return x, y

def yToRcnn(y):
    
    masks = []
    boxes = []
    labels = []
    
    unique_val = np.unique(y)
    unique_val = unique_val[unique_val != 0]
    
    for val in unique_val:
        single_class_mask = (y==val).astype(np.uint8)
        pos = np.where(single_class_mask)
        
        if pos[0].size == 0 or pos[1].size == 0:
            continue # skipping empty mask
        
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        
        boxes.append([xmin, ymin, xmax, ymax])
        masks.append(torch.tensor(single_class_mask, dtype = torch.uint8))
        labels.append(val)
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.stack(masks)
        }
        
    return target