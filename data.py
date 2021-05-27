import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as f

Age2Class = {
    '(0, 2)': 0,
    '(4, 6)': 1,
    '(8, 12)': 2,
    '(15, 20)': 3,
    '(25, 32)': 4,
    '(38, 43)': 5,
    '(48, 53)': 6,
    '(60, 100)': 7,
}

Gender2Class = {
    'f': 0,
    'm': 1
}

# decide the image preprocessing
prepro_resize = transforms.Resize((256, 256), interpolation=2)
def img_precessing(img):
    output = prepro_resize(img)
    return output

# from mtcnn import MTCNN
# detector = MTCNN()
def mtcnn_precessing(img):
    det = detector.detect_faces(np.array(img))
    box = det[0]['box']
    output = f.crop(img, box[1], box[0], box[3], box[2])
    output = prepro_resize(img)
    return output


# decide the image augmantation
img_augment_0 = transforms.RandomHorizontalFlip()
img_augment = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomPerspective(p=0.5),
    ]),
    ])


class Facedata(Dataset):
    def __init__(self, data_path="./Adience_Benchmark", folds=[0, 1, 2, 3, 4], img_processing=img_precessing, img_augment=img_augment_0):
        images = []
        targets_gender = []
        targets_age = []

        for fold in folds:
            count = 0
            uncount = 0
            ann_file_path = os.path.join(data_path, f'fold_{fold}_data.txt')
            head = True
            for line in open(ann_file_path, 'r'):
                if head:
                    head = False
                    continue
                sample = line.split()
                img_path = os.path.join(
                    data_path,
                    'faces',
                    sample[0],
                    f"coarse_tilt_aligned_face.{sample[2]}.{sample[1]}"
                    )
                if os.path.isfile(img_path):
                    try:
                        gender = Gender2Class[sample[5]]
                        age_class = Age2Class[sample[3]+' '+sample[4]]
                        count += 1
                        targets_gender.append(gender)
                        targets_age.append(age_class)
                        images.append(img_path)
                    except:
                        uncount += 1

            print(f'Total {count} images successfully loaded, and {uncount} images failed to loaded in fold {fold}')
        self.images = images
        self.targets_gender = torch.LongTensor(targets_gender)
        self.targets_age = torch.LongTensor(targets_age)
        self.img_processing = img_processing
        self.img_augment = img_augment

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        self.to_tensor = transforms.Compose([transforms.ToTensor(), normalize])

    def __getitem__(self, index):
        path = self.images[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.img_processing is not None:
            sample = self.img_processing(sample)
        if self.img_augment is not None:
            sample = self.img_augment(sample)
        sample = self.to_tensor(sample)


        gender = self.targets_gender[index]
        age = self.targets_age[index]

        return sample, gender, age

    def __len__(self):
        return len(self.images)
