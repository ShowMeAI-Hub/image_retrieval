# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os

from torch.utils.data import dataloader, Dataset
from PIL import Image


def get_file_list(file_path_list, sort=True):
    """
    递归读取文件夹下的所有文件内容
    :param file_path: 文件夹路径
    :return: 文件列表
    """
    import random
    if isinstance(file_path_list, str):
        file_path_list = [file_path_list]
    file_lists = []
    for file_path in file_path_list:
        assert os.path.isdir(file_path)
        file_list = os.listdir(file_path)
        if sort:
            file_list.sort()
        else:
            random.shuffle(file_list)
        file_list = [file_path + file for file in file_list]
        file_lists.append(file_list)
    if len(file_lists) == 1:
        file_lists = file_lists[0]
    return file_lists


class Gallery(Dataset):
    """
    预读取数据
    """

    def __init__(self, image_paths, transform=None):
        super().__init__()

        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, image_path

    def __len__(self):
        return len(self.image_paths)


# 加载数据
def load_data(data_path, batch_size=1, shuffle=False, transform='default'):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if transform == 'default' else transform

    image_path_list = get_file_list(data_path)

    gallery_data = Gallery(image_paths=image_path_list,
                           transform=data_transform,
                           )

    data_loader = dataloader.DataLoader(dataset=gallery_data,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=0,
                                        )
    return data_loader


# 抽取CNN特征
def extract_feature(model, dataloaders, use_gpu=True):
    features = torch.FloatTensor()
    path_list = []

    use_gpu = use_gpu and torch.cuda.is_available()
    for img, path in dataloaders:
        img = img.cuda() if use_gpu else img
        input_img = Variable(img.cuda()) if use_gpu else Variable(img)
        outputs = model(input_img)
        ff = outputs.data.cpu()
        # 特征归一化
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        path_list += list(path)
    return features, path_list

# 抽取检索图片特征
def extract_feature_query(model, img, use_gpu=True):
    c, h, w = img.size()
    img = img.view(-1,c,h,w)
    use_gpu = use_gpu and torch.cuda.is_available()
    img = img.cuda() if use_gpu else img
    input_img = Variable(img)
    outputs = model(input_img)
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff,p=2,dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff


# 加载检索图片
def load_query_image(query_path):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    query_image = datasets.folder.default_loader(query_path)
    query_image = data_transforms(query_image)
    return query_image


# 加载模型
def load_model(pretrained_model=None, use_gpu=True):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    add_block = []
    add_block += [nn.Linear(num_ftrs, 30)]  #number of training classes
    model.fc = nn.Sequential(*add_block)
    model.load_state_dict(torch.load(pretrained_model))

    # 去掉最后的fc层
    model.fc = nn.Sequential()
    # 切换到test模式
    model = model.eval()
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model


# 图片排序
def sort_img(qf, gf):
    score = gf*qf
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    import numpy as np
    s = np.around(s, 3)
    return s, index


if __name__ == '__main__':

    # 准备好数据
    data_loader = load_data(data_path='../static/image_database/',
                            batch_size=2,
                            shuffle=False,
                            transform='default',
                            )

    # 加载预训练模型
    model = load_model(pretrained_model='./models/pretrained_model.pth', use_gpu=True)

    # 抽取图像特征
    gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader)

    # 加载检索图片
    query_image = load_query_image('../static/upload_image/query.jpg')

    # 抽取检索图片特征
    query_feature = extract_feature_query(model=model, img=query_image)

    # 排序
    similarity, index = sort_img(query_feature, gallery_feature)

    sorted_paths = [image_paths[i] for i in index]
    print(sorted_paths)
