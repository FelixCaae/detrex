import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from detectron2.modeling.backbone import ResNet, BasicStem
from segment_anything import sam_model_registry, SamPredictor
# 设置数据路径和域
data_root = './test_imgs/*.jpg'
# 预处理设置：调整图像尺寸并转换为张量
transform = transforms.Compose([
    transforms.Resize((560, 560)),  # R50 模型的输入尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

sam_model = 'vit_l'
sam_checkpoint = '/gpfsdata/home/caizhi/CrowdSAM/weights/sam_vit_l_0b3195.pth'
sam_model = sam_model_registry[sam_model](checkpoint=sam_checkpoint)
model = sam_model 
# dino_model =  torch.hub.load('../CrowdSAM/dinov2',
#                             'dinov2_vitl14',
#                             source='local',pretrained=False).cuda()
# dino_model.load_state_dict(torch.load('/gpfsdata/home/caizhi/CrowdSAM/weights/dinov2_vitl14_pretrain.pth'))
# model = dino_model
# 去掉 ResNet50 最后的全连接层，只提取特征
# model = torch.nn.Sequential(*list(model.children())[:-1])

# 保存图像特征和标签（域名）
features_list = []
labels_list = []

img_filenames = glob.glob(data_root)
for img_path in  (img_filenames):
    img_name = img_path.split('/')[-1]
    # annots = [annot for annot in json_content['annotations'] if annot['image_id'] == id_]
    # 加载图像并预处理
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).cuda()  # 扩展 batch 维度，并将数据传到 CUDA
    
    # 提取特征
    with torch.no_grad():
        features_dict = model.forward_features(img.cuda())
        dino_feats = features_dict['x_norm_patchtokens'].view(1, 40, 40, -1).cpu().numpy()
        # reduce dimension with PCA
        pca = PCA(n_components=3)
        # 拟合数据并转换
        dino_feats_reduced = pca.fit_transform(dino_feats.reshape(1600,-1))
        x = dino_feats_reduced = dino_feats_reduced.reshape(40,40,3)
        x = torch.tensor(x)
        x = x.permute(2,0,1)
        x = torch.nn.functional.interpolate(x.unsqueeze(0), (160,160), mode='bilinear')[0]
        x = x.permute(1,2,0)
        x = (x-x.min())/ (x.max() - x.min())
    plt.figure(figsize=(10,6))
    plt.imshow(x)
    plt.savefig(f'vis_out/{img_name}_pca.jpg')
            # feature = model(img)['res5'].mean(dim=[2,3]).cpu().numpy().flatten()
    # 保存特征和对应的标签（域）

