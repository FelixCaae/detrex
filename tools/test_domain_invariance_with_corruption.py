import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import tqdm
from detrex.data.transforms import ACVCGenerator
from detectron2.modeling.backbone import ResNet, BasicStem

import json
import seaborn as sns
import sys
import clip

# 设置数据路径和域
data_root = './datasets/crossdomain_urban'
domain_dir = "daytime_clear"
acvc = ACVCGenerator()
corruptions=[[],['gaussian_blur'],['gaussian_noise'],['contrast'],['high_pass_filter']]

# corruptions = [
#     [],
#     ["defocus_blur",
#     "glass_blur",
#     "gaussian_blur",
#     "motion_blur"],
#     ["speckle_noise",
#     "shot_noise",
#     "impulse_noise",
#     "gaussian_noise"],
#     ["jpeg_compression",
#     "pixelate",
#     "elastic_transform",
#     "brightness",
#     "saturate",
#     "contrast"],
#     ["high_pass_filter",
#     "phase_scaling"]
# ]
domain_names = corruptions#['original', 'blur','noise','digital',  'fourier']#corruptions#["daytime_clear", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"]


def apply_corrupt(img, corrupt):
    if corrupt == []:
        return Image.fromarray(img)
    # s = np.random.randint(1,5)  # 可以根据需要随机化
    s = 3
    aug_img = img
    c = np.random.choice(corrupt, 1, replace=False)[0]
    aug_img = acvc.apply_corruption(aug_img, c, s)
    # for c in corrupt:
    #     aug_img = np.array(aug_img)
    #     aug_img = acvc.apply_corruption(aug_img, c, s)
    return aug_img.convert("RGB")

# 预处理设置：调整图像尺寸并转换为张量
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # R50 模型的输入尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

# 设置默认模型类型
model_type = 'resnet50'
# 允许运行时修改
if len(sys.argv) > 1:
    model_type = sys.argv[1]

# 读取模型
if model_type == 'resnet50':
    from torchvision.models import resnet50
    model = resnet50(pretrained=True).cuda()
    # 去掉 ResNet50 最后的全连接层，只提取特征
    model = torch.nn.Sequential(*list(model.children())[:-2])
elif model_type == 'resnet50_dino':
    # 加载 ResNet50 预训练模型
    model = ResNet(
            stem =BasicStem(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=ResNet.make_default_stages(
                depth=50,
                stride_in_1x1=False,
                norm="FrozenBN",
            ),
            out_features=["res3", "res4", "res5"],
            freeze_at=1)
    model = models.resnet50(pretrained=True)
    state_dict = torch.load('/gpfsdata/home/caizhi/detrex/output/sgod_dinoaug1_r50_20ep/model_final.pth')['model']
    new_state_dict = {}
    for k,v in state_dict.items():
        if 'backbone' in k:
            k = k[9:]
            new_state_dict[k] = v
    model = model.eval().cuda()  # 使用 eval 模式和 CUDA 加速
    model.load_state_dict(new_state_dict)

elif model_type == 'swin':
    from transformers import SwinForImageClassification, SwinImageProcessor
    model = SwinTransformer(
        pretrain_img_size=224,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        out_indices=(1, 2, 3),)
    state_dict = torch.load('/gpfsdata/home/caizhi/detrex/swin_base_patch4_window7_224_22k.pth')['model']
    model.load_state_dict(state_dict)
elif model_type == 'clip':
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    
else:
    dino_model =  torch.hub.load('../CrowdSAM/dinov2',
                                'dinov2_vitl14',
                                source='local',pretrained=False).cuda()
    dino_model.load_state_dict(torch.load('/gpfsdata/home/caizhi/CrowdSAM/weights/dinov2_vitl14_pretrain.pth'))
    model = dino_model

print(model_type)
# 去掉 ResNet50 最后的全连接层，只提取特征
# model = torch.nn.Sequential(*list(model.children())[:-1])
# 保存图像特征和标签（域名）
features_list = []
labels_list = []

# 遍历每个域文件夹，提取特征
# for label, ddir in enumerate(domain_dirs):
for label,corrupt in enumerate(corruptions):
    print(f"processing {label} ")
    img_dir = os.path.join(data_root, domain_dir, 'VOC2007', 'JPEGImages')
    img_filenames = os.listdir(img_dir)[:50]  # 每个域取 50 张图像
    
    for img_name in tqdm.tqdm(img_filenames): 
        img_path = os.path.join(img_dir, img_name)
        
        # 加载图像并预处理
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        # # 随机选择起始点进行裁剪
        # left = np.random.randint(0, width - 224)
        # top = np.random.randint(0, height - 224)
        # right = left + 224
        # bottom = top + 224
        
        # # 裁剪图像
        # img = img.crop((left, top, right, bottom))
        img = np.array(img)

        origimg = img = apply_corrupt(img, corrupt)
        img = transform(img).unsqueeze(0).cuda()  # 扩展 batch 维度，并将数据传到 cpu
        
        # 提取特征
        # 提取特征
        with torch.no_grad():
            if model_type  == 'resnet50':
                feature = model(img).mean(dim=[2,3]).cpu().numpy().flatten()  # 提取并压平特征向量
            elif model_type == 'resnet50_dino':
                feature = model(img)['res5'].mean(dim=[2,3]).cpu().numpy().flatten()  # 提取并压平特征向量
            elif model_type == 'dino':
                features_dict = model.forward_features(img)
                feature = features_dict['x_norm_clstoken'].flatten().cpu().numpy() 
                # feature = features_dict['cls_token'].mean(dim=[0,1]).cpu().numpy()  # 提取patch tokens
            elif model_type == 'clip':
                pre_image = preprocess(origimg).unsqueeze(0).cuda()
                feature = image_features = model.encode_image(pre_image).cpu().numpy().flatten()
                print(feature.shape)
            else:
                feature = model(img).cpu().numpy().flatten()
        # 保存特征和对应的标签（域）
        features_list.append(feature)
        labels_list.append(label)

# 转换为 numpy 数组
features = np.array(features_list)
labels = np.array(labels_list)

domain_names = ["no corruption", "blur", "noise", "contrast", "high_pass_filter"]
sim_type = 'cosine'
def cka_similarity(X, Y):
    # 计算两个特征矩阵的内积
    K = X @ X.T
    L = Y @ Y.T

    # 中心化
    H = np.eye(K.shape[0]) - np.ones(K.shape) / K.shape[0]
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    # 计算CKA
    hsic = np.sum(K_centered * L_centered)
    norm_x = np.sqrt(np.sum(K_centered * K_centered))
    norm_y = np.sqrt(np.sum(L_centered * L_centered))
    cka_score = hsic / (norm_x * norm_y)
    
    return cka_score
def cosine_similarity(X, Y):
    # 计算均值
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    
    # 计算 L2 范数
    X_norm = np.linalg.norm(X_mean)
    Y_norm = np.linalg.norm(Y_mean)
    
    # 计算余弦相似度
    if X_norm == 0 or Y_norm == 0:
        return 0.0  # 避免除以零
    
    return np.dot(X_mean, Y_mean) / (X_norm * Y_norm)
# 初始化相似度

sims = np.zeros((5,5))
# 计算两两相似度
for i in range(5):
    for j in range(i,5):
        if sim_type == 'cka':
            sim_score = cka_similarity(features[labels==i], features[labels==j])
        else:
            sim_score = cosine_similarity(features[labels==i], features[labels==j])
        sims[i,j] = sim_score
        sims[j,i] = sim_score
# 绘制结果
plt.figure(figsize=(8, 6))
sns.heatmap(sims, annot=True,  fmt="f", cmap="Blues", cbar=False, xticklabels=domain_names, yticklabels=domain_names)
plt.xlabel("Source Domains")
plt.ylabel("Target Domains")
plt.title(f"{sim_type} similarity of {model_type} features between different domains")
plt.show()
plt.savefig(f'{sim_type}_{model_type}_with_corruption.png')