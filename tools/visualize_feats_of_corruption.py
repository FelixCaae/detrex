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
# model = models.resnet50(pretrained=False)
state_dict = torch.load('/gpfsdata/home/caizhi/detrex/output/sgod_dinoaug1_r50_20ep/model_final.pth')['model']
new_state_dict = {}
for k,v in state_dict.items():
    if 'backbone' in k:
        k = k[9:]
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model = model.eval().cpu()  # 使用 eval 模式和 cpu 加速

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
        
        # 随机选择起始点进行裁剪
        left = np.random.randint(0, width - 224)
        top = np.random.randint(0, height - 224)
        right = left + 224
        bottom = top + 224
        
        # 裁剪图像
        img = img.crop((left, top, right, bottom))
        img = np.array(img)

        img = apply_corrupt(img, corrupt)
        img = transform(img).unsqueeze(0).cpu()  # 扩展 batch 维度，并将数据传到 cpu
        
        # 提取特征
        with torch.no_grad():
            feature_dict = model(img)
            feature = feature_dict['res5'].cpu().numpy().flatten()  # 提取并压平特征向量
        
        # 保存特征和对应的标签（域）
        features_list.append(feature)
        labels_list.append(label)

# 转换为 numpy 数组
features = np.array(features_list)
labels = np.array(labels_list)

# 使用 t-SNE 降维
tsne = TSNE(n_components=2, random_state=18)
features_tsne = tsne.fit_transform(features)

reducer = umap.UMAP(n_components=2, random_state=42)
features_umap = reducer.fit_transform(features)

# 可视化
# 可视化
plt.figure(figsize=(10, 6))

# 为了实现 legend，需要为每个域创建单独的散点图
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple']  # 不同域的颜色
for i, domain in enumerate(domain_names):
    idx = labels == i  # 获取当前域的索引
    plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], c=colors[i], label=domain, alpha=0.7)

# 添加颜色条和图例
plt.colorbar()
plt.legend()
plt.title('t-SNE Visualization of ResNet-50 Features with Legend')
plt.show()
plt.savefig('vis_out/augment_cluster_tsne.jpg')
plt.close()

plt.figure(figsize=(10, 6))

for i, domain in enumerate(domain_names):
    idx = labels == i  # 获取当前域的索引
    plt.scatter(features_umap[idx, 0], features_umap[idx, 1], c=colors[i], label=domain, alpha=0.7)

# 添加颜色条和图例
plt.colorbar()
plt.legend()
plt.title('UMAP Visualization of ResNet-50 Features with Legend')
plt.show()
plt.savefig('vis_out/augment_cluster_umap.jpg')
plt.close()