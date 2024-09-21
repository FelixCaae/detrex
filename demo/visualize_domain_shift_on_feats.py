import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import json
from detectron2.modeling.backbone import ResNet, BasicStem
# 设置数据路径和域
data_root = './datasets/crossdomain_urban'
domain_dirs = ["daytime_clear", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"]
json_files = ['daytime_train.json', 'daytime_foggy_val.json', 'dusk_rainy_val.json', 'night_rainy_val.json', 'night_sunny_val.json']
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
# model = models.resnet50(pretrained=True)
state_dict = torch.load('/gpfsdata/home/caizhi/detrex/output/sgod_dinoaug1_r50_20ep/model_final.pth')['model']
new_state_dict = {}
for k,v in state_dict.items():
    if 'backbone' in k:
        k = k[9:]
        new_state_dict[k] = v
model = model.eval().cuda()  # 使用 eval 模式和 CUDA 加速
model.load_state_dict(new_state_dict)
# 去掉 ResNet50 最后的全连接层，只提取特征
# model = torch.nn.Sequential(*list(model.children())[:-1])

# 保存图像特征和标签（域名）
features_list = []
labels_list = []

# 遍历每个域文件夹，提取特征
for label, ddir in enumerate(domain_dirs):
    json_file = json_files[label]
    img_dir = os.path.join(data_root, ddir)
    json_path = os.path.join(data_root, 'annotations', json_file)
    #load coco json 

    json_content = json.load(open(json_path,'r')) 
    #extract 50 images
    # img_filenames = os.listdir(img_dir)[:50]  # 每个域取 50 张图像
    images = json_content['images'][:50]
    img_filenames = [item['file_name'] for item in images]
    img_ids = [item['id'] for item in images]

    for img_name, id_ in  zip(img_filenames, img_ids):
        img_path = os.path.join(img_dir, img_name)
        # annots = [annot for annot in json_content['annotations'] if annot['image_id'] == id_]
        # 加载图像并预处理
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).cuda()  # 扩展 batch 维度，并将数据传到 CUDA
        
        # 提取特征
        with torch.no_grad():
            feature = model(img)['res5'].mean(dim=[2,3]).cpu().numpy().flatten()  # 提取并压平特征向量
        
        # 保存特征和对应的标签（域）
        features_list.append(feature)
        labels_list.append(label)

# 转换为 numpy 数组
features = np.array(features_list)
labels = np.array(labels_list)

# 使用 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

reducer = umap.UMAP(n_components=2, random_state=42)
features_umap = reducer.fit_transform(features)

# 可视化
# 可视化
plt.figure(figsize=(10, 6))

# 为了实现 legend，需要为每个域创建单独的散点图
colors = ['r', 'g', 'b', 'c', 'm']  # 不同域的颜色
domain_names = ["daytime_clear", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"]

for i, domain in enumerate(domain_names):
    idx = labels == i  # 获取当前域的索引
    plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], c=colors[i], label=domain, alpha=0.7)

# 添加颜色条和图例
plt.colorbar()
plt.legend()
plt.title('t-SNE Visualization of ResNet-50 Features with Legend')
plt.show()
plt.savefig('vis_out/tsne.jpg')
plt.close()

plt.figure(figsize=(10, 6))

# 为了实现 legend，需要为每个域创建单独的散点图
colors = ['r', 'g', 'b', 'c', 'm']  # 不同域的颜色
domain_names = ["daytime_clear", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"]

for i, domain in enumerate(domain_names):
    idx = labels == i  # 获取当前域的索引
    plt.scatter(features_umap[idx, 0], features_umap[idx, 1], c=colors[i], label=domain, alpha=0.7)

# 添加颜色条和图例
plt.colorbar()
plt.legend()
plt.title('UMAP Visualization of ResNet-50 Features with Legend')
plt.show()
plt.savefig('vis_out/umap.jpg')
plt.close()