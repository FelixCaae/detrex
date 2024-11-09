import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
from torchvision.ops import roi_align
import tqdm
from matplotlib import pyplot as plt
from matplotlib import patches
from pycocotools import mask
import torch.nn.functional as F

from sklearn.manifold import TSNE
import umap

# 设置数据路径和注释路径
data_root = './datasets/crossdomain_urban/daytime_clear'
annot_root = './datasets/crossdomain_urban/annotations/daytime_train_with_sam_masks.json'
# data_root = './datasets/crossdomain_urban/daytime_foggy'
# annot_root = './datasets/crossdomain_urban/annotations/daytime_foggy_val.json'

# 加载注释文件（COCO格式）
with open(annot_root, 'r') as f:
    annotations = json.load(f)

# 预处理设置：调整图像尺寸并转换为张量
transform = transforms.Compose([
    transforms.Resize((560, 560)),  # R50 模型的输入尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

# 加载DINO模型
dino_model = torch.hub.load('../CrowdSAM/dinov2',
                            'dinov2_vitl14',
                            source='local', pretrained=False).cuda()
dino_model.load_state_dict(torch.load('/gpfsdata/home/caizhi/CrowdSAM/weights/dinov2_vitl14_pretrain.pth'))
model = dino_model

# 存储每个实例的特征
features_list = []

data_length = 100
# 遍历注释文件中的每张图片
for img_info in tqdm.tqdm(annotations['images'][:data_length]):
    img_id = img_info['id']
    img_path = os.path.join(data_root, img_info['file_name'])
    
    # 加载对应的目标框（bounding boxes）
    instances = [annot for annot in annotations['annotations'] if annot['image_id'] == img_id]
    
    # 加载图像并预处理
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()  # 扩展 batch 维度，并将数据传到 CUDA
    
    # 提取图像的全局特征
    with torch.no_grad():
        features_dict = model.forward_features(img_tensor)
        dino_feats = features_dict['x_norm_patchtokens'].view(1, 40, 40, -1).cpu()  # 提取patch tokens

    # plt.imshow(img)
    img_width, img_height = img.size
    # 遍历该图像的所有bounding boxes
    for instance in instances:
        # COCO格式中的bounding box格式是 [x_min, y_min, width, height]

        x,y,w,h = instance['bbox']
        scale_x, scale_y = 40 / img_width, 40 / img_height
        x,y,w,h = int(x * scale_x), int(y * scale_y), int(w * scale_x),  int(h * scale_y)
        if w == 0 or h == 0:
            continue
        # rles = instance['segmentation']
        # #decode rle segm
        # segm = torch.from_numpy(mask.decode(rles))
        # segm = F.interpolate(segm.unsqueeze(0).unsqueeze(0).float(), (40,40), mode='bilinear').bool()[0,0]
        # if segm.sum() == 0:
        #     continue
        # region_feats = dino_feats[0,segm]
        
        region_feats = dino_feats[0, y:y+h, x:x+w, :]
        # 使用ROI Align进一步调整特征大小
        # region_feats = roi_align(region_feats, [torch.tensor([[x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled]])], output_size=(7, 7))
        region_feats = region_feats.mean(dim=[0,1])
        # 将特征存储起来
        features_list.append({
            'category_id':instance['category_id'],
            'features': region_feats.squeeze(0).numpy()  # 保存特征
        })
    # plt.savefig('test.jpg')
# 提取每个类别对应的feature，做平均
category_ids = [item['id'] for item in annotations['categories']]
prototype_list = []
for id_ in category_ids:
    feat_list = [item['features'] for item in features_list if item['category_id'] == id_]
    feat_list = np.array(feat_list)
    print(str(id_), len(feat_list))
    prototype = feat_list.mean(axis=0)
    prototype_list.append(prototype)


# # 验证prototype分割性能
# import pdb;pdb.set_trace()
# sim = F.cosine_similarity(dino_feats.flatten(0,2), torch.tensor(prototype_list[2]))
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.subplot(1,2,2)
# plt.imshow(sim.reshape(40,40))
# plt.savefig('vis_out/car_seg_demo.jpg')
# plt.close()

# 保存特征为npy文件
output_path = './dino_features.npy'
np.save(output_path, prototype_list)
print(f"Features saved at {output_path}")


# 为了实现 legend，需要为每个域创建单独的散点图
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange'] # 不同域的颜色
domain_names = ['bus', 'bike', 'car', 'motor', 'person', 'rider', 'truch']

# reducer = umap.UMAP(n_components=2, random_state=42)
features = []
labels = []
for item in features_list:
    labels.append(item['category_id'])
    features.append(item['features'])
features = np.stack(features)

# features_umap = reducer.fit_transform(features)
tsne = TSNE(n_components=2, random_state=42, metric='cosine')
features_tsne = tsne.fit_transform(features)
labels = np.stack(labels)
for i, domain in enumerate(domain_names):
    idx = labels == i  # 获取当前域的索引
    plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], c=colors[i], label=domain, alpha=0.7)

# 添加颜色条和图例
plt.figure
plt.colorbar()
plt.legend()
plt.title('TSNE Visualization of DINO Features with Legend')
plt.show()
plt.savefig('vis_out/prototypes_tsne.jpg')
plt.close()