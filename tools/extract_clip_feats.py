import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
import tqdm
import numpy as np
import sys


# 设置数据路径和注释路径
data_root = './datasets/crossdomain_urban/daytime_clear'
annot_root = './datasets/crossdomain_urban/annotations/daytime_train_with_sam_masks.json'
output_dir = './datasets/crossdomain_urban/daytime_clear/dino_feats'
os.makedirs(output_dir, exist_ok=True)
# 加载注释文件（COCO格式）
with open(annot_root, 'r') as f:
    annotations = json.load(f)

# 预处理设置：调整图像尺寸并转换为张量
transform = transforms.Compose([
    transforms.Resize((560, 560)),  # R50 模型的输入尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

if int(sys.argv[1]) == 0:
    start_idx = 0 
    end_idx = len(annotations['images']) // 2
    torch.cuda.set_device(0)
else:
    start_idx = len(annotations['images']) // 2 
    end_idx = len(annotations['images']) 
    torch.cuda.set_device(1)
print(start_idx, end_idx)

import clip
model, preprocess = clip.load("ViT-B/32", device='cuda')

# 存储每个实例的特征
features_list = []


# 遍历注释文件中的每张图片
for img_info in tqdm.tqdm(annotations['images'][start_idx: end_idx]):
    img_id = img_info['id']
    img_path = os.path.join(data_root, img_info['file_name'])
    
    # 加载图像并预处理
    img = Image.open(img_path).convert('RGB')

    # 提取图像的全局特征
    with torch.no_grad():
        pre_image = preprocess(img).unsqueeze(0).cuda()
        feature = image_features = model.encode_image(pre_image).cpu().numpy().flatten()
        np.save(os.path.join( output_dir, f'{img_id}_clip.npy'), feature)