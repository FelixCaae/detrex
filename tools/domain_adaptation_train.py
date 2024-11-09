import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import json
from detectron2.modeling.backbone import ResNet, BasicStem
import numpy as np
import seaborn as sns
import sys
import clip
import itertools
import torch.nn.functional as F

class DropMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        p: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.p = p

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if i < self.num_layers - 1:  # Optional: Avoid dropout after the last layer
                x = F.dropout(x, self.p, training=self.training)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

domain_names = ["daytime_clear", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"]
sim_type = 'cosine'
def cka_similarity(X, Y):
    # 计算两个特征矩阵的内积
    K = X @ X.T
    L = Y @ Y.T

    # 中心化
    H = torch.eye(K.shape[0]) - torch.ones(K.shape) / K.shape[0]
    H = H.to(X.device)
    K_centered = H @ K @ H
    L_centered = H @ L @ H

    # 计算CKA
    hsic = torch.sum(K_centered * L_centered)
    norm_x = torch.sqrt(torch.sum(K_centered * K_centered))
    norm_y = torch.sqrt(torch.sum(L_centered * L_centered))
    cka_score = hsic / (norm_x * norm_y)
    return cka_score

def cosine_similarity(X, Y):
    # 计算均值
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    return F.cosine_similarity(X_mean.unsqueeze(0), Y_mean.unsqueeze(0))
    # 计算 L2 范

# 初始化相似度
# 计算两两相似度
def compute_sim(features, labels, sim_type):
    sims = []
    for i in range(5):
        for j in range(i,5):
            if sim_type == 'cka':
                sim_score = cka_similarity(features[labels==i], features[labels==j])
            else:
                sim_score = cosine_similarity(features[labels==i], features[labels==j])
            sims.append(sim_score)
    return sum(sims)/len(sims)
#

# 设置数据路径和域   
data_root = './datasets/crossdomain_urban'
domain_dirs = ["daytime_clear", "daytime_foggy", "dusk_rainy", "night_rainy", "night_sunny"]
json_files = ['daytime_train.json', 'daytime_foggy_val.json', 'dusk_rainy_val.json', 'night_rainy_val.json', 'night_sunny_val.json']

# 设置最大样本数目
sample_num = 20
train_num = 80

iterations = 2000

# 预处理设置：调整图像尺寸并转换为张量
transform = transforms.Compose([
    transforms.Resize((448, 448)),  # R50 模型的输入尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
])

# 设置默认模型类型
model_type = 'resnet50'
# 允许运行时修改
if len(sys.argv) > 1:
    model_type = sys.argv[1]
device = ['cpu', 'cuda'][1]
# 读取模型

dino_model =  torch.hub.load('../CrowdSAM/dinov2',
                            'dinov2_vitl14',
                            source='local',pretrained=False).to(device)
dino_model.load_state_dict(torch.load('/gpfsdata/home/caizhi/CrowdSAM/weights/dinov2_vitl14_pretrain.pth'))
model = dino_model

print(model)

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
    # img_filenames = os.listdir(img_dir)[:50]  # 每个域取 50 张图像
    images = json_content['images'][:sample_num]
    img_filenames = [item['file_name'] for item in images]
    img_ids = [item['id'] for item in images]
    import tqdm
    for img_name, id_ in  tqdm.tqdm(zip(img_filenames, img_ids)):
        img_path = os.path.join(img_dir, img_name)
        # annots = [annot for annot in json_content['annotations'] if annot['image_id'] == id_]
        # 加载图像并预处理
        origimg = img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)  # 扩展 batch 维度，并将数据传到 CUDA
        
        # 提取特征
        with torch.no_grad():
          
            features_dict = model.forward_features(img)
            feature = features_dict['x_norm_patchtokens'].flatten(0,1).mean(dim=0).cpu().numpy()                 
        # 保存特征和对应的标签（域）
        features_list.append(feature)
        labels_list.append(label)

# 转换为 numpy 数组
features = np.array(features_list)
labels = np.array(labels_list)
rand_ind = np.random.choice(np.arange(len(features)), len(features), replace=False)
features = features[rand_ind]
labels = labels[rand_ind]
#to tensor
features = torch.tensor(features)
labels = torch.tensor(labels)

# proj_layer = nn.Sequential(nn.Linear(1024,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,256)).cuda()
# proj_layer2 = nn.Sequential(nn.Linear(256,512), nn.ReLU(), nn.Linear(512,512), nn.ReLU(), nn.Linear(512,1024)).cuda()
proj_layer = DropMLP(1024, 256, 256, 3).cuda()
proj_layer2 = DropMLP(256, 256, 1024, 3).cuda()

# proj_layer = nn.Linear(1024, 256).cuda()
# proj_layer2 = nn.Linear(256, 1024).cuda()
train_params = itertools.chain(proj_layer.parameters(), proj_layer2.parameters())
lr = 1e-5
optim = torch.optim.AdamW(params=train_params, lr = lr, weight_decay = 1e-5)
features = features.cuda()

train_feats = features[:train_num]
val_feats = features[train_num:]
train_labels = labels[:train_num]
val_labels = labels[train_num:]

for i in range(iterations):
    proj_feats = proj_layer(train_feats)
    rev_feats = proj_layer2(proj_feats)
    cos_sim = compute_sim(proj_feats, train_labels, 'cos')
    cka_sim = cka_similarity(proj_feats, train_feats)
    reconstrut_loss = F.mse_loss(rev_feats, train_feats)

    loss = 2 - cos_sim  - cka_sim + reconstrut_loss

    optim.zero_grad()
    loss.backward()
    optim.step()
    if ((i+1)%100==0):
        print(f"{i}/{iterations} train set cos:{float(cos_sim)} cka:{float(cka_sim)} mse:{reconstrut_loss}" )
        #compute valid loss
        with torch.no_grad():
            proj_feats = proj_layer(features[train_num:])
            import pdb;pdb.set_trace()
            cos_sim = compute_sim(train_feats, labels[train_num:], 'cos')

            cka_sim = cka_similarity(proj_feats, features[train_num:])
            rev_feats = proj_layer2(proj_feats)
            mse_error = F.mse_loss(rev_feats, val_feats)
        print(f"{i}/{iterations} val set  cos:{float(cos_sim)} cka:{float(cka_sim)} mse:{mse_error} std:{proj_feats.std(dim=0).mean()}" )
torch.save(proj_layer.state_dict(), "proj.pth")