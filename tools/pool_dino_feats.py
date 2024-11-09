import numpy as np
import torch
import torch.nn as nn
import os
import tqdm
data_root = './datasets/crossdomain_urban/daytime_clear/dino_feats'
import torch.nn
proj_feats = True
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

if proj_feats:
    proj_layer = DropMLP(1024, 256, 256, 3).cuda()
    proj_layer.eval()
    # proj_layer =  torch.nn.Linear(1024, 256).cuda()
    proj_layer.load_state_dict(torch.load('proj.pth'))
    post_fix = 'proj'
else:
    post_fix = 'pool'
for i in  tqdm.tqdm(os.listdir(data_root)):
    output_path = os.path.join( data_root, f'{i[:-4]}_{post_fix}.npy')
    # if os.path.exists(output_path):
    #     continue
    if '.npy' in i and 'pool' not in i and 'proj' not in i:
        feats_path = os.path.join(data_root,  i)
        x = np.load(feats_path)
        x = torch.tensor(x.astype(np.float32))
        x = x.mean(dim=[1,2])#.numpy()
        if proj_feats:
            with torch.no_grad():
                x = proj_layer(x.cuda())
        x = x.cpu().numpy()
        np.save(output_path, x)
    # dino_feats.append(x)