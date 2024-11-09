import json
import os
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import mask_to_rle_pytorch, coco_encode_rle
from PIL import Image
from tqdm import tqdm 
import torch
from pycocotools import mask as mask_util
# Initialize SAM predictor
model_type = 'vit_l'  # You can choose 'vit_b', 'vit_h', or 'vit_l'
checkpoint_path = '/gpfsdata/home/caizhi/CrowdSAM/weights/sam_vit_l_0b3195.pth'  # Update with the path to your SAM checkpoint
sam = sam_model_registry[model_type](checkpoint=checkpoint_path).cuda()
predictor = SamPredictor(sam)

# Load dataset annotations
dataset_root = '/gpfsdata/home/caizhi/detrex/datasets/crossdomain_urban'
dataset_annotation = './annotations/daytime_train.json'

with open(os.path.join(dataset_root,dataset_annotation), 'r') as f:
    coco_data = json.load(f)

# Iterate over each image in the dataset
for img_info in tqdm(coco_data['images']):
    image_id = img_info['id']
    img_path = os.path.join(dataset_root, 'daytime_clear', img_info['file_name'])

    # Load the image
    image = np.array(Image.open(img_path))
    predictor.set_image(image)

    # Get all annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    for ann in annotations:
        # Get the bounding box (bbox is in [x, y, width, height] format)
        bbox = ann['bbox']
        input_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # Convert to [x1, y1, x2, y2]

        # Use SAM to predict the mask for the bounding box
        masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
        masks = torch.tensor(masks)
        rle = mask_to_rle_pytorch(masks)
        rle = coco_encode_rle(rle[0])
        # Use pycocotools to encode the binary mask into RLE format
        # rle = mask_util.encode(np.asfortranarray(masks))
        # Update the annotation with the segmentation mask
        ann['segmentation'] = rle #masks[0].tolist()

# Save the updated annotations to a new JSON file
output_path = './annotations/daytime_train_with_sam_masks.json'
with open(os.path.join(dataset_root,output_path), 'w') as f:
    json.dump(coco_data, f)

print(f"Updated annotations saved to {output_path}")
