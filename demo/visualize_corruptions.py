from detrex.data.transforms import ACVCGenerator
import cv2
from matplotlib import pyplot as plt
from PIL import Image

acvc = ACVCGenerator()
corruptions = [
    ["defocus_blur",
    "glass_blur",
    "gaussian_blur",
    "motion_blur"],
    ["speckle_noise",
    "shot_noise",
    "impulse_noise",
    "gaussian_noise"],
    ["jpeg_compression",
    "pixelate",
    "elastic_transform",
    "brightness"],
    ["saturate",
    "contrast",
    "high_pass_filter",
    "phase_scaling"]
]


def apply_corrupt(img, corrupt):
    if corrupt == []:
        return Image.fromarray(img)
    s = 3  # 可以根据需要随机化
    aug_img = img
    import pdb;pdb.set_trace()
    for c in corrupt:
        aug_img = acvc.apply_corruption(img, c, s)
    return aug_img.convert("RGB")

demo_path = "test_imgs/dc_0.jpg"
demo_img = cv2.imread(demo_path)
demo_img_rgb = cv2.cvtColor(demo_img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

plt.figure(figsize=(15, 10))  # 调整输出图像尺寸

for k, corruption in enumerate(corruptions):
    aug_img = apply_corrupt((demo_img_rgb), corruption)  # 确保转换为PIL格式
    plt.subplot(4, 4, k + 1)  # 调整子图布局为4x4
    plt.imshow(aug_img)
    plt.title(corruption)
    plt.axis('off')  # 隐藏坐标轴

plt.tight_layout()  # 自动调整布局
plt.savefig('vis_out/aug_demo.jpg')
plt.show()
