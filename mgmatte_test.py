# Benjamin Axline - Mask Refinement - MG Matte
# 9 July 2024


# import
import os
import cv2
import toml
import argparse
import numpy as np

import torch
from torch.nn import functional as F

import utils
from   utils import CONFIG
import networks
from infer import single_inference, generator_tensor_dict

# configure device
device = 'cuda' #if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load model
model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
# print(model)
model.to(device)

print('Torch Version: ', torch.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='MGMatting-main/code-base/config/MGMatting-DIM-100k.toml')
parser.add_argument('--checkpoint', type=str, default='/teamspace/studios/this_studio/MGMatting-main/code-base/latest_model.pth', # 'checkpoints/MGMatting-DIM-100k/latest_model.pth',
                    help="path of checkpoint")
parser.add_argument('--image-dir', type=str, default='/teamspace/studios/this_studio/MGMatting-main/code-base/image', help="input image dir")
parser.add_argument('--mask-dir', type=str, default='/teamspace/studios/this_studio/MGMatting-main/code-base/masks', help="input mask dir")
parser.add_argument('--image-ext', type=str, default='.jpg', help="input image ext")
parser.add_argument('--mask-ext', type=str, default='.png', help="input mask ext")
parser.add_argument('--output', type=str, default='results/', help="output dir")
parser.add_argument('--guidance-thres', type=int, default=128, help="guidance input threshold")
parser.add_argument('--post-process', action='store_true', default=False, help='post process to keep the largest connected component')

# Parse configuration
args = parser.parse_args()
with open(args.config) as f:
    utils.load_config(toml.load(f))

# load checkpoint
# checkpoint = torch.load("MGMatting-main/code-base/latest_model_DIM.pth")
checkpoint = torch.load("MGMatting-main/code-base/latest_model_RWP.pth")
model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

# inference
model = model.eval()
torch.save(model.state_dict(), f="mgmatte_model.pth")

# print(model.type)
# single photo
image_name = "nature-7" # no extension!!
extension = 'jpg'
image_path = f"MGMatting-main/code-base/data/image/nature/{image_name}.{extension}"
mask_path = f"MGMatting-main/code-base/data/masks/nature/{image_name}.png"
print(f"Image: {image_path}, Mask: {mask_path}")
# img = cv2.imread(image_path)
# msk = cv2.imread(mask_path, 0)
# cv2.imwrite('img1.jpg', img)
# cv2.imwrite('img1.png', msk)

image_dict = generator_tensor_dict(image_path, mask_path, args)
print("Image dict keys:", image_dict.keys())

alpha_pred = single_inference(model, image_dict, post_process=args.post_process)
print("Alpha prediction shape: ", alpha_pred.shape)
# with torch.no_grad():
#     image, mask = image_dict["image"], image_dict["mask"]
#     image.to(device=device)
#     mask.to(device=device)

#     prediction = model(image, mask)
#     print(prediction.type)


# pic1 = pred1[0][0]
# pic2 = pred2[0][0]
# pic3 = pred3[0][0]

# cv2.imwrite("pic1.png", pic1.cpu().numpy()*255)
# cv2.imwrite("pic2.png", pic2.cpu().numpy()*255)
# cv2.imwrite("pic3.png", pic3.cpu().numpy()*255)

cv2.imwrite(f"MGMatting-main/code-base/data/refined/nature/{image_name}.png", alpha_pred)
print(f"Mask saved successfully as {image_name}.png")

"""
# for a whole directory
for image_name in os.listdir(args.image_dir):
    # assume image and mask have the same file name
    image_path = os.path.join(args.image_dir, image_name)
    mask_path = os.path.join(args.mask_dir, image_name.replace(args.image_ext, args.mask_ext))
    print('Image: ', image_path, ' Mask: ', mask_path)
    image_dict = generator_tensor_dict(image_path, mask_path, args)

    alpha_pred = single_inference(model, image_dict, post_process=args.post_process)

    cv2.imwrite(os.path.join(args.output, image_name), alpha_pred)
"""


