import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.archs.codeformer_arch import CodeFormer

# Argument parser to choose the enhancement option
parser = argparse.ArgumentParser(description='CodeFormer Face Restoration')
parser.add_argument('--enhance_option', type=int, default=1, choices=[1, 2], help='Enhancement level: 1 for low, 2 for high.')
args = parser.parse_args()

# Load the pre-trained model for CodeFormer
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()

# ------------------------ Input & Output ------------------------
detection_model = 'retinaface_resnet50'
w = 0.7  # Balance between restoration quality and fidelity
upscale = 2  # Upscaling factor for images
img_name = f"/content/makeup.jpg"
result_root = f'results/test_img_{w}'

# Create the result directory
if not os.path.exists(result_root):
    os.makedirs(result_root)

# ------------------ Set up CodeFormer restorer -------------------
net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                 connect_list=['32', '64', '128', '256']).to(device)

ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                               model_dir='weights/CodeFormer', progress=True, file_name=None)
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

# ------------------ Set up FaceRestoreHelper -------------------
face_helper = FaceRestoreHelper(
    upscale,
    face_size=512,
    crop_ratio=(1, 1),
    det_model=detection_model,
    save_ext='png',
    use_parse=True,
    device=device
)

# Load the input image and detect faces
image = face_recognition.load_image_file(img_name)
pil_image = Image.fromarray(image)
opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
face_helper.read_image(opencvImage)

# Get face landmarks and align faces
num_det_faces = face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
print(f'Detected {num_det_faces} faces')
face_helper.align_warp_face()

# Restore faces with CodeFormer
for idx, cropped_face in enumerate(face_helper.cropped_faces):
    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            output = net(cropped_face_t, w=w, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        del output
        torch.cuda.empty_cache()
    except Exception as error:
        print(f'Failed inference for CodeFormer: {error}')
        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

    restored_face = restored_face.astype('uint8')
    face_helper.add_restored_face(restored_face, cropped_face)

# Enhance the restored faces
if args.enhance_option == 1:
    # Low enhancement option
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
elif args.enhance_option == 2:
    # High enhancement option
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    brightness = 4
    contrast = 1.2
    restored_img = cv2.addWeighted(restored_img, contrast, np.zeros(restored_img.shape, restored_img.dtype), 0, brightness)

# Save the restored image
name = os.path.basename(img_name)
save_restore_path = os.path.join(result_root, f'restored_{name}')
imwrite(restored_img, save_restore_path)

print(f'All results are saved in {result_root}')
