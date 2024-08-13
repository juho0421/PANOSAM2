import os
from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import supervision as sv
from tqdm import tqdm
import glob
import argparse
from imageio.v2 import imread, imwrite
from PanoPlane360 import utils
import cv2
import subprocess


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# 현재 디렉토리 경로 확인
current_dir = os.getcwd()
print(current_dir)


### Part 0. 환경 조성 ###

# 대상 이미지
img_folder_path = r"C:\Users\Lab_ICT\PycharmProjects\PANO+SAM\datasets\image5"

img_dir = []
for img_name in os.listdir(img_folder_path):
  if img_name.endswith(('.jpg', '.jpeg', '.png')):
    img_dir.append(os.path.join(img_folder_path,img_name))
print('%d images in total.' % len(img_dir))

sam_checkpoint = "./SAM/checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=64,    # 이미지 한 면에 찍을 점(N*N 만큼)의 수 # points_per_side = 64 # points_per_batch = 32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.85,    # 마스크 필터링 안정성, 마스크 예측을 이진화하는 임계값을 조절하는 데 사용(높을수록 안정적이나 마스크가 많이 걸러짐
    min_mask_region_area=1,)


### Part 1. SAM ###

## mask 정의 ##
SAM_masks = []
annotated_images = []
images = []

## 이미지 전처리 함수 ##
def mask_to_image(mask):
    # True를 255(흰색), False를 0(검은색)으로 변환하여 흑백 이미지 생성
    mask_image = np.float32(mask) * 255
    mask_image = np.expand_dims(mask_image, axis=-1)  # 이미지 차원 확장 (H, W) -> (H, W, 1)
    return mask_image

def image_resize(img):
    crop_black = 80/512
    crop = int(crop_black * img.shape[0])
    crop_img = img[crop:-crop]
    return crop_img

## SAM 적용 ##
for img_path in tqdm(img_dir, desc='SAM', total=len(img_dir),leave=True):
    # Part 1-1. 이미지 전처리
    img_name = os.path.basename(img_path)[:-4]
    bgr_image = cv2.imread(img_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized_image = image_resize(rgb_image)
    image = resized_image
    images.append(image)

    # Part 1-2 SAM 적용
    sam_result = mask_generator.generate(image)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_images.append(annotated_image)
    cv2.imwrite('./datasets/image5_SAM_mask/' + f"{img_name}_SAM_mask.png", annotated_image)

    # Part 1-3 SAM mask 추가
    masks = [
        mask['segmentation']
        for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)
    ]
    for idx, mask in enumerate(masks):
      mask_image = mask_to_image(mask)
      cv2.imwrite('/content/drive/MyDrive/Colab Notebooks/CCC/datasets/target_image_object_mask/' + f"{img_name}_object_mask_{idx}.png", mask_image)

    SAM_masks.append(masks)


### Part 2. Panoplane360 적용 ###

# Arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--glob', help='Input mode 1: path to input images')
parser.add_argument('--txt', help='Input mode 2: path to image name txt file')
parser.add_argument('--root', help='Input mode 2: path to input images root')
parser.add_argument('--pth', help='path to dumped .pth file')
parser.add_argument('--outdir')
parser.add_argument('--device', default='cuda')
parser.add_argument('--rgb_mean', default=[123.675, 116.28, 103.53], type=float, nargs=3) # for normalization
parser.add_argument('--rgb_std', default=[58.395, 57.12, 57.375], type=float, nargs=3) # for normalization
parser.add_argument('--base_height', default=512, type=int)
parser.add_argument('--crop_black', default=80/512, type=float)
# args = parser.parse_args()
args, unknown = parser.parse_known_args()
args.pth = r"C:\Users\Lab_ICT\PycharmProjects\PANO+SAM\PanoPlane360\ckpt\mp3d.pth"
args.outdir = r"C:\Users\Lab_ICT\PycharmProjects\PANO+SAM\datasets\image5_Pano_mask"

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

# Load trained checkpoint
print('Loading checkpoint...', end='', flush=True)
net, args_model = utils.load_trained_model(args.pth)
net = net.eval().to(args.device) # trained model structure
print('done')

# Inference on all images
PANO_masks ={"h_planes":[], "v_planes":[]}
for img_path in tqdm(img_dir, desc='PanoPlane360', total=len(img_dir),leave=True):
    k = os.path.split(img_path)[1][:-4]
    rgb_np = imread(img_path)[..., :3]

    with torch.no_grad():
        # Prepare 1,3,H,W input tensor
        input_dict = {
            'rgb': torch.from_numpy(rgb_np.transpose(2, 0, 1)[None].astype(np.float32)), # pytorch -> (c, h, w)
        }
        input_dict = utils.preprocess(input_dict, args)  # Normalize and cropping
        input_dict['filename'] = k

        # Call network interface for estimated HV map
        infer_dict = net.infer_HVmap(input_dict, args)

    # Dump results
    for name, v in infer_dict.items():
        if name == 'h_planes':
            imwrite(os.path.join(args.outdir, k + '_h_planes.exr'), v)
            h_planes = v.copy()
            PANO_masks["h_planes"].append(h_planes)
            # v_uni = np.unique(v)

            # v_cp = v.copy()
            # for i in v_uni:
            #     v_cp[v == i] = np.random.randint(0, 255, size=1)
            # v_cp = v_cp.astype(np.int8) # 데이터 type 변경
            # cv2.imwrite(os.path.join(args.outdir, k + '_h_planes.png'), v_cp)

        elif name == 'v_planes':
            imwrite(os.path.join(args.outdir, k + '_v_planes.exr'), v)
            v_planes = v.copy()
            PANO_masks["v_planes"].append(v_planes)
            # v_uni = np.unique(v)

            # v_cp = v.copy()
            # for i in v_uni:
            #     v_cp[v == i] = np.random.randint(0, 255, size=1)
            # v_cp = v_cp.astype(np.int8)
            # cv2.imwrite(os.path.join(args.outdir, k + '_v_planes.png'), v_cp)


### Part 3. 라벨 할당 ###

## 라벨 할당함수 ##
# targets: SAM_masks[i]
# query: PANO_masks["h_planes"][i]
def label_predict_h(targets, query):
    Enhanced_mask = np.zeros_like(query, dtype=np.float32) # (352,1024)

    for idx, mask in enumerate(targets):
        labels, label_counts = np.unique(query[mask], return_counts=True)
        max_label_idx = np.argmax(label_counts)
        allocated_label = labels[max_label_idx]
        if label_counts[max_label_idx] >= np.sum(label_counts) / 2:
            # 전체 label_counts의 절반 이상을 가지는 경우 해당 레이블 할당
            Enhanced_mask[mask] = allocated_label
        else:
            # 절반 이상을 가지지 못하는 경우 특정 값(예: 0)으로 할당
            Enhanced_mask[mask] = 0
    return Enhanced_mask

# targets: SAM_masks[i]
# query: PANO_masks["v_planes"][i]
def label_predict_v(targets, query):
    Enhanced_mask = np.zeros_like(query, dtype=np.float32) # (352,1024,3)

    for idx, mask in enumerate(targets):
        labels, label_counts = np.unique(query[mask], return_counts=True, axis=0)
        max_label_idx = np.argmax(label_counts)
        allocated_label = labels[max_label_idx]
        if label_counts[max_label_idx] >= np.sum(label_counts) / 2:
            # 전체 label_counts의 절반 이상을 가지는 경우 해당 레이블 할당
            Enhanced_mask[mask] = allocated_label
        else:
            # 절반 이상을 가지지 못하는 경우 특정 값(예: 0)으로 할당
            Enhanced_mask[mask] = 0
    return Enhanced_mask

# 라벨 할당
# Example of adding debugging print statements
Enhanced_pano_masks = {"h_planes":[], "v_planes":[]}

for i in range(len(img_dir)):
    Enhanced_pano_mask_h = label_predict_h(SAM_masks[i], PANO_masks["h_planes"][i])
    Enhanced_pano_masks["h_planes"].append(Enhanced_pano_mask_h)
    Enhanced_pano_mask_v = label_predict_v(SAM_masks[i], PANO_masks["v_planes"][i])
    Enhanced_pano_masks["v_planes"].append(Enhanced_pano_mask_v)

for img_path in tqdm(img_dir, desc='Enhancing', total=len(img_dir), leave=True):
    k = os.path.split(img_path)[1][:-4]
    i = img_dir.index(img_path)
    imwrite(os.path.join(args.outdir, k + '_h_planes.exr'), Enhanced_pano_masks["h_planes"][i], format='exr')
    imwrite(os.path.join(args.outdir, k + '_v_planes.exr'), Enhanced_pano_masks["v_planes"][i], format='exr')


### Part 4. 3D 모델링 ###
data_dir = r"C:\Users\Lab_ICT\PycharmProjects\PANO+SAM\datasets"
origin_image_dir = os.path.join(data_dir, "image5") # 원본 이미지 경로
pano_mask_dir = os.path.join(data_dir, "image5_Pano_mask") # 원본 이미지 h,v 마스크 경로
pano_3d_dir = os.path.join(data_dir, "image5_3D") # 원본 모델 경로

# h,v 평면을 통한 3d 모델링 (To always visualize all the planes, add --mesh_show_back_face.)
def make_3d_model(origin_image_dir, pano_mask_dir, pano_3d_dir, image_name):
    subprocess.run(["python",
                    "PanoPlane360/vis_planes.py",
                    "--img", os.path.join(origin_image_dir, image_name),
                    "--h_planes", os.path.join(pano_mask_dir, image_name).replace(".jpg", "")+"_h_planes.exr",
                    "--v_planes", os.path.join(pano_mask_dir, image_name).replace(".jpg", "")+"_v_planes.exr",
                    "--mesh",            # mesh -> TriangleMesh / mesh_show_back_face -> PointCloud
                    "--save_path", pano_3d_dir])
    return

# 적용하고자 하는 이미지 지정
for image_name in tqdm(os.listdir(origin_image_dir), desc='Modeling', total=len(img_dir),leave=True):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        make_3d_model(origin_image_dir, pano_mask_dir, pano_3d_dir, image_name)