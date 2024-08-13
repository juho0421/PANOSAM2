import numpy as np
import torch
import cv2
import os

# 현재 디렉토리 경로 확인
current_dir = os.getcwd()
print(current_dir)


from SAM.segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# SAM 모델 로드 및 초기화
sam = sam_model_registry["default"](checkpoint="SAM/checkpoint/sam_vit_h_4b8939.pth")
# mask_generator = SamAutomaticMaskGenerator(sam)
mask_generator_2 = SamAutomaticMaskGenerator(
    model = sam,
    points_per_side = 64,  # 이미지 한 면에 찍을 점(N*N 만큼)의 수
    points_per_batch = 64,  # gpu로 한 번에 수행할 이미지 수
    pred_iou_thresh = 0.85,  # iou = 교집합/합집합, 예측된 마스크가 실제 객체와 얼마나 일치하는지
    stability_score_thresh = 0.96,  # 마스크 필터링 안정성, 마스크 예측을 이진화하는 임계값을 조절하는 데 사용(높을수록 안정적이나 마스크가 많이 걸러짐)
    stability_score_offset = 1,   # 이진화 임계값을 조정하여 안정성 점수를 계산하는 과정에서 임계값을 이동시키는 데 사용
    box_nms_thresh = 0.7,   # Box의 IoU로 겹치는 객체 제거
    crop_n_layers = 0,  # 이미지의 크롭에서 다시 마스크 예측을 실행할 것인지, 더 정확한 마스크 예측이 가능(각 레이어는 2^(i_layer) 개의 이미지 크롭을 가짐)
    crop_nms_thresh = 0.7,  # 서로 다른 크롭 간에 중복된 마스크를 필터링
    crop_overlap_ratio = 512/1024,  # 크롭들이 서로 겹치는 정도
    crop_n_points_downscale_factor = 1, # n 번째 layer에서 sampling되는 한 변의 점의 수가 (crop_n_points_downscale_factor)^n으로 축소
    point_grids = None,    # 이미지에서 샘플링에 사용되는 명시적인 그리드의 목록(0~1), n번째 crop layer에서 n번째 grid가 사용됨
    min_mask_region_area = 0,   # 마스크 내부의 연결되지 않은 영역 및 구멍의 크기가 "min_mask_region_area"보다 작은 경우에 대해 후처리가 적용
    output_mode = "binary_mask",
)

def save_anns(anns, output_path):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True) # 영역 크기순으로 정렬
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3), dtype=np.uint8) * 255

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.randint(0, 255, size=3)
        img[m] = color_mask
    cv2.imwrite(output_path, img)


# GPU 사용 가능 여부
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
sam.to(device)

# data 경로 지정
data_dir = r"C:\Users\Lab_ICT\PycharmProjects\CCC\datasets"

### 상세 data 경로 지정 ###
origin_image_dir = os.path.join(data_dir, "image5") # 원본 이미지 경로
SAM_mask_dir = os.path.join(data_dir, "image5_SAM_mask") # 원본 이미지 SAM 결과 이미지 경로
# SAM_mask_info_dir = os.path.join(data_dir, "image100_SAM_mask_info")

# 원본 이미지 폴더 내의 모든 이미지 파일에 대해 마스크 생성 및 저장
for filename in os.listdir(origin_image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):

        # 이미지 파일의 경로 생성
        image = cv2.imread(os.path.join(origin_image_dir, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 파일 경로을 통해 마스크 생성
        masks = mask_generator_2.generate(image)
        print(len(masks))

        save_anns(masks, os.path.join(SAM_mask_dir, filename))
        print("Mask_img saved as:", filename)

