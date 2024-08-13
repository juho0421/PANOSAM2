import os
import subprocess

# 현재 디렉토리 경로 확인
current_dir = os.getcwd()
print(current_dir)

# data 경로 지정
data_dir = r"C:\Users\Lab_ICT\PycharmProjects\PANO+SAM\datasets"

origin_image_dir = os.path.join(data_dir, "image5") # 원본 이미지 경로
pano_mask_dir = os.path.join(data_dir, "image5_Pano_mask") # 원본 이미지 h,v 마스크 경로
pano_3d_dir = os.path.join(data_dir, "image5_Pano_3D") # 원본 모델 경로


# h,v 평면을 통해 3d 모델링 (To always visualize all the planes, add --mesh_show_back_face.)
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
for image_name in os.listdir(origin_image_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        print(image_name)
        # PART1. H, V Segmentation
        plane_inference = subprocess.run(["python",
                                          "PanoPlane360/inference.py",
                                          "--pth", "PanoPlane360/ckpt/mp3d.pth",
                                          "--glob", os.path.join(origin_image_dir, image_name),
                                          "--outdir", pano_mask_dir])
        # PART2. 3D 모델링
        make_3d_model(origin_image_dir, pano_mask_dir, pano_3d_dir, image_name)