import mmcv
import numpy as np
from mmengine import load
import os

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import CameraInstance3DBoxes 

selected_cam = 'CAM_FRONT'
info_file = load('./data/nuscenes/filtered_output.pkl') 
cam2img = np.array(info_file['data_list'][0]['images'][selected_cam]['cam2img'], dtype=np.float32)
img_filename = info_file['data_list'][0]['images'][selected_cam]['img_path']
img_path = os.path.join('./data/nuscenes/samples', selected_cam, img_filename)
bboxes_3d = []
cnt = 0

for instance in info_file['data_list'][0]['cam_instances'][selected_cam]:
    cnt += 1
    if cnt > 1:
        break
    bboxes_3d.append(instance['bbox_3d'])
gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32)

gt_bboxes_3d = CameraInstance3DBoxes(  # still the box in nuScenes format (x forward, y left, z up)
    gt_bboxes_3d,
    box_dim=gt_bboxes_3d.shape[-1],
    origin=(0.5, 0.5, 0.5))

print('-----------')
print(gt_bboxes_3d)

input_meta = {'cam2img': cam2img}

visualizer = Det3DLocalVisualizer()

img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')
visualizer.set_image(img)
# project 3D bboxes to image
visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta)
visualizer.show()
