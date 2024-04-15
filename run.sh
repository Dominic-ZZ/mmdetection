# python tools/train.py /workspace/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py

# 训练
python tools/train.py /workspace/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py

# test
# python tools/test.py /workspace/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py /workspace/mmdetection/work_dirs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco/best_coco_bbox_mAP_epoch_27.pth --show-dir outputs

# 检测视频
# python demo/video_demo.py rabbit.mp4 /workspace/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py\
#  /workspace/mmdetection/work_dirs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco/best_coco_bbox_mAP_epoch_27.pth \
#  --out /workspace/mmdetection/outputs/out.mp4 --device 'cuda:0'