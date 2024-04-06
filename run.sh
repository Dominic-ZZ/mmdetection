# python tools/train.py /workspace/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py

# 训练
# python tools/train.py /workspace/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py
# python tools/test.py /workspace/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py /workspace/mmdetection/work_dirs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco/epoch_15.pth --show-dir outputs

# 检测视频
python demo/video_demo.py test1.mp4 /workspace/mmdetection/projects/EfficientDet/configs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco.py /workspace/mmdetection/work_dirs/efficientdet_effb0_bifpn_8xb16-crop512-300e_coco/epoch_15.pth --out /workspace/mmdetection/outputs/out.mp4 --device 'cuda:0'