if [ "$1" == "local" ]; then
    SOURCE="/home/bvdc1/thanhtt/repos/Kaggle/hubmap_hv/InternImage/detection/modified"
    IMAGES="/media/bvdc1/data1/thanhtt_data/Kaggle/competitions/hubmap_hv"
    OUTPUT="/media/bvdc1/data1/thanhtt_data/Kaggle/competitions/hubmap_hv/outputs/internimage"
    CONFIG="/home/bvdc1/thanhtt/repos/Kaggle/hubmap_hv/InternImage/detection/configs"
    DOCKER_IMAGE="mmdet:0.1"
elif [ "$1" == "dgx3" ]; then
    SOURCE="/u01/data/thanhtt/repos/Kaggle/hubmap_hv/InternImage/detection/modified"
    IMAGES="/u01/data/thanhtt/data/kaggle/hubmap_hv"
    OUTPUT="/u01/data/thanhtt/data/kaggle/hubmap_hv/outputs/internimage"
    CONFIG="/u01/data/thanhtt/repos/Kaggle/hubmap_hv/InternImage/detection/configs"
    DOCKER_IMAGE="intern:0.1"

else
    echo "Unknown server"
    exit
fi

# docker build -f docker/Dockerfile -t intern:0.1 .
# dataset_dir = /usr/src/hubmap_hv/repos/yolov8_yaml_v0.0.1/dataset.yaml
# update code manuall: cp -r /usr/src/yolov8/ultralytics ./ultralytics
# test image: /usr/src/hubmap_hv/hubmap-hacking-the-human-vasculature/test/72e40acccadf.tif
# eval:         yolo segment val data=/usr/src/hubmap_hv/repos/yolov8_yaml_v0.0.1/dataset.yaml model=runs/segment/train/weights/best.pt conf=0.25 iou=0.7
# train:        yolo segment train model=yolov8n-seg.pt data=/usr/src/hubmap_hv/repos/yolov8_yaml_v0.0.1/dataset.yaml epochs=50 imgsz=1024 project=runs/hubmap_hhv name=v8n_3cls_dataset1_imgz1024 exist_ok=True batch=16 workers=4

# sudo docker run --gpus all -it \
#     --shm-size=8gb --env="DISPLAY" --net=host \
#     # --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     # --mount type=bind,source=$SOURCE,target=/usr/src/ultralytics/ultralytics \
#     # --mount type=bind,source=$IMAGES,target=/usr/src/hubmap_hv \
#     # --mount type=bind,source=$OUTPUT,target=/usr/src/ultralytics/runs \
#     # --volume=$HOME/.torch/fvcore_cache:/tmp:rw \
#     --name=hubmap_hhv_mmdet_0.1 $DOCKER_IMAGE

docker run -it --gpus all\
    --shm-size=8gb --env="DISPLAY" --network=host \
    --mount type=bind,source=$SOURCE,target=/internimage/detection/modified \
    --mount type=bind,source=$IMAGES,target=/internimage/detection/hubmap_hv \
    --mount type=bind,source=$OUTPUT,target=/internimage/detection/outputs \
    --mount type=bind,source=$CONFIG,target=/internimage/detection/configs \
    --name=hubmap_hhv_intern_0.1 $DOCKER_IMAGE

## When container is on, run following additional commands to compile CUDA operators:
# cd ./ops_dcnv3
# sh ./make.sh
## unit test (should see all checking is True)
# python test.py