# run training
#CUDA_VISIBLE_DEVICES=1 python train_coco.py -Ddavis /data/sdg/tracking_datasets/DAVIS/ -Dcoco /data/sdb/coco_2017 \
#-backbone resnet50 -save coco_weights/ -log_iter 500

# debug data
python dataset/coco.py -Ddavis /data/sdg/tracking_datasets/DAVIS/ -Dcoco /data/sdb/coco_2017 -o data_examples