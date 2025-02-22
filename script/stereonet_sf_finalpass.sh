python3 train.py fit -c script/stereonet_sf_finalpass.yaml

# python3 train.py \
# fit \
# --log_dir logs \
# --exp_name stereonet_sf_cleanpass \
# --model StereoNet \
# --sync_batchnorm True \
# --gpus 1 \
# --max_steps 2000000 \
# --accelerator ddp \
# --max_disp 192 \
# --optmizer RMS \
# --lr 1e-3 \
# --lr_decay 14000 0.9 \
# --lr_decay_type Step \
# --batch_size 8 \
# --batch_size_val 8 \
# --num_workers 2 \
# --num_workers_val 2 \
# --data_type SceneFlow \
# --data_augmentation 0 \
# --data_root_train /mnt/howler/md0/data/flyingthings3d_source \
# --data_root_val /mnt/howler/md0/data/flyingthings3d_source \
# --data_list_train lists/sceneflow_train_fly3d_only.list \
# --data_list_val lists/sceneflow_test.list \
# --data_size_train 960 512 \
# --data_size_val 960 540 \
# --robust_loss_a 0.9 \
# --robust_loss_c 0.1