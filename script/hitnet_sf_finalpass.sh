python3 train.py fit -c script/hitnet_sf_finalpass.yaml
# python3 train.py \
# --log_dir logs \
# --exp_name hitnet_sf_finalpass \
# --model HITNet_SF \
# --gpus -1 \
# --max_steps 2000000 \
# --accelerator ddp \
# --max_disp 320 \
# --max_disp_val 192 \
# --optmizer Adam \
# --lr 4e-4 \
# --lr_decay 1000000 0.25 1300000 0.1 1400000 0.025 \
# --lr_decay_type Lambda \
# --batch_size 8 \
# --batch_size_val 8 \
# --num_workers 2 \
# --num_workers_val 2 \
# --data_augmentation 0 \
# --data_type_train SceneFlow \
# --data_root_train /home/tiger/SceneFlow \
# --data_list_train lists/sceneflow_train_fly3d_only.list \
# --data_size_train 960 320 \
# --data_type_val SceneFlow \
# --data_root_val /home/tiger/SceneFlow \
# --data_list_val lists/sceneflow_test.list \
# --data_size_val 960 540 \
# --robust_loss_a 0.9 \
# --robust_loss_c 0.1
