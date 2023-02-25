# lr
lr0: 0.0001
warmup_lr: 0.00001   
warm_epoch:1


# setting
num_classes: 2  # 绝缘子串 II02为标准串，others为比较模糊的串
# num_classes: 15  # [SAI-KEY] 指定类别数量

# training
epochs: 1000
batch_size: 4
save_interval: 20
test_interval: 1
