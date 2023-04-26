import os
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils.load_data import load_data, create_image_cube, split_data
from models.cnn2d import cnn2d
from tensorflow.keras.utils import to_categorical
from keras.callbacks import LambdaCallback, ModelCheckpoint
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback
import logging
import psutil


def main(args):
    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1 数据预处理
    # 加载数据集
    data, label, num_class = load_data(args.dataset)
    # 构建数据立方体
    data, label = create_image_cube(data, label, window_size=args.spatial_size, remove_zero_labels = True)
    # 将标签转换成 one-hot 编码
    label = to_categorical(label)
    # 划分训练集和测试集
    test_percent = 1 - args.train_percent
    train_data, test_data, train_label, test_label = split_data(data, label,
                                                                test_percent=test_percent,
                                                                random_state=args.random_state)
    if args.use_val:
        test_data, val_data, test_label, val_label = split_data(test_data, test_label,
                                                                test_percent=args.val_percent,
                                                                random_state=args.random_state)

    # 输出训练集和测试集的形状和训练设置
    logging.info(f"Train data shape: {train_data.shape}")
    logging.info(f"Train label shape: {train_label.shape}")
    logging.info(f"Test data shape: {test_data.shape}")
    logging.info(f"Test label shape: {test_label.shape}")
    logging.info(f"Training settings: {args}")

    #2 模型初始化
    # 初始化wandb

    run_name = f"{args.dataset}_{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    experiment = wandb.init(project='LAND_CLASSIFICATION', resume='allow', anonymous='never', name=run_name)
    experiment.config.update(dict(learning_rate=args.learning_rate, num_epochs=args.num_epochs,
                                  batch_size=args.batch_size, test_percent=test_percent,
                                  train_percent=args.train_percent, val_percent=args.val_percent / args.train_percent,
                                  spatial_size=args.spatial_size, dataset=args.dataset))
    save_path = os.path.join(args.checkpoint_dir, run_name)

    
    # 获取模型
    input_shape = train_data.shape[1:]
    model = cnn2d(input_shape, num_class)

    #3 训练
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{run_name}_{{epoch:02d}}.h5")

    # 每个 epoch 保存一次模型
    train_data_batches = len(train_data) // args.batch_size
    save_interval_batches = args.save_interval * train_data_batches

    # 保存模型的回调函数
    model_checkpoint = ModelCheckpoint(filepath=args.checkpoint_dir, save_freq=save_interval_batches, save_weights_only=False, verbose=0)
    model_checkpoint = ModelCheckpoint(os.path.join(args.checkpoint_dir, f"best_model.h5"), monitor='val_accuracy', verbose=0, save_best_only=True)

    # 模型验证集
    val_data, val_label = (val_data, val_label) if args.use_val else (test_data, test_label)

    # 训练模型
    history = model.fit(train_data, train_label, 
                        batch_size=args.batch_size, 
                        epochs=args.num_epochs,
                        verbose=1, 
                        validation_data=(val_data, val_label),
                        callbacks=[model_checkpoint, WandbCallback()])


    # 在测试集上评估模型
    test_loss, test_acc = model.evaluate(test_data, test_label)
    logging.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # 保存模型
    

    # 结束wandb
    experiment.finish()


if __name__ == '__main__':
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--display_step', type=int, default=10, help='显示间隔')
    parser.add_argument('--dataset', type=str, default='indian_pines', choices=['indian_pines', 'salinas'], help='选择要使用的数据集')

    parser.add_argument('--train_percent', default=0.15, type=float, help='训练集比例')
    parser.add_argument('--use_val', action='store_true', default=True, help='使用验证集')
    parser.add_argument('--val_percent', default=0.1, type=float, help='验证集比例(验证集从测试集中划分)')

    parser.add_argument('--save_interval', type=int, default=10, help='保存模型的间隔')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='保存模型的目录')
    parser.add_argument('--random_state', type=int, default=42, help='随机数种子')
    parser.add_argument('--spatial_size', default=19, type=int, help='构建数据立方体时采用的窗口大小')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='使用 GPU 训练')
    parser.add_argument('--log_dir', type=str, default='./logs', help='保存日志的目录')

    args = parser.parse_args()

    main(args)