import os
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils.load_data import load_data, create_image_cube, split_data
from models.cnn2d import cnn2d
from models.GAN import generator, discriminator
from models.ACGAN import acgan_generator, acgan_discriminator
from tensorflow.keras.utils import to_categorical
from keras.callbacks import LambdaCallback, ModelCheckpoint
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback
import logging





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
    # experiment = wandb.init(project='GAN', resume='allow', anonymous='never', name=run_name)
    # experiment.config.update(dict(learning_rate=args.learning_rate, num_epochs=args.num_epochs,
    #                               batch_size=args.batch_size, test_percent=test_percent,
    #                               train_percent=args.train_percent, val_percent=args.val_percent / args.train_percent,
    #                               spatial_size=args.spatial_size, dataset=args.dataset))
    save_path = os.path.join(args.checkpoint_dir, run_name)

    
        # 获取模型
    input_shape = train_data.shape[1:]
    # print(input_shape)
    # exit()
    G = acgan_generator(input_shape)  # Create the generator model
    D = acgan_discriminator(input_shape, num_class) 

    # G.compile(loss="binary_crossentropy", optimizer="adam")
    # D.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    #3 训练
    
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{run_name}_{{epoch:02d}}.h5")

    # 每个 epoch 保存一次模型
    train_data_batches = len(train_data) // args.batch_size
    save_interval_batches = args.save_interval * train_data_batches

    #3 训练
    # Train the ACGAN
    for epoch in range(args.num_epochs):
        for batch in range(len(train_data) // args.batch_size):
            # Select a random batch of images
            idx = np.random.randint(0, train_data.shape[0], args.batch_size)
            real_images = train_data[idx]

            # Generate a batch of fake images
            noise = np.random.normal(0, 1, (args.batch_size, input_shape[0], input_shape[1], input_shape[2]))
            fake_images = G.predict(noise)

            # Combine real and fake images for training the discriminator
            print(fake_images.shape)
            print(real_images.shape)
            exit()
            combined_images = np.concatenate([real_images, fake_images])
            combined_labels = np.concatenate([train_label[idx], np.zeros((args.batch_size, num_class))])

            # Train the discriminator
            d_loss = D.train_on_batch(combined_images, combined_labels)

            # Train the generator
            noise = np.random.normal(0, 1, (args.batch_size, input_shape[0], input_shape[1], input_shape[2]))
            valid_labels = np.ones((args.batch_size, num_class))
            g_loss = D.train_on_batch(G.predict(noise), valid_labels)

            # Log the losses
            step = epoch * (len(train_data) // args.batch_size) + batch
            # experiment.log({"d_loss": d_loss, "g_loss": g_loss}, step=step)

    # 4. Test the ACGAN
    noise_test = np.random.normal(0, 1, (test_data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    generated_test_data = G.predict(noise_test)
    test_loss, test_acc = D.evaluate(generated_test_data, test_label)
    logging.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # 结束wandb
    # experiment.finish()

if __name__ == '__main__':
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--display_step', type=int, default=10, help='显示间隔')
    parser.add_argument('--dataset', type=str, default='indian_pines', choices=['indian_pines', 'salinas'], help='选择要使用的数据集')

    parser.add_argument('--train_percent', default=0.15, type=float, help='训练集比例')
    parser.add_argument('--use_val', action='store_true', default=True, help='使用验证集')
    parser.add_argument('--val_percent', default=0.1, type=float, help='验证集比例(验证集从测试集中划分)')

    parser.add_argument('--save_interval', type=int, default=10, help='保存模型的间隔')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='保存模型的目录')
    parser.add_argument('--random_state', type=int, default=42, help='随机数种子')
    parser.add_argument('--spatial_size', default=9, type=int, help='构建数据立方体时采用的窗口大小')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='使用 GPU 训练')
    parser.add_argument('--log_dir', type=str, default='./logs', help='保存日志的目录')

    args = parser.parse_args()

    main(args)