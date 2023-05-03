import argparse
import datetime
import logging
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Conv3DTranspose, multiply, Embedding, BatchNormalization, LeakyReLU, Flatten, Dense, Reshape, Activation, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

from utils.load_data import load_data, split_data, create_image_cube
from utils.metrics import reports, save_report, read_report, visualize_report
from utils.visualization import RGBImage

import wandb
from PIL import Image


class GAN3D_2D_HS():
    def __init__(self, input_shape, num_classes, latent_dim, checkpoint_dir):
		# Input shape
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        optimizer = Adam(lr=0.0002, beta_1=0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
	
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses, 
			optimizer=optimizer,  
		    metrics=['accuracy'])
	
        self.generator = self.build_generator()
	
	    # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))

        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator

        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)



    def build_generator(self):

        model = Sequential()

        model.add(Dense(1 * 1 * 2 * 128, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((1, 1, 2, 128)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv3DTranspose(128, kernel_size=(3, 3, 7), strides=(1, 1, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv3DTranspose(64, kernel_size=(3, 3, 5), strides=(3, 3, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv3DTranspose(32, kernel_size=(3, 3, 3), strides=(3, 3, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv3DTranspose(1, kernel_size=(3, 3, 3), strides=(1, 1, 2), padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv3D(8, kernel_size=(3, 3, 7), strides = (1, 1, 2), padding="same", input_shape=self.input_shape))
        # model.add(Conv3D(16, kernel_size=(3, 3, 20), strides = (1, 1, 4), input_shape=self.input_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv3D(16, kernel_size=(3, 3, 5), strides = (1, 1, 1), padding="same"))
        # model.add(Conv3D(32, kernel_size=(3, 3, 10), strides = (1, 1, 2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv3D(32, kernel_size=(3, 3, 3), strides = (1, 1, 1), padding="same"))
        # model.add(Conv3D(64, kernel_size=(3, 3, 5), strides = (1, 1, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        # Compute the output shape of the last Conv3D layer
        output_shape = model.layers[-1].output_shape[1:]
        reshaped_shape = (output_shape[0], output_shape[1], output_shape[2] * output_shape[3])

        # Reshape the output to the desired format
        model.add(Reshape(reshaped_shape))
        model.add(Conv2D(64, kernel_size=(3, 3)))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())

        model.summary()

        img = Input(shape=self.input_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])
    
    def train(self, x_train, y_train, epochs, batch_size, save_interval = 50, run_name='GAN3D_HS'):  
        wandb.init(project='GAN3D_2D_HS', name=run_name)
        wandb.config.update({"epochs": epochs, "batch_size": batch_size})

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        best_op_acc_val = float('inf')
        num_batches = int(x_train.shape[0] / batch_size)
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of data
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_data = x_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # The labels of the digits that the generator tries to create an
                # image representation of
                fake_labels = np.random.randint(0, self.num_classes, batch_size)

                # Generate a half batch of new images
                fake_data = self.generator.predict([noise, fake_labels])

                # Image labels. 0-9 
                real_labels = y_train[idx]

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(real_data, [valid, real_labels])
                d_loss_fake = self.discriminator.train_on_batch(fake_data, [fake, fake_labels])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, fake_labels], [valid, fake_labels])

                # Plot the progress

                print ("Training Metrics: %d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [D loss fake: %f, acc.: %.2f%%, op_acc: %.2f%%] [D loss real: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f] " \
                        % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], \
                           d_loss_fake[0], 100*d_loss_fake[3], 100*d_loss_fake[4], \
                           d_loss_real[0], 100*d_loss_real[3], 100*d_loss_real[4], \
                           g_loss[0]))
                wandb.log({
                    'D_loss': d_loss[0],
                    # 'D_acc': 100 * d_loss[3],
                    'D_op_acc': d_loss[4],
                    'G_loss': g_loss[0]
                })

                
             # Evaluate on the validation set
            idx_val = np.random.randint(0, x_test.shape[0], batch_size)
            real_data_val = x_test[idx_val]
            real_labels_val = y_test[idx_val]
            d_loss_val = self.discriminator.evaluate(real_data_val, [valid, real_labels_val], verbose=0)
            g_loss_val = self.combined.evaluate([noise, fake_labels], [valid, fake_labels], verbose=0)
            print("Validation Metrics: %d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss_val[0], 100 * d_loss_val[3], 100 * d_loss_val[4], g_loss_val[0]))
            
            wandb.log({

                'D_loss_val': d_loss_val[0],
                # 'D_acc_val': 100 * d_loss_val[3],
                'D_op_acc_val': d_loss_val[4],
                'G_loss_val': g_loss_val[0]
            })

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_model(epoch)

            if d_loss_val[4] < best_op_acc_val:
                best_op_acc_val = d_loss_val[4]
                self.save_best_model()
        wandb.finish()

    def save_model(self, epoch):
        def save(model, model_name, epoch):
            weights_path = os.path.join(self.checkpoint_dir, "%s_%d.hdf5" % (model_name, epoch))
            model.save_weights(weights_path)

        # save(self.generator, "G", epoch)
        save(self.discriminator, "D", epoch)

    def save_best_model(self):
        weights_path = os.path.join(self.checkpoint_dir, "%s_best.hdf5" % ("D"))
        self.discriminator.save_weights(weights_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=400, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--dataset', type=str, default='indian_pines', choices=['indian_pines', 'salinas'], help='选择要使用的数据集')

    parser.add_argument('--num_components', type=int, default=3, help='PCA 降维后的维度')
    parser.add_argument('--preprocess', type=str, default='standard', choices=['minmax', 'standard', 'none'], help='数据预处理方式')    
    parser.add_argument('--train_percent', default=0.15, type=float, help='训练集比例')
    parser.add_argument('--repeat', default=1, type=int, help='实验重复次数')
    # parser.add_argument('--use_val', action='store_true', default=True, help='使用验证集')
    # parser.add_argument('--val_percent', default=0.1, type=float, help='验证集比例(验证集从测试集中划分)')

    parser.add_argument('--save_interval', type=int, default=50, help='保存模型的间隔')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/GAN3D_2D_HS', help='保存模型的目录')
    parser.add_argument('--random_state', type=int, default=42, help='随机数种子')
    parser.add_argument('--spatial_size', default=9, type=int, help='构建数据立方体时采用的窗口大小')
    
    parser.add_argument('--use_gpu', action='store_true', default=True, help='使用 GPU 训练')
    parser.add_argument('--test_only', action='store_true', default=False, help='只测试模型')

    args = parser.parse_args()

    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 1 数据预处理
    # 加载数据集
    data_ori, label_ori, num_classes = load_data(args.dataset, preprocessing=args.preprocess)

    data_ori, label_ori = create_image_cube(data_ori, label_ori, window_size=args.spatial_size, remove_zero_labels = False)
    
    # # 0 代表背景，不参与训练
    data = data_ori[label_ori!=0]
    label = label_ori[label_ori!=0] - 1


    for pos in range(args.repeat):
        save_path = os.path.join(args.checkpoint_dir, str(pos))

        test_percent = 1 - args.train_percent
        x_train, x_test, y_train, y_test = split_data(data, label,
                                                test_percent=test_percent,
                                                random_state=args.random_state + pos)
        x_test  = x_test[..., np.newaxis]
        x_train = x_train[..., np.newaxis]

        # 输出训练集和测试集的形状和训练设置
        logging.info(f'x_train shape: {x_train.shape}')
        logging.info(f'y_train shape: {y_train.shape}')
        logging.info(f'x_test shape: {x_test.shape}')
        logging.info(f'y_test shape: {y_test.shape}')
        
        input_shape = x_train.shape[1:]
        latent_dim = 100

        run_name = f'GAN3D_2D_HS_{args.dataset}_b{args.batch_size}_e{args.num_epochs}_w{args.spatial_size}_n{pos}'

        if not args.test_only:
        # 训练部分
            GAN = GAN3D_2D_HS(input_shape, num_classes, latent_dim, checkpoint_dir=save_path)
            GAN.train(x_train, y_train, args.num_epochs, args.batch_size, args.save_interval, run_name = run_name)
        
        # 测试部分
        # del GAN
        GAN_D = GAN3D_2D_HS(input_shape, num_classes, latent_dim, checkpoint_dir=save_path).discriminator
        GAN_D.load_weights(os.path.join(save_path, 'D_best.hdf5'))


        _, y_pred = GAN_D.predict(x_test)
        report = reports(np.argmax(y_pred, axis=1), y_test)
        save_report(*report, save_path)
        # Read the report from the CSV file
        report_dict = read_report(save_path)

        # Visualize the report in Python
        visualize_report(report_dict)

        # 生成预测图像
        _, y_pred = GAN_D.predict(data_ori[..., np.newaxis])

        image_pred = np.argmax(y_pred, axis=1).reshape(145,145)

        # 将背景像素设置为0，分类标签从1开始
        label_ori = label_ori.reshape(145,145)
        image_pred += 1
        image_pred[ label_ori == 0] = 0

        rgb_pred = RGBImage(image_pred)
        rgb_true = RGBImage(label_ori)

        img_rgb_pred = Image.fromarray(np.uint8(rgb_pred))
        img_rgb_true = Image.fromarray(np.uint8(rgb_true))
        img_rgb_pred.save(os.path.join(save_path, "y_pred.png"))
        img_rgb_true.save(os.path.join(save_path, "y_true.png"))
