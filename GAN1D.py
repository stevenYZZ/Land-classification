import argparse
import datetime
import logging
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, multiply, UpSampling1D, Embedding, BatchNormalization, LeakyReLU, Flatten, Dense, Reshape, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

from utils.load_data import load_data, split_data
from utils.metrics import reports, save_report, read_report, visualize_report
from utils.visualization import RGBImage



class GAN1D():
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
        # print([noise, label])
        # print([valid, target_label])

        # exit()
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)



    def build_generator(self):
	
        model = Sequential()
	
        model.add(Dense(512, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((1, 512)))
        model.add(BatchNormalization(momentum=0.8))
	
        model.add(UpSampling1D())
        model.add(Conv1D(512, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
	
        model.add(UpSampling1D(size=5))
        model.add(Conv1D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
	
        model.add(Conv1D(1, kernel_size=4, padding='same'))
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
	
        model.add(Conv1D(256, kernel_size=4, strides=1, input_shape=self.input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
	
        model.add(Conv1D(512, kernel_size=4, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
	
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(128, kernel_size=4, strides=1, padding="same"))
        model.add(Flatten())
	
        model.summary()
	
        img = Input(shape=self.input_shape)

        # Extract feature representation
        features = model(img)
	
        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)
	
        return Model(img, [validity, label])
    
    def train(self, x_train, y_train, epochs, batch_size, save_interval = 50, run_name='GAN1D'):  
        '''
            Train the GAN1D
                x_train: training data, 2D numpy array of shape (num_samples, num_bands),
                y_train: training labels, 1D numpy array of shape (num_samples,)
                epochs: number of epochs to train for
                batch_size: batch size to use during training
                save_interval: interval at which to save the model
                run_name: name of the model to save
        '''
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        best_d_loss = float('inf')
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of data
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_data = x_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # noise = np.random.randn(self.latent_dim * batch_size).reshape(batch_size, self.latent_dim)
            

            # The labels of the digits that the generator tries to create an
            # image representation of
            fake_labels = np.random.randint(0, self.num_classes, batch_size)

            # print(fake_labels.shape)
            # exit()
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

            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            # If at save interval => save generated image samples
            

            if epoch % save_interval == 0:
                self.save_model(epoch)

            if d_loss[0] < best_d_loss:
                best_d_loss = d_loss[0]
                self.save_best_model()
    
    def save_model(self, epoch):
        def save(model, model_name, epoch):
            weights_path = os.path.join(self.checkpoint_dir, "%s_%d.hdf5" % (model_name, epoch))
            model.save_weights(weights_path)

        # save(self.generator, "G", epoch)
        save(self.discriminator, "D", epoch)

    def save_best_model(self):
        weights_path = os.path.join(self.checkpoint_dir, "%s_best.hdf5" % ("D"))
        self.discriminator.save_weights(weights_path)

    # def test(self, x_test, y_test):
    #     """
    #     Test the discriminator using test data.
    #         x_test: Test data, 2D numpy array of shape (num_samples, num_bands)
    #         y_test: Test labels, 1D numpy array of shape (num_samples,)
    #     """

    #     # Predict the validity and class labels using the discriminator
    #     stats = np.ones(self.num_classes + 3) * -1000.0 # OA, AA, K, Aclass
    #     _, y_pred = self.discriminator.predict(x_test)
    #     stats = reports(np.argmax(y_pred, axis=1), y_test)[2]
    #     print(stats)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=400, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--dataset', type=str, default='indian_pines', choices=['indian_pines', 'salinas'], help='选择要使用的数据集')

    parser.add_argument('--num_components', type=int, default=10, help='PCA 降维后的维度')
    parser.add_argument('--preprocess', type=str, default='standard', choices=['minmax', 'standard', 'none'], help='数据预处理方式')    
    parser.add_argument('--train_percent', default=0.15, type=float, help='训练集比例')
    parser.add_argument('--repeat', default=1, type=int, help='实验重复次数')
    # parser.add_argument('--use_val', action='store_true', default=True, help='使用验证集')
    # parser.add_argument('--val_percent', default=0.1, type=float, help='验证集比例(验证集从测试集中划分)')

    parser.add_argument('--save_interval', type=int, default=50, help='保存模型的间隔')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/GAN1D', help='保存模型的目录')
    parser.add_argument('--random_state', type=int, default=42, help='随机数种子')
    # parser.add_argument('--spatial_size', default=19, type=int, help='构建数据立方体时采用的窗口大小')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='使用 GPU 训练')
    parser.add_argument('--test_only', action='store_true', default=False, help='只测试模型')


    # parser.add_argument('--log_dir', type=str, default='./logs', help='保存日志的目录')
    args = parser.parse_args()

    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 1 数据预处理
    # 加载数据集
    data_ori, label_ori, num_classes = load_data(args.dataset, num_components = args.num_components, preprocessing=args.preprocess)

    data = data_ori.reshape(-1, data_ori.shape[-1])
    label = label_ori.reshape(-1)
    
    # 0 代表背景，不参与训练
    data = data[label!=0]
    label = label[label!=0] - 1

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
        if not args.test_only:
        # 训练部分
            GAN = GAN1D(input_shape, num_classes, latent_dim, checkpoint_dir=save_path)
            GAN.train(x_train, y_train, args.num_epochs, args.batch_size, args.save_interval)
        
        # 测试部分
        # del GAN
        GAN_D = GAN1D(input_shape, num_classes, latent_dim, checkpoint_dir=save_path).discriminator
        GAN_D.load_weights(os.path.join(args.checkpoint_dir, 'D_best.hdf5'))


        _, y_pred = GAN_D.predict(x_test)
        report = reports(np.argmax(y_pred, axis=1), y_test)
        save_report(*report, save_path)
        # Read the report from the CSV file
        report_dict = read_report(save_path)

        # Visualize the report in Python
        visualize_report(report_dict)

        # 生成预测图像
        _, y_pred = GAN_D.predict(data_ori.reshape(145*145,10,1))

        image_pred = np.argmax(y_pred, axis=1).reshape(145,145)

        # 将背景像素设置为0，分类标签从1开始
        image_pred += 1
        image_pred[label_ori == 0] = 0

        rgb_pred = RGBImage(image_pred)
        rgb_true = RGBImage(label_ori.reshape(145,145))
        cv2.imwrite(os.path.join(save_path, "y_pred.png"), rgb_pred)
        cv2.imwrite(os.path.join(save_path, "y_true.png"), rgb_true)








	