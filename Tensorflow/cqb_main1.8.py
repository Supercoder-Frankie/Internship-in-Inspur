import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet import ResNet50, ResNet152
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization,Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split, KFold

from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.layers import PReLU, Activation
from tensorflow.python.keras.models import Model
import os
import glob
import logging
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def load_preprocess_img_with_aug(img_path, labels):
    """
    功能：用于根据指定路径载入图片文件
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224,224))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # random crop / central crop
    # image = tf.image.random_brightness(image, 0.2)
    # image = tf.image.random_contrast(image, 0, 1)
    image = image/255.
    return image, labels


def load_preprocess_img(img_path, labels):
    """
    功能：用于根据指定路径载入图片文件
    """
    image = tf.io.read_file(img_path)
    # image = tf.image.decode_png(image, channels=3) if img_path[-3:] == 'png' else tf.image.decode_jpeg(image)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, (224,224))
    image = image/255.
    return image, labels
#
# def correct_num_batch(y_true, y_pred):
#     correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
#     correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
#     return correct_num
#
# def cross_entropy_batch(y_true, y_pred, label_smoothing):
#     cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
#     return tf.reduce_mean(cross_entropy)
#
# def l2_loss(model, weights=1e-4):
#     variable_list = []
#     for v in model.trainable_variables:
#         if 'kernel' in v.name:
#             variable_list.append(tf.nn.l2_loss(v))
#     return tf.add_n(variable_list) * weights
#
# @tf.function 计算梯度
# def train_step(model, images, labels, optimizer):
#     with tf.GradientTape() as tape:
#         prediction = model(images, training=True)
#         ce = cross_entropy_batch(labels, prediction, label_smoothing=0.1)
#         l2 = l2_loss(model)
#         loss = ce + l2
#         gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return ce, prediction


CLASS_NUM = 2
BATCH_SIZE = 16
EPOCHES = 100
LR = 1e-3
FOLDS = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE   #
images_path = []
labels = []
root_path = '/Users/liruizhi/Downloads/实习/浪潮Inspur/标注/data2'
class_dir = os.listdir(root_path)
model_save_dir = 'models/resnet/cqb_main0107'
class_dir = [d for d in class_dir if not '.' in d]
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
for i, cls in enumerate(class_dir):  # 给图片加标签
    for file_path in glob.glob(os.path.join(root_path, cls, '*')):
        labels.append(i)
        images_path.append(file_path)

print('这是labels：')
print(labels)
print()
print('这是imagse_path:')
print(images_path)

images_path = np.array(images_path)
labels = np.array(labels)
# img_train_path, img_test_path, label_train, label_test = train_test_split(
#     images_path,labels, test_size=0.1, stratify=labels, random_state=2020)

# learning_rate_schedules = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1., decay_steps=1, decay_rate=0.96)
# learning_rate_schedules = tf.keras.experimental.CosineDecay(
#                 initial_learning_rate=0.1, decay_steps=100)
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model = tf.keras.models.Sequential()
model.add(ResNet50(include_top=False, weights="imagenet")) ########################## IMAGENET PRETRAINED MODEL
model.add(keras.layers.GlobalAveragePooling2D(name='avg_pool'))
model.add(tf.keras.layers.Dense(CLASS_NUM, activation='softmax'))
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()


reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(    # Reduce learning rate when a metric has stopped improving.
    monitor='val_loss', factor=0.5, patience=5, verbose=1,
    mode='auto', min_delta=0.01, cooldown=0, min_lr=0.0001
)
early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10)


kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=2020)
day = datetime.datetime.now().day
for fold, (trn_idx, val_idx) in enumerate(kfold.split(images_path, labels)):
    logging.info("***************fold:{}***************".format(fold))
    filepath = model_save_dir + "/model-fold{0}-day{1}.hdf5".format(fold+1, day)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(   # 回调以某种频率保存Keras模型或模型权重。
        filepath=filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True)
    img_train_path, img_test_path, label_train, label_test = images_path[trn_idx], images_path[val_idx], \
                                                             labels[trn_idx], labels[val_idx]
    ds_train = tf.data.Dataset.from_tensor_slices((img_train_path, label_train)) \
        .map(load_preprocess_img_with_aug, num_parallel_calls=AUTOTUNE).repeat(4).shuffle(100).batch(
        BATCH_SIZE).prefetch(16)   #?
    ds_test = tf.data.Dataset.from_tensor_slices((img_test_path, label_test)) \
        .map(load_preprocess_img, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

    model.fit(ds_train, epochs=EPOCHES, validation_data=ds_test, callbacks=[early_stop])


keras
history
tensorboard

# 酒店
# 餐饮相关
