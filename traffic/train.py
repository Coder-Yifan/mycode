from model import Lenet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from load_data import load_data
import time
EPOCHS = 35
INIT_LR = 1e-3
BATCH_SIZE = 32
CLASS_NUM = 62
norm_size = 32

train_path = 'E:/data/traffic/traffic-sign/train'
test_path = 'E:/data/traffic/traffic-sign/Humpback Whale'
s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())  #时间戳
log_path = 'E:/data/traffic/logs/log_%s'%(s_time)
model_path = 'E:/data/traffic/mymodel.h5'
train_data,train_label = load_data(train_path,norm_size,norm_size,CLASS_NUM)
test_data,test_label = load_data(test_path,norm_size,norm_size,CLASS_NUM)
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#
test_datagen = ImageDataGenerator(rescale=1./255)
#
train_generator = train_datagen.flow(
        train_data,train_label,
        #target_size=(norm_size, norm_size),
        batch_size=BATCH_SIZE,
        #class_mode='sparse'
        )
print(train_generator)
# validation_generator = test_datagen.flow_from_directory(
#         test_data,test_label,
#         target_size=(norm_size, norm_size),
#         batch_size=BATCH_SIZE,
#         class_mode='sparse')

model = Lenet.build(width=norm_size,height=norm_size,depth=3,classes=CLASS_NUM)
adm = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss='categorical_crossentropy',
              optimizer=adm,
              metrics=['accuracy'])

board = TensorBoard(log_dir=log_path,histogram_freq=1)
H = model.fit_generator(
        train_generator,
        #steps_per_epoch=len(train_generator)/EPOCHS,
        epochs=EPOCHS,
        validation_data=(test_data,test_label),
        callbacks=[board]
)
model.save(model_path)