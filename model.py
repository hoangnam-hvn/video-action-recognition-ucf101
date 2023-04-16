from keras. models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.applications import InceptionResNetV2
from keras.layers import Dense, TimeDistributed, GlobalAveragePooling3D, Rescaling, Input

from preprocess_data import *


def dar(x):
    u = []
    for i in range(10):
        xith = x[:, i]
        u.append(xith)
    return u
  
input_shape = (general_options[0], general_options[1], general_options[2], general_options[3])

irn = InceptionResNetV2(
    include_top=False,
    input_shape=(general_options[1], general_options[2], general_options[3]),
    )
irn.trainable = False

input_layer = Input(input_shape)

r = Rescaling(scale=1./255)(input_layer)

t = TimeDistributed(irn)(r)
t = Dense(256, activation='relu')(t)
t = GlobalAveragePooling3D()(t)
t = Dense(128, activation='relu')(t)

output_layer = Dense(10, activation='softmax')(t)

model = Model(input_layer, output_layer)

model.summary()
model.compile(Adam(0.001), CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(train_ds, epochs=2, verbose=1, validation_data=val_ds)
model.evaluate(test_ds)
model.save('ucf101model.h5')
