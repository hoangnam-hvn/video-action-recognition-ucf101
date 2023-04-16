import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import to_categorical

np.random.seed(43)


ds_path = './ucf101'

train_csv = ds_path + '/train.csv'
test_csv = ds_path + '/test.csv'
val_csv = ds_path + '/val.csv'

train_np = pd.read_csv(train_csv).to_numpy()[:, [1, 2]]
test_np = pd.read_csv(test_csv).to_numpy()[:, [1, 2]]
val_np = pd.read_csv(val_csv).to_numpy()[:, [1, 2]]

# Paths and labels
classes = 10

unique_labels = np.unique(val_np[:, 1])
selected_labels = np.random.choice(unique_labels, size=classes)
label_map = {label: i for i, label in enumerate(selected_labels)}
label_reverse = {i: label for i, label in enumerate(selected_labels)}

selected_train = train_np[np.isin(train_np[:, 1], selected_labels)]
selected_test = test_np[np.isin(test_np[:, 1], selected_labels)]
selected_val = val_np[np.isin(val_np[:, 1], selected_labels)]

selected_train[:, 0] = ds_path + selected_train[:, 0]
selected_test[:, 0] = ds_path + selected_test[:, 0]
selected_val[:, 0] = ds_path + selected_val[:, 0]

n = 50
train_paths = []
test_paths = []
val_paths = []
train_labels = []
test_labels = []
val_labels = []

for label in selected_labels:
    A = selected_train[np.isin(selected_train[:, 1], label)][:n]
    B = selected_test[np.isin(selected_test[:, 1], label)][:int(0.2*n)]
    C = selected_val[np.isin(selected_val[:, 1], label)][:int(0.2*n)]
    
    train_paths.extend(list(A[:, 0]))
    test_paths.extend(list(B[:, 0]))
    val_paths.extend(list(C[:, 0]))
    
    train_labels.extend(list(A[:, 1]))
    test_labels.extend(list(B[:, 1]))
    val_labels.extend(list(C[:, 1]))

train_labels = [label_map[label] for label in train_labels]
test_labels = [label_map[label] for label in test_labels]
val_labels = [label_map[label] for label in val_labels]

# One hot label
y_train = to_categorical(train_labels, classes)
y_test = to_categorical(test_labels, classes)
y_val = to_categorical(val_labels, classes)
# General Options

# (frame_count, width, height, channels, step, batch_size, buffer_size)
general_options = (10, 224, 224, 3, 5, 5, tf.data.AUTOTUNE)

# Preprocess and read frames

def get_frames(path, frame_count, width, height, channels, step):
    cap = cv2.VideoCapture(path)
    frames = []
    
    rlength = frame_count * step
    clength = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if clength < rlength:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        start = np.random.randint(0, clength - rlength + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
    # Extract frames from the video file and preprocess each frame
    for i in range(rlength):
        ret, frame = cap.read()
        if i % step == 0:
            if ret:
                frame = PPF(frame, width, height)
                frames.append(frame)
            else:
                black_frame = np.zeros(shape=(width, height, channels))
                frames.append(black_frame)
    cap.release()
    frames = np.stack(frames, axis=0)
    return frames

# Preprocessing function for each frame 
def PPF(frame, width, height):
    frame = tf.image.resize_with_crop_or_pad(frame, width, height)
    ppf = frame[:, :, ::-1]
    return ppf

# Random Sample Visualization

# # Show frames
# def display_frames(frames, label):
#     fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
#     for i, ax in enumerate(axes.flat):
#         f = frames[i] / 255.0
#         ax.imshow(f)
#         ax.set_title(f'{label} {i+1}')
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# # Random index
# i = np.random.randint(low=len(train_paths) - 1)
# ex_path = train_paths[i]
# ex_label = label_reverse[train_labels[i]]

# frames = get_frames(ex_path, 10, 224, 224, 3, 6)
# display_frames(frames, ex_label) 

# Generate frames and labels

# Generate preprocessed data        
class GPAL:
    def __init__(self, paths, frame_count, width, height, channels, step, labels):
        self.paths = paths
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.channels = channels
        self.step = step
        self.labels = labels
        
    def __call__(self):
        PL = list(zip(self.paths, self.labels))
        p = (self.frame_count, self.width, self.height, self.channels, self.step)
        for path, label in PL:
            frames = get_frames(path, *p)
            yield frames, label
            
# Get dataset

def get_ds(generator, paths, frame_count, width, height, channels, step, batch_size, bf, labels):
    generator = GPAL(paths, frame_count, width, height, channels, step, labels)
    ds = tf.data.Dataset.from_generator(generator,
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=((frame_count, width, height, channels), (10,)))
    ds = ds.shuffle(1024)    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=bf)
    return ds


train_ds = get_ds(GPAL, train_paths, *general_options, y_train)

test_ds = get_ds(GPAL, test_paths, *general_options, y_test)

val_ds = get_ds(GPAL, val_paths, *general_options, y_val)