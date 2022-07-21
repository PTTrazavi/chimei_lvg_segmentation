# Import libraries
import tensorflow as tf
print(tf.__version__)
import keras
print(keras.__version__)
print(tf.test.gpu_device_name())
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
# Loading dataset
DATA_DIR = './chimei_1304_144/'

train_folder = DATA_DIR + 'train/'
train_L_folder = DATA_DIR + 'trainannot/'
print("total images in train folder: ", len(os.listdir(train_folder)))
print("total images in train_L folder: ", len(os.listdir(train_L_folder)))
val_folder = DATA_DIR + 'val/'
val_L_folder = DATA_DIR + 'valannot/'
print("total images in val folder: ", len(os.listdir(val_folder)))
print("total images in val_L folder: ", len(os.listdir(val_L_folder)))
test_folder = DATA_DIR + 'test/'
test_L_folder = DATA_DIR + 'testannot/'
print("total images in testl folder: ", len(os.listdir(test_folder)))
print("total images in test_L folder: ", len(os.listdir(test_L_folder)))

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')
# Dataloader and utility functions
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(20, 5)) #16,5
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['bg', 'heart'] #, 'unlabelled'

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['bg', 'heart'])
image, mask = dataset[58] # get some sample
print(mask.shape)
# Augmentations¶
import albumentations as A
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        # A.HorizontalFlip(p=0.5),

        #A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0, p=0.5, border_mode=0), # strong augmentation
        # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0, p=0.3, border_mode=0), # weak augmentation

        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.RandomCrop(height=512, width=512, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),

        A.IAAPerspective(p=0.5), # strong augmentation
        # A.IAAPerspective(p=0.3), # weak augmentation

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9, # strong augmentation
            # p=0.3, # weak augmentation
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9, # strong augmentation
            # p=0.3, # weak augmentation
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9, # strong augmentation
            # p=0.3, # weak augmentation
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.RandomCrop(height=512, width=512, always_apply=True),
        A.PadIfNeeded(512, 512)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# Lets look at augmented data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['bg', 'heart'], augmentation=get_training_augmentation())
image, mask = dataset[11] # get some sample
print(mask.shape)

# Segmentation model training¶
import segmentation_models as sm
BACKBONE = 'efficientnetb4' #'efficientnetb4' 'resnet50'
BATCH_SIZE = 2 #4 8
CLASSES = ['bg', 'heart']
LR = 0.0001 #0.0001 0.01
EPOCHS = 3 #40

preprocess_input = sm.get_preprocessing(BACKBONE)
# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation) #, encoder_weights=None
# Set loss and metrics¶
# define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss ['bg', 'heart', 'unlabelled']
# class_weights=np.array([0.5, 1, 0.01]) pixel 0-1 level

# dice_loss = sm.losses.DiceLoss() #class_weights=np.array([0.5, 1, 0.01])
jaccard_loss = sm.losses.JaccardLoss() #class_weights=np.array([0.5, 1, 0.01])

focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
# ce_loss = sm.losses.BinaryCELoss() if n_classes == 1 else sm.losses.CategoricalCELoss() #class_weights=np.array([0.5, 1, 0.01])

# total_loss = dice_loss + (1 * focal_loss)
# total_loss = dice_loss + (1 * ce_loss)
total_loss = jaccard_loss + (1 * focal_loss)
# total_loss = jaccard_loss + (1 * ce_loss)
# total_loss = ce_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), # , class_indexes=[1]
          #  sm.metrics.FScore(threshold=0.5), # , class_indexes=[1]
           sm.metrics.Precision(threshold=0.5), # , class_indexes=[1]
           sm.metrics.Recall(threshold=0.5)] # , class_indexes=[1]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# Load dataset
# Dataset for train images
train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

# Dataset for validation images
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 512, 512, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 512, 512, n_classes)

assert valid_dataloader[0][0].shape == (1, 512, 512, 3)
assert valid_dataloader[0][1].shape == (1, 512, 512, n_classes)

# Define callbacks for learning rate scheduling and best checkpoints saving
def my_scheduler(epoch, lr):
  if epoch < 31:
    return lr
  else:
    return 0.00001 # lr * tf.math.exp(-0.1)

callbacks = [
    # monitor='val_iou_score', mode='max' # mode='min'
    keras.callbacks.ModelCheckpoint('checkpoints/UNET_1304_1to40_jf_{epoch:02d}_{val_iou_score:.2f}_model.h5', save_weights_only=True, save_best_only=True, monitor='val_iou_score', mode='max'),
    keras.callbacks.ReduceLROnPlateau(patience=12, verbose=1), # patience=12, min_lr=0.00001
    # keras.callbacks.LearningRateScheduler(schedule=my_scheduler, verbose=1)
    keras.callbacks.CSVLogger('checkpoints/UNET_1304_1to40_jf_model.csv')
]
# Train the model
# train model
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)

model.save_weights('checkpoints/UNET_1304_1to40_jf_40_model.h5')
