__author__ = "blu3crab"
__license__ = "Apache License 2.0"
__version__ = "0.0.1"

# https://www.kaggle.com/matucker/exercise-convolutions-for-computer-vision/edit

#https://www.kaggle.com/matucker/exercise-tensorflow-programming/edit

import os
from os.path import join


hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/hot_dog'

hot_dog_paths = [join(hot_dog_image_dir,filename) for filename in
                            ['1000288.jpg',
                             '127117.jpg']]

not_hot_dog_image_dir = '../input/hot-dog-not-hot-dog/seefood/train/not_hot_dog'
not_hot_dog_paths = [join(not_hot_dog_image_dir, filename) for filename in
                            ['823536.jpg',
                             '99890.jpg']]

img_paths = hot_dog_paths + not_hot_dog_paths
print("hot_dog_paths->", hot_dog_paths)
print("not_hot_dog_paths->", not_hot_dog_paths)

hotdog_image_data = read_and_prep_images(hot_dog_paths)
preds_for_hotdogs = model.predict(hotdog_image_data)
print("preds_for_hotdogs->", is_hot_dog(preds_for_hotdogs))

from IPython.display import Image, display
from learntools.deep_learning.decode_predictions import decode_predictions
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array


image_size = 224

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    output = preprocess_input(img_array)
    return(output)


my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
test_data = read_and_prep_images(img_paths)
preds = my_model.predict(test_data)
print(preds)

most_likely_labels = decode_predictions(preds, top=3)
print(most_likely_labels)

for i, img_path in enumerate(img_paths):
    display(Image(img_path))
    print(most_likely_labels[i])

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_3 import *
print("Setup Complete")

# Experiment with code outside the function, then move it into the function once you think it is right

# the following lines are given as a hint to get you started
decoded = decode_predictions(preds, top=1)
print(decoded)

labels = [d[0][1] for d in decoded]
print("labels->")
print(labels)
hotdogs = [l == 'hotdog' for l in labels]
print(hotdogs)


def is_hot_dog(preds):
    '''
    inputs:
    preds_array:  array of predictions from pre-trained model

    outputs:
    is_hot_dog_list: a list indicating which predictions show hotdog as the most likely label
    '''
    labels = [d[0][1] for d in decoded]
    print(labels)
    hotdogs = [l == 'hotdog' for l in labels]
    print(hotdogs)
    return hotdogs
    # pass


# Check your answer
q_1.check()

print("hot_dog_paths->", hot_dog_paths)
print("not_hot_dog_paths->", not_hot_dog_paths)

new_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

hotdog_image_data = read_and_prep_images(hot_dog_paths)
preds_for_hotdogs = new_model.predict(hotdog_image_data)
print("preds_for_hotdogs->", is_hot_dog(preds_for_hotdogs))

not_hotdog_image_data = read_and_prep_images(not_hot_dog_paths)
preds_for_not_hotdogs = new_model.predict(not_hotdog_image_data)
print("preds_for_not_hotdogs->", is_hot_dog(preds_for_not_hotdogs))

# def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):
#     # iterate through images testing prediction against actual
#     # if match, bump accurate counter
#     # accuracy = accurate counter / total
#     my_model = ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
#     test_data = read_and_prep_images(img_paths)
#     preds = my_model.predict(test_data)

#     hotdog_count = len(paths_to_hotdog_images)
#     not_hotdog_count = len(paths_to_other_images)
#     print("total hotdog vs not hotdog = ", hotdog_count, " vs ", not_hotdog_count)

#     pred_hotdog_count = 0
#     pred_not_hotdog_count = 0
#     hotdogs = is_hot_dog(preds)
#     print(hotdogs)
#     # find total pred hotdog - not_hotdog
#     for pred in hotdogs:
#         if (pred): pred_hotdog_count = pred_hotdog_count + 1
#         else: pred_not_hotdog_count = pred_not_hotdog_count + 1

#     print("total pred hotdog vs not hotdog = ", pred_hotdog_count, " vs ", pred_not_hotdog_count)

#     total = hotdog_count + not_hotdog_count
#     delta = (abs(hotdog_count - pred_hotdog_count)) + (abs(not_hotdog_count - pred_not_hotdog_count))
#     accuracy = ((total - delta)/total)
#     print("accuracy = ", accuracy)
#     return accuracy

# #print(preds)

# # Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths were created in the setup code
# my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)
# print("Fraction correct in small test set: {}".format(my_model_accuracy))

# # Check your answer
# q_2.check()

def calc_accuracy(model, paths_to_hotdog_images, paths_to_other_images):
    # We'll use the counts for denominator of accuracy calculation
    num_hot_dog_images = len(paths_to_hotdog_images)
    num_other_images = len(paths_to_other_images)
    print("total hotdog vs not hotdog = ", num_hot_dog_images, " vs ", num_other_images)

    hotdog_image_data = read_and_prep_images(paths_to_hotdog_images)
    preds_for_hotdogs = model.predict(hotdog_image_data)
    print("preds_for_hotdogs->", is_hot_dog(preds_for_hotdogs))
    # Summing list of binary variables gives a count of True values
    num_correct_hotdog_preds = sum(is_hot_dog(preds_for_hotdogs))

    other_image_data = read_and_prep_images(paths_to_other_images)
    preds_other_images = model.predict(other_image_data)
    print("preds_other_images->", is_hot_dog(preds_other_images))
    # Number correct is the number judged not to be hot dogs
    num_correct_other_preds = num_other_images - sum(is_hot_dog(preds_other_images))

    print("total correct hotdog vs not hotdog = ", num_correct_hotdog_preds, " vs ", num_correct_other_preds)

    total_correct = num_correct_hotdog_preds + num_correct_other_preds
    total_preds = num_hot_dog_images + num_other_images

    print("total_correct=", total_correct)
    print("total_preds=", total_preds)

    return total_correct / total_preds


# Code to call calc_accuracy.  my_model, hot_dog_paths and not_hot_dog_paths were created in the setup code
print("hot_dog_paths->", hot_dog_paths)
print("not_hot_dog_paths->", not_hot_dog_paths)
my_model_accuracy = calc_accuracy(my_model, hot_dog_paths, not_hot_dog_paths)
print("Fraction correct in small test set: {}".format(my_model_accuracy))

# Check your answer
q_2.check()

# import the model
from tensorflow.keras.applications import VGG16


vgg16_model = VGG16(weights='../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
# calculate accuracy on small dataset as a test
vgg16_accuracy = calc_accuracy(vgg16_model, hot_dog_paths, not_hot_dog_paths)

print("Fraction correct in small dataset: {}".format(vgg16_accuracy))

# Check your answer
q_3.check()


