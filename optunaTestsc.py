from numpy.core.numeric import NaN
from MCtool.RFilter import gray
from genericpath import exists
from matplotlib import image
import math
import sys
import time

import cv2
from matplotlib import pyplot as plt
from tensorflow.python.keras.backend import dtype
from DeepLearning import LearnAndTest
from Rpkg.Rfund.InputFeature import InputFeature
import datetime
import os
import gc
import tensorflow as tf
import random
import numpy as np
import pandas as pd

from Rpkg.Rfund import ReadFile, WriteFile
from Rpkg.Rmodel import Unet, Mnet

import Filtering

import torch
from torch import nn


import DeepLearning
from tensorflow.keras.optimizers import Adam

from Rpkg.Rfund.InputFeature import InputFeature
from Rpkg.Rfund import ReadFile, WriteFile
from Rpkg.Rmodel import Unet, Mnet

from MCtool import RFilter, resultEval
from DeepLearning import save_eval_result

import numpy as np
import cv2
import torch
from transformations import ComposeDouble, FunctionWrapperDouble, create_dense_target, normalize_01
from customdatasets import SegmentationDataSet1
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pathlib
from skimage.transform import resize

#early stopping なし
from unet import UNet
from trainer import Trainer
from sklearn.model_selection import StratifiedKFold, train_test_split
import optuna


def objective(trial):
    N_BLOCK = trial.suggest_int("n_blocks", 2, 6)
    LR = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    OUTPUT_DIR = '20241212-OptunaTest-MConv-1'
    KERNEL_SIZE_CONV = trial.suggest_categorical('kernel_size_conv', [1, 3, 5, 7, 9])
    KERNEL_OUT_SIZE = trial.suggest_categorical('kernel_out_size', [1, 3, 5])
    IN_CHANNEL = 45

    AUGMENTED = False
    AUGMENTATION  =  30


    CROSS_VAL = False
    N_SPLIT = 4


    FUSION_OUT_CHANNEL = 