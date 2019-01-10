#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:56:42 2019

@author: amin
"""

from __future__ import print_function, division
from keras.layers import Dense, Flatten, Dropout
from keras.layers import BatchNormalization, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from config import Config
from gap_db import Gap_Db
import datetime as dt
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import face_recognition
from image_utils import Image_Utils
from tqdm import tqdm
import time


#%%============================================================================
# Settings
# =============================================================================
config = Config()
FACE_IMG_DIR = config.FACE_IMG_DIR
FACE_WIDTH = Config.FACE_WIDTH
TRAIN_RATIO = Config.TRAIN_RATIO


#%%============================================================================
# Helpers
# =============================================================================
class IncomePred():
    def __init__(self):
        # Input shape
        self.img_rows = FACE_WIDTH
        self.img_cols = FACE_WIDTH
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self._gap_db = Gap_Db()

        # Build and compile the discriminator
        self.model = self.build_model()


    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        optimizer = Adam(0.0002, 0.5)

        model.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mse'])

        model.summary()

        return model


    def train(self, epochs, batch_size=128):
        # Load the dataset
        (X, y) = self._gap_db.load_data(False)

        print("converting images to embeddins...")
        tic = time.time()
        X_enc = []
        y_enc = []
        for cv_img, income in tqdm(zip(X, y)):
            dlib_image = cv_img[:,:,[2,1,0]]
            face_encs = face_recognition.face_encodings(dlib_image)
            if len(face_encs)>0:
                X_enc.append(face_encs[0])
                y_enc.append(income)
        X_enc = np.array(X_enc)
        y_enc = np.array(y_enc).reshape(-1,1)
        print("Done in %.4s sec!"%int(time.time()-tic))

        # startr training
        from sklearn.ensemble import RandomForestRegressor
        regr = RandomForestRegressor(n_estimators=8, n_jobs=-1, oob_score=True, random_state=124)
        regr.fit(X_enc, y_enc)
        print(regr.oob_score_)
        print(regr.predict(X_test))
        print(regr.score(X_train, y_train))
        print(regr.score(X_test, y_test))
        print(regr.feature_importances_)


        X = (X.astype(np.float32) - 127.5) / 127.5
        y = np.log(y).reshape(-1, 1)

        len_train = int(TRAIN_RATIO * len(X))
        X_train = X[0:len_train]
        y_train = y[:len_train]
        X_test = X[len_train:]
        y_test = y[len_train:]

        print("\n##################################")
        print("null Hypothesis mse for the train set is: %.5s"%mean_squared_error(np.ones(len(y_train))*y_train.mean(), y_train))
        print("null Hypothesis mse for the test set: %.5s"%mean_squared_error(np.ones(len(y_test))*y_test.mean(), y_test))
        print("##################################")


        logdir = "/tmp/tensorboard/" + dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "/"
        filepath="income_weights.best.hdf5"
        call_back = [TensorBoard(log_dir=logdir), 
                     ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                     save_best_only=True, mode='min')]
        history = self.model.fit(X_train, y_train, 
                    epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_test, y_test), 
                    verbose=1, shuffle=True, callbacks=call_back)
        return history


#%%============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    income_pred = IncomePred()
    income_pred.train(epochs=10, batch_size=124)
