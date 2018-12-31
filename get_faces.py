#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:04:17 2018

@author: amin
"""
import glob
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import os
from pyagender import PyAgender
from gap_db import Gap_Db
import imutils


#%%============================================================================
# Settings
# =============================================================================
FAMILY_IMG_DIR = "/home/amin/projects/gap/families/*.jpg"
FACE_IMG_DIR = "/home/amin/projects/gap/faces"


#%%============================================================================
# Helpers
# =============================================================================
class Image_Utils():
    def show_key_points(self, img, faces, width=10):
        image = img.copy()
        for face in faces:
            key_points = face['keypoints'].values()
            for key_point in key_points:
                cv2.circle(image, (key_point[0], key_point[1]), width, \
                           (0, 0, 255))
        cv2.imshow('face', image)
    
    def show_face_patches(self, img, faces, show=True):
        counter = 0
        face_patches = []
        for face in faces:
            counter += 1
            rect = face['box']
            face_image = img[rect[1]:rect[1] + rect[3], rect[0]: rect[0] + \
                             rect[2]]
            face_patches.append(face_image)
            if show:
                cv2.imshow('face%s'%counter, face_image)
        return face_patches
    
    def show_bounding_boxes(self, img, faces, width=10):
        image = img.copy()
        rects = [face['box'] for face in faces]
        for rect in rects:
            cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], \
                          rect[1] + rect[3]), (0, 0, 255), width)
        cv2.imshow('bounding boxes', image)

    def squarify_box(self, faces):
        for face in faces:
            face['center'] = (int(face['box'][0] + face['box'][2]/2), 
                              int(face['box'][1] + face['box'][3]/2))
            sq_size = int(max(face['box'][2:])*1.8)
            face['box'] = [max(0, int(face['center'][0] - sq_size/2)),
                           max(0, int(face['center'][1] - sq_size/2)),
                           sq_size,
                           sq_size]

class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=317):
        # store desired output left
        # eye position, and desired output face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, faces, img):
        faces_aligned = []
        for face in faces:
            # convert the landmark (x, y)-coordinates to a NumPy array
            key_points = face['keypoints']
    
            # compute the center of mass for each eye
            leftEyeCenter = key_points['left_eye']
            rightEyeCenter = key_points['right_eye']
    
            # compute the angle between the eye centroids
            dY = leftEyeCenter[1] - rightEyeCenter[1]
            dX = leftEyeCenter[0] - rightEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
    
            # compute the desired right eye x-coordinate based on the
            # desired x-coordinate of the left eye
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
    
            # determine the scale of the new resulting img by taking
            # the ratio of the distance between eyes in the *current*
            # img to the ratio of distance between eyes in the
            # *desired* img
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist
    
            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input img
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                          (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    
            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
    
            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])
    
            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
            faces_aligned.append(output)

            # return the aligned face
        return faces_aligned



#%%============================================================================
# detect faces MTCNN
# =============================================================================
detector = MTCNN()
fa = FaceAligner()
image_utils = Image_Utils()
agender = PyAgender()
gap_db = Gap_Db()

# check if face directory exists
if not os.path.exists(FACE_IMG_DIR):
    os.system("mkdir -p %s"%FACE_IMG_DIR)

fns = glob.glob(FAMILY_IMG_DIR)
counter = 0
# TODO: edit line
for fn in fns[:10]:
    print("\nprocessing file %s... File %s of %s"%(fn, counter, len(fns)))
    counter += 1

    # check if faces already detected for the family
    family_image_name = os.path.basename(fn).split('.')[0]
    family_id = gap_db.get_family_id(family_image_name)
    if gap_db.is_face_done(family_id):
        print("Faces already detected for the family! Moving on...\n")
        continue
    else:
        # if faces not detected for the family yet, remove previous faces for 
        # the family
        os.system("rm -rf %s"%(os.path.join(FACE_IMG_DIR, '%s*'%family_image_name)))
        

    img_original = cv2.imread(fn)
    if img_original.shape[0] >= img_original.shape[1]:
        img = imutils.resize(img_original, height=2000)
    else:
        img = imutils.resize(img_original, width=2000)

    faces = detector.detect_faces(img)
    image_utils.squarify_box(faces)
    # TODO: Remove line
    #image_utils.show_key_points(img, faces)
    face_patches = image_utils.show_face_patches(img, faces, show=True)

    # align faces
    faces_aligned = fa.align(faces, img)
    assert len(faces_aligned) == len(face_patches)

    # detect age and gender
    ages = []
    genders = []
    for face in face_patches:
        g, a = agender.gender_age(face)
        gender = ['female' if g>0.5 else 'male'][0]
        age = int(a)
        ages.append(age)
        genders.append(gender)
        print("gender: %s, age: %i"%(gender, age))

    # save faces
    face_images = []
    face_no = 0
    for face in faces_aligned:
        face_fn = family_image_name + '_' + str(face_no) + '.jpg'
        face_images.append(face_fn[:-4]) # remove 'extension'
        face_f_fn = os.path.join(FACE_IMG_DIR, face_fn)
        cv2.imwrite(face_f_fn, face)
        face_no += 1

    ## update database
    # find family_id
    for image, age, gender in zip(face_images, ages, genders):
        gap_db.update_faces(family_id, image, age, gender)
    