#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:10:20 2018

@author: amin
"""
#%%============================================================================
# Imports
# =============================================================================
import os
import urllib
import requests
import pandas as pd
from gap_db import Gap_Db
from bs4 import BeautifulSoup as bs
import re
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
from pyagender import PyAgender
import imutils
import time
from config import Config
from image_utils import Image_Utils
import glob


#%%============================================================================
# Settings
# =============================================================================
config = Config()
FAMILY_IMG_DIR = config.FAMILY_IMG_DIR
FACE_IMG_DIR = config.FACE_IMG_DIR
FACE_WIDTH = Config.FACE_WIDTH

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)


#%%============================================================================
# Helpers
# =============================================================================
class Gap_Scrapper():
    def __init__(self):
        self._gap_db = Gap_Db()

    def get_url_from_href(self, url, href):
        """ This function returns url from href"""
        new_url = urllib.parse.urljoin(url, href)
        return new_url
    
    def dl_jpg_from_url(self, img_url, dst_fn):
        dirname = os.path.dirname(dst_fn)
        if not os.path.exists(dirname):
            print("creating directory '%s'..."%dirname)
            os.makedirs(dirname)
        if not os.path.exists(dst_fn):
            print("downloading photo '%s'..."%dst_fn)
            os.system("""wget -O "%s" "%s" """%(dst_fn, img_url))
    
    def http_request(self, url):
        response = requests.get(url=url, timeout=(3.05, None))
        res_dict = response.json()
        return res_dict

    def dl_family_photos(self, url, directory):
        """This function gets a url as input and returns all hyperlinks on the page"""
        res = self.http_request(url)
        families = [i for i in res['data']['streetPlaces']]
        for family in families:
            try:
                image, country, region, income, incomeQuality, lat, lng, link = \
                family['image'], family['country'], family['region'], \
                family['income'], family['incomeQuality'], \
                family['lat'], family['lng'], family['background']
                
                # download and save photo
                dst_fn = os.path.join(directory, image+'.jpg')
                #link = link.replace("480x480", "original")
                img_url = self.get_url_from_href("https://", link)
                self.dl_jpg_from_url(img_url, dst_fn)
                
                # update database
                self._gap_db.update_family_country(country, region, income, incomeQuality, image, lat, lng)
    
            except Exception as e:
                print(e)
                pass
    
    def get_average_country_income(self, url):
        headers = {'Upgrade-Insecure-Requests': '1', \
                   'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"}
        response = requests.get(url=url, timeout=(5, None), headers=headers)
        html = response.text
        soup = bs(html, "lxml")
        countries = [i.string for i in soup.findAll("a", {"class" : re.compile("fl_.*")})]
        incomes = [int(i.string.split()[0].replace(',', '')) for i in soup.findAll("td", {"class" : "right nowrap"})][1::2]
        df_income = pd.DataFrame(np.array([countries, incomes]).T, columns=['name', 'monthly_income'])
        self._gap_db.update_incomes(df_income)


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=FACE_WIDTH, desiredFaceHeight=None):
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


class Age_Gender_Detect():
    def __init__(self):
        self._detector = MTCNN()
        self._fa = FaceAligner()
        self._image_utils = Image_Utils()
        self._agender = PyAgender()
        self._gap_db = Gap_Db()


    def detect_face_age_gender(self, fn, debug_mode=False):
        print("\nprocessing file %s..."%fn)
    
        # check if faces already detected for the family
        family_image_name = os.path.basename(fn).split('.')[0]
        family_id = self._gap_db.get_family_id(family_image_name)
        if self._gap_db.is_face_done(family_id):
            print("Faces already detected for the family! Moving on...\n")
            return
        else:
            # if faces not detected for the family yet, remove previous faces for 
            # the family
            os.system("rm -rf %s"%(os.path.join(FACE_IMG_DIR, '%s*'%family_image_name)))
            
    
        img_original = cv2.imread(fn)
        if img_original.shape[0] >= img_original.shape[1]:
            img = imutils.resize(img_original, height=2000)
        else:
            img = imutils.resize(img_original, width=2000)

        faces = self._detector.detect_faces(img)
        self._image_utils.squarify_box(faces)
        
        if debug_mode:
            self._image_utils.show_key_points(img, faces)
        face_patches = self._image_utils.show_face_patches(img, faces, show=False)

        # align faces
        faces_aligned = self._fa.align(faces, img)
        assert len(faces_aligned) == len(face_patches)
    
        # detect age and gender
        ages = []
        genders = []
        for face in face_patches:
            g, a = self._agender.gender_age(face)
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
        for image, age, gender in zip(face_images, ages, genders):
            self._gap_db.update_faces(family_id, image, age, gender)


#%%============================================================================
# Download family photos
# =============================================================================
# instantiate
gap_scrapper = Gap_Scrapper()

# low-res images
url = "https://consumer-api-prod.dollarstreet.org/v1/things?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month&resolution=480x480"

# menu items
url = "https://consumer-api-prod.dollarstreet.org/v1/things-filter?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month"

# high res images
url = "https://consumer-api-prod.dollarstreet.org/v1/things?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month&resolution=original"
gap_scrapper.dl_family_photos(url, directory="/home/amin/projects/gap/families")


# get country income
url = "https://www.worlddata.info/average-income.php"
gap_scrapper.get_average_country_income(url)


#%%============================================================================
# detect faces MTCNN
# =============================================================================
tic = time.time()

age_gender_detect = Age_Gender_Detect()

# check if face directory exists
if not os.path.exists(FACE_IMG_DIR):
    os.system("mkdir -p %s"%FACE_IMG_DIR)

fns = glob.glob(FAMILY_IMG_DIR)
for fn in fns:
    age_gender_detect.detect_face_age_gender(fn)

print("execution succesfully finished in %.4s sec!"%int(time.time()-tic))