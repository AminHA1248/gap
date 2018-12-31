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


#%%============================================================================
# Settings
# =============================================================================
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)


#%%============================================================================
# Helpers
# =============================================================================
class Gap_Scrapper():
    def __init__(slef):
        pass

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
        #with open('families.pkl', 'wb') as fp:
        #    pickle.dump(families, fp, protocol=-1)
        #with open('families.pkl', 'rb') as fp:
        #    families = pickle.load(fp)
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
                gap_db.update_family_country(country, region, income, incomeQuality, image, lat, lng)
    
            except Exception as e:
                print(e)
                pass
    


#%%============================================================================
# Download family photoes
# =============================================================================
# instantiate
gap_db = Gap_Db()
gap_scrapper = Gap_Scrapper()

# low-res images
url = "https://consumer-api-prod.dollarstreet.org/v1/things?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month&resolution=480x480"

# menu items
url = "https://consumer-api-prod.dollarstreet.org/v1/things-filter?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month"

# high res images
url = "https://consumer-api-prod.dollarstreet.org/v1/things?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month&resolution=original"
gap_scrapper.dl_family_photos(url, directory="/home/amin/projects/gap/families")
