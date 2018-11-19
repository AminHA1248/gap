#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:10:20 2018

@author: amin
"""

import os
import urllib
import requests
from tqdm import tqdm


#%%
def get_url_from_href(url, href):
    """ This function returns url from href"""
    new_url = urllib.parse.urljoin(url, href)
    return new_url

def dl_jpg_from_url(img_url, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.system("wget -P /home/amin/projects/gapminder/%s '%s'"%(directory, img_url))

def dl_all_links(url, directory):
    """This function gets a url as input and returns all hyperlinks on the page"""
    links = [i['background'] for i in requests.get(url).json()['data']['streetPlaces']]
    for link in tqdm(links):
        try:
            img_url = get_url_from_href("https://", link)
            # print("downloading %s..."%img_url)
            dl_jpg_from_url(img_url, directory)
        except:
            pass


#%%
url = "https://consumer-api-prod.dollarstreet.org/v1/things?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month&resolution=original"
dl_all_links(url, "families")



#%%============================================================================
# Tmporary
# =============================================================================
# low-res images
url = "https://consumer-api-prod.dollarstreet.org/v1/things?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month&resolution=480x480"

# list of countries
url = "https://consumer-api-prod.dollarstreet.org/v1/countries-filter?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month"

# menu items
url = "https://consumer-api-prod.dollarstreet.org/v1/things-filter?lang=en&thing=Families&countries=World&regions=World&zoom=4&mobileZoom=4&row=1&lowIncome=13&highIncome=10813&currency=usd&time=month"