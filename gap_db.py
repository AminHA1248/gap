import sys
sys.path.append('../common')
import tools
import warnings
from config import Config
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


#%%============================================================================
# Settings
# =============================================================================
config = Config()
warnings.filterwarnings("ignore")
FACE_IMG_DIR = config.FACE_IMG_DIR


#%%============================================================================
# Helpers
# =============================================================================
class Gap_Db():
    def __init__(self):
        self._db = self._connectiondb()

    def _connectiondb(self, database='MYSQLDB'):
        config = tools.read_config_file()
        params = tools.get_mysql_connection_parameters(config, database)
        db = tools.connect_mysql(params)
        return db
    
    def get_country_code(self, country, region, lat, lng):
        cursor = self._db.cursor()

        # update table with new value if country doesn't exist
        sql = """INSERT IGNORE INTO countries (name, region, lat, lng) VALUES (%s,%s,%s,%s)"""
        cursor.execute(sql, [country, region, lat, lng])
        self._db.commit()
        
        sql = """SELECT id FROM countries WHERE name="%s" """%country
        cursor.execute(sql)
        ret_val = cursor.fetchone()[0]
        return ret_val

    def update_countries(self, df_countries):
        # update database
        cursor = self._db.cursor()
        if df_countries.shape[1] == 2:
            sql = "INSERT IGNORE INTO countries (name, region) VALUES (%s,%s)"
            cursor.executemany(sql, df_countries.values.tolist())
            self._db.commit()
        elif df_countries.shape[1] == 4:
            sql = "INSERT IGNORE INTO countries (name, region, lat, lng) VALUES (%s,%s,%s,%s)"
            cursor.executemany(sql, df_countries.values.tolist())
            self._db.commit()
        else:
            raise ValueError("countries must contain name, region, [lat], [lng]")

    def update_families(self, image, country_code, income, incomeQuality):
        cursor = self._db.cursor()
        sql = """
        INSERT IGNORE INTO families (`image`, `country`, `income`, `income_quality`) VALUES (%s,%s,%s,%s)
        """
        cursor.execute(sql, [image, country_code, income, incomeQuality])
        self._db.commit()

    def update_family_country(self, country, region, income, incomeQuality, image, lat, lng):
        country_code = self.get_country_code(country, region, lat, lng)
        self.update_families(image, country_code, income, incomeQuality)

    def get_family_id(self, image):
        cursor = self._db.cursor()

        sql = """SELECT id FROM families WHERE image="%s" """%image
        cursor.execute(sql)
        ret_val = cursor.fetchone()[0]
        return ret_val

    def update_faces(self, family, image, age, gender):
        cursor = self._db.cursor()
        sql = """
        INSERT IGNORE INTO faces (`family`, `image`, `age`, `gender`) VALUES (%s,%s,%s,%s)
        """
        cursor.execute(sql, [family, image, age, gender])
        self._db.commit()

    def is_face_done(self, family_id):
        cursor = self._db.cursor()
        sql = """SELECT * FROM faces WHERE family=%s"""%family_id
        cursor.execute(sql)
        res = cursor.fetchall()
        if len(res) > 0:
            is_face_done = True
        else:
            is_face_done = False
        return is_face_done

    def load_data(self, shuffle_data=False):
        cursor = self._db.cursor()
        sql = """SELECT face.image, family.income
                 FROM faces AS face,
                      families AS family
                WHERE family.id=face.family"""
        cursor.execute(sql)
        columns = ['image', 'income']
        df = pd.DataFrame(list(cursor.fetchall()), columns=columns)
        labels = df['income'].values

        images = []
        for image in df['image']:
            fn = os.path.join(FACE_IMG_DIR, image+'.jpg')
            images.append(cv2.imread(fn))
        images = np.array(images)

        if shuffle_data:
            images, labels = shuffle(images, labels, random_state=0)

        return images, labels