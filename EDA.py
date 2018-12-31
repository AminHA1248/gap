#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:35:32 2018

@author: amin
"""

from gap_db import Gap_Db
import pandas as pd
from plotly.offline import iplot, plot
import plotly.graph_objs as go
import time


#%%============================================================================
# Helpers
# =============================================================================
def plot_ly(df, mode='lines', title=None, xaxis='date', yaxis=None, barchart=False, jupyter=False):
    if title is None:
        title = df.columns[0]
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=xaxis
        ),
        yaxis=dict(
            title=yaxis
        ),
        bargap=0.1,
        bargroupgap=0.1
    )
    if barchart:
        go_plot = go.Bar
    else:
        go_plot = go.Scatter
    data = []
    for col in df.columns:
        trace = go_plot(
            x = df.index,
            y = df[col],
            name = col,
            opacity = 0.5
        )
        data.append(trace)
    fig = go.Figure(data=data, layout=layout)
    if jupyter:
        iplot(fig)
    else:
        plot(fig)
    time.sleep(1)


#%%============================================================================
# Main
# =============================================================================
gap_db = Gap_Db()
db = gap_db._connectiondb()

sql = """
    SELECT family, COUNT(face.image) AS members, income
    FROM faces AS face,
         families AS family
    WHERE face.family=family.id
    GROUP BY family
    ORDER BY income
"""

cursor = db.cursor()
cursor.execute(sql)
columns = ['family_id', 'members', 'income']
df = pd.DataFrame(list(cursor.fetchall()), columns=columns)

df_plot = df[['income', 'members']]
df_plot['income'] = df_plot['income']/1000
plot_ly(df_plot.rolling(20).mean())
