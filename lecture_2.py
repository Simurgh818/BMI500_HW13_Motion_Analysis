#!/usr/bin/env python
# coding: utf-8

# # Lab Part 2
# 
# 1. Write a function to calculate included angle from two vectors
# 2. Identify gait speed vs. time
# 3. Identify ankle height vs. time (R, L)
# 4. Identify knee angle vs. time (R, L)
# 5. Set up pandas dataframe with outcome variables
# 6. Code rows in dataframe by participant
# 7. Plot the data

# In[1]:


# imports etc.
import validators
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()


# # Base functions for interacting with .json files
# 
# The following routines allow you to load and make a basic plot of a `.json` file from OpenPose.

# In[2]:


def extract_kp_from_resource(f,person_id = -1):
    """
    Extract keypoints from resource (file or url) as a numpy vector
    """

    # load json from remote if it is an url; otherwise load as file
    if validators.url(f):
        json_temp = json.load(urllib.request.urlopen(u))['people']
    if Path(f).exists():
        json_temp = json.load(open(f))['people']
    
    try:
        # extract the keypoints of the person specified by person_id; default is last person identified
        keypoints = np.array(extract_kp_from_json(json_temp)["pose_keypoints_2d"][person_id]).astype('float')
    except:
        keypoints = np.empty((75,))
        keypoints[:] = np.NaN
    
    # set missing points (imputed as 0) to nan so that they are not plotted
    keypoints[keypoints==0] = np.nan
    
    return keypoints

def extract_kp_from_json(json_people):
    """
    subfunction for extract_kp_from_resource
    """
    person_id = []
    pose_keypoints_2d = []
    for i in range(0,len(json_people)):
        person_id.append(json_people[i]["person_id"])
        pose_keypoints_2d.append(json_people[i]["pose_keypoints_2d"])
    # return a dict
    return {'person_id': person_id, 'pose_keypoints_2d': pose_keypoints_2d}

def convert_kp_to_df(keypoints):
    """
    reshape keypoint vector to dataframe
    """
    # reshape to 25 X 3; the coordinates are x, y, confidence in estimate
    kin = keypoints.reshape((-1,3))
        
    # create a dataframe
    df = pd.DataFrame({'keypoint': ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"], 'x': kin[:,0], 'y': kin[:,1], 'confidence': kin[:,2]})

    return df.set_index('keypoint')

def plot_body25_df(df):
    """
    plot a dataframe corresponding to body25 coordinates
    """
    segments = [['Neck', 'REar', 'LEar', 'Neck'],
        ['Nose', 'REye', 'LEye', 'Nose'],
        ['RShoulder', 'Neck', 'LShoulder','RShoulder'],
        ["LShoulder", "LElbow", "LWrist"],
        ["RShoulder", "RElbow", "RWrist"],
        ['RShoulder', 'RHip', 'LHip','LShoulder','RShoulder'],
        ['LHip', 'MidHip', 'RHip'],
        ["LHip", "LKnee", "LAnkle"],
        ["RHip", "RKnee", "RAnkle"],
        ['LAnkle', 'LHeel', 'LBigToe', 'LSmallToe', 'LAnkle'],
        ['RAnkle', 'RHeel', 'RBigToe', 'RSmallToe', 'RAnkle']]
    
    fig, ax = plt.subplots()
    ax.set(xlim=[0, 1920], ylim=[1080, 0], xlabel='X', ylabel='Y')  # setting the correct parameters from the slides
    [sns.lineplot(data=df_f.loc[s], x = "x", y = "y", ax = ax) for s in segments]


# # Sample load and plot
# 
# Here is a sample of how to load/plot from a local file or online resource

# In[9]:


u = "https://jlucasmckay.bmi.emory.edu/global/bmi500/keypoints.json"
f = "keypoints.json"

kp_u = extract_kp_from_resource(u)
kp_f = extract_kp_from_resource(f)

df_u = convert_kp_to_df(kp_u)
df_f = convert_kp_to_df(kp_f)

plot_body25_df(df_f)


# # Functions to calculate outcomes
# 
# To complete the lab, you must fill in the following function prototypes:
# 
# ```python
# def calculate_angle(v1,v2):
#     """
#     return the angle (in degrees) between two vectors v1 and v2.
#     """
#     
# def df_to_outcomes(d):
#     """
#     return a dataframe with kinematic outcomes derived from a single body25 dataframe.
#     """
#         
# def calc_outcomes(f):
#     """
#     calculate outcomes from a file or other resource
#     return as a dataframe in standard format
#     """
# ```

# In[7]:

# ADD FUNCTIONS HERE


# # Loop over files
# 
# Next, you must loop over files in the `json/` directory and concatenate all of the outcomes into a pandas dataframe called `outcomes`.

# In[10]:

# ADD FUNCTIONS HERE

# # Write to file
# 
# Write the data to a `csv` file

# In[11]:


# write to file
outcomes.to_csv("outcomes.csv",index=False)


# # Plot and save
# 
# ## Normal gait (right knee angle, first 200 frames)

# In[12]:

# ADD PLOT CODE

# ## Vaulting gait (right knee angle, first 200 frames)

# In[13]:

# ADD PLOT CODE

# # What to turn in
# 
# 1. A completed notebook with documented code
# 2. The aggregated csv file

