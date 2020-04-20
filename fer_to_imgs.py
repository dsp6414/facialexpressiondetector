"""Module to convert FER CSV dataset to PNGs"""

import pandas as pd
import numpy as np
import os
from PIL import Image

# Original Categories: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# Remove Disgust, Fear, and Surprise -> Low datapoints and reduce categories
# Adjusted Categories: 0=Angry, 3=Happy, 4=Sad, 6=Neutral

def convert_pixel_str_to_img(pixel_str):
  """Convert pixel string into PIL Image object"""
  pixel_arr = np.fromstring(pixel_str, dtype=int, sep=' ').reshape(48,48)
  img = Image.fromarray(pixel_arr).convert('RGB')
  return img

def convert_fer_to_img():
  """Convert FER data to PNGs"""
  FER_PATH = 'train.csv'
  fer_raw_df = pd.read_csv(FER_PATH)

  emotion_dict = {0: 'angry', 3:'happy', 4:'sad', 6:'neutral' }
  fer_raw_df = fer_raw_df.loc[fer_raw_df.emotion.isin([0,3,4,6])].reset_index(drop=True)

  root_data_dir = 'data'
  if(not os.path.isdir(root_data_dir)):
    os.mkdir(root_data_dir)

  fer_raw_length = len(fer_raw_df)
  for index in range(fer_raw_length):
    print(f'??Progress: {index}/{fer_raw_length}???')
    current_row = fer_raw_df.iloc[index]

    img = convert_pixel_str_to_img(current_row.pixels)

    emotion_str = emotion_dict[current_row.emotion]
    emotion_path = os.path.join(root_data_dir, emotion_str)

    if(not os.path.isdir(emotion_path)):
      os.mkdir(emotion_path)

    emotion_filename = f'{emotion_str}_{index}.png'
    img.save(os.path.join(emotion_path, emotion_filename))

