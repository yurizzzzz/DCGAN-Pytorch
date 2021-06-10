import os
import cv2
from tqdm import tqdm

data_list = [file for file in os.listdir('./dataset') if file.endswith('.jpg')]
data_list = tqdm(data_list)

for index in data_list:
    img = cv2.imread('./dataset/' + index)
    img = cv2.resize(img, (64, 64))
    cv2.imwrite('./dataset/' + index, img)
