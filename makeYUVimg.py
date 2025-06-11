import cv2
import numpy as np

# 讀取圖片
img = cv2.imread('test2.jpg')
# 轉換為 YUV420 格式
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
# 儲存為原始檔案
yuv.tofile('test2.yuv')