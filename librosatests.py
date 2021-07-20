import librosa
import librosa.display
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

file = sys.argv[1]

clip, sr = librosa.load(file)
S = librosa.stft(clip)
S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)
plt.figure()
librosa.display.specshow(S_db)



# cv.namedWindow('image', cv.WINDOW_AUTOSIZE)
# cv.imshow('image', img)
# cv.waitKey()