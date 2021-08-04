import librosa
import librosa.display
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FC
import numpy as np
import cv2
from pygame import mixer
import time

file = sys.argv[1]
# mixer.init()
# sound = mixer.Sound(file)
# sound.play()
# while (mixer.get_busy() ):  # wait for the sound to end
#             time.sleep(00.1)
clip, sr = librosa.load(file)
S = librosa.stft(clip)
S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)
fig = plt.figure()
canvas = FC(fig)
librosa.display.specshow(S_db)
canvas.draw()
image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
file_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
cv2.imshow('image', file_img)
cv2.waitKey()