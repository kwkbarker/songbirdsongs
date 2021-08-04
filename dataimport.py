import matplotlib
from numpy.lib.npyio import save
from birdsongs_nopath import *
import sys
from io import BytesIO


## CREATE SEPARATE PROGRAM TO BUILD IMAGE DATASET


def load_data(directory):
    # loads directory

    fail = False
    errors = 0
    for (root,dirs,files) in os.walk(directory, topdown=True):
        # for dir in dirs:
        #     print(dir)
        #     os.chdir(os.path.join(root, dir))
            # for (root,dirs,files) in os.walk(os.path.join(root, dir), topdown=True):
        # print(files)
        for file in files:
            path = os.path.join(root, file)
            print(path)
            canvas = audio_to_image(path)
            
            # img = cv2.imread(os.path.join(root, filename_img), cv2.IMREAD_UNCHANGED)
            if canvas is not None:
                
                # file_img = cv2.imread(png, cv2.IMREAD_UNCHANGED)
                # cv2.imshow('image', file_img)
                # cv2.waitKey(0)
                # resized = cv2.resize(file_img, (30, 30))
                save_path = os.path.join(os.path.split(root)[1])                
                filename = f"{file}.png"
                try:
                    os.chdir('new_dataset')
                except FileNotFoundError:
                    os.mkdir('new_dataset')
                    os.chdir('new_dataset')
                if save_path:
                    try:
                        os.chdir(save_path)
                    except FileNotFoundError:
                        os.mkdir(save_path)
                        os.chdir(save_path)
                png_output = filename
                canvas.print_png(png_output)
            else:
                print("No image returned.")
                fail = True
                errors += 1
    if fail == False:
        print("Data successfully loaded.")
    else:
        print(f"Something went wrong. There were {errors} errors.")

def audio_to_image(file):
    # converts audio file into spectrogram image.
    # assumes data is an audio file. returns corresponding spectrogram.
    try:
        clip, sr = librosa.load(file)
        S = librosa.stft(clip)
        S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)
        fig = plt.figure()
        canvas = FC(fig)
        librosa.display.specshow(S_db)
        # png_output = 'test.png'
        # image = canvas.print_png(png_output)
    except FileNotFoundError:
        print(f'File not found: {file}')
        return None
    return canvas

def main():

    load_data(sys.argv[1])

if __name__ == "__main__":
    main()