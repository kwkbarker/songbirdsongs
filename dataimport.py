from birdsongs import *


## CREATE SEPARATE PROGRAM TO BUILD IMAGE DATASET


def load_data(directory):
    # loads directory. assumes directory of audio files. returns new directory of png files as "{original filename.extension}.png".
    
    # get working dir  
    wd = os.getcwd()

    # set up error catching
    fail = False
    errors = 0

    # walk through files
    for (root,dirs,files) in os.walk(directory, topdown=True):        
        for file in files:
            if file == '.DS_Store':
                continue
            # send path to spectrogram function
            path = os.path.join(root, file)
            canvas = audio_to_image(path)
            # if spectrogram
            if canvas is not None:
                # set path/name for new file
                save_path = os.path.join(os.path.split(root)[1])                
                filename = f"{file}.png"
                # change working directory into new file location
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
                # save file
                png_output = filename
                canvas.print_png(png_output)
                plt.clf()
                plt.close()
                print(filename)
                
                # restore working dir
                os.chdir(wd)
            
            # catch errors if canvas is None
            else:
                print("No image returned.")
                fail = True
                errors += 1
    
    # print success/fail message
    if fail == False:
        print("Data successfully loaded.")
    else:
        print(f"Something went wrong. There were {errors} errors.")

def audio_to_image(file):
    # converts audio file into spectrogram image.
    # assumes data is an audio file. returns corresponding spectrogram.

    # load file into librosa
    clip, sr = librosa.load(file)
    S = librosa.stft(clip)
    S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)
    
    # initialize matplotlab
    fig = plt.figure()
    canvas = FC(fig)
    librosa.display.specshow(S_db)
    return canvas

def main():

    load_data(sys.argv[1])

if __name__ == "__main__":
    main()