### SONGBIRDSONGS
## by Kevin Barker

# DESCRIPTION

A backend deep neural network model to differentiate between audio samples of different bird songs. Title taken from the John Adams LP:
https://www.discogs.com/John-Adams-Songbirdsongs/release/1917692
https://youtu.be/Zpj9tGjKVLY

# TECHNOLOGIES USED

Python with TensorFlow, Librosa, Matplotlib and OpenCV libraries.

# METHODS

**addbird.py** takes as an argument a bird species and downloads all the recordings of that species' calls from the xeno-canto.org archive. I used code from Nazariy Tivirikin's xeno-canto-py as a helper function, which I modified to fit my model's architecture. addbird.py downloads the files into a numbered folder in the 'data' directory, generating the number based on existing data loaded. The number, which serves as the label for training, is then mapped to the name of the bird species in a csv file, BIRDS.csv, which is loaded via the load_labels() function. This function takes no argument and returns a dictionary mapping number labels to the bird species associated with it, read in from BIRDS.csv.

**dataimport.py** takes a data directory of directories, each containing a collection of audio files of a specific bird species and labeled with a number, and returns a new directory of similar structure (numbered directories) with each audio file converted to spectrograms in png format. It contains three functions:

1. __main()__ calls the load_data() function on the data directory, delivered as a sys.argv on the command line.

2. __load_data()__ walks the data directory and sends each file to the audio_to_image() function, and saves the retuned object to "{original_filename}.png" in its numbered directory inside a new directory "new_dataset".

3. __audio_to_image()__ takes the audio file and returns a spectrogram of the audio using the librosa library's short-time Fourier transform (STFT) function. This spectrogram is plotted to a canvas with matplotlib and the canvas object is returned.

**birdsongs.py** takes the new_dataset directory, containing labeled directories of spectrogram images, as an argument and trains a 2D convolutional neural network to distinguish between the various categories. It contains three functions:

1. __main()__ calls the load_data() function on the data directory, delivered as a command line argument. It takes the tuple of images and labels returned from that function and splits them into train and test sets. It then calls the get_model() function to create the CNN model, then fitting the model returned from that function using the .fit method on the training data and labels. Finally, it evaluates the model using the .evaluate method on the test set.

2. __load_data()__ ingests the dataset and creates a tuple of images and labels, the latter of which are the numbers of the folder the images are found in. These can be mapped to the bird species later by accessing the BIRDS.csv database. load_data() reads in the images using OpenCV, and then applies greyscale, thresholding, and resizing to optimize them for training accuracy and speed. 

3. __get_model()__ uses the TensorFlow library to build a sequential 2D convolutional neural network with two Convolution/MaxPool layers followed by a series of dense layers, including dropout. The model is compiled with the Adam optimizer, the loss function is sparse categorical cross-entropy, and the metric is accuracy.


