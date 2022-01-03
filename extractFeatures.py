import os
import librosa
import csv
import os
import time
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import size

DATA_FOLDER = 'data'
DATASET = 'genres'
FEATURES_FILE = os.path.join(DATA_FOLDER, 'features.csv')

# csv headers
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1,21):
	header += ' mfcc'+str(i)
header += ' label'
header =  header.split()


start = time.time()
def extractFeatures():
    isSample = True
    sample = dict()

    # open csv file to write features to
    with open(FEATURES_FILE, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        # extract features for audio clips in each genre
        # genres = "blues classical country disco hiphop jazz metal pop reggae rock".split()
        genres = "blues".split()
        for i, g in enumerate(genres):
            print('extracting features for {genre}({ind})'.format(genre = g, ind = i))
            
            fp = os.path.join(DATA_FOLDER, DATASET, g)
            clips = os.listdir(fp)  # get all audio files from genre
            print('{count} clips'.format(count = len(clips)))

            # extract features for each clips
            for clip in clips:
                song_name = os.path.join(fp, clip)
                try:
                    # sample the clip at a certain sample rate
                    y, sr = librosa.load(song_name, mono=True, duration=30)

                    '''
                        Here, y represents the aplitude of the waveform. The atmospheric variations
                        around the microphone, with respect to time
                    '''

                    # get a sample for later use
                    if isSample:
                        isSample = False
                        sample['song_name'] = song_name
                        sample['y'] = y
                        sample['sr'] = sr
                    
                except:
                    # if there is an error during feature extraction, skip the clip
                    print('exception handling file {clipname}. SKipping.'.format(clipname = song_name))
                    pass

            print()
        
    print('sample audio clip information')
    song_name = sample['song_name']
    y = sample['y']

    print('waveform :', y)

    zero_idx = np.where(y == 0)
    
    plt.figure(figsize=(15, 8))
    plt.plot(list(range(0, y.shape[0])), y, zorder=0)
    plt.autoscale(False)
    plt.scatter(zero_idx, y[zero_idx], c='red', zorder=1)
    plt.title(song_name)
    plt.xlabel('time quanta')
    plt.ylabel('amplitude')
    plt.show()


try:
    extractFeatures()
except FileNotFoundError:
    os.mkdir(DATA_FOLDER)
    extractFeatures()
