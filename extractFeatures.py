import os
import librosa
from numba.cuda.simulator.kernelapi import FakeCUDALocal
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import sys
import numpy as np

DATA_FOLDER = 'data'
DATASET = 'genres'
FEATURES_FILE = os.path.join(DATA_FOLDER, 'features.csv')

# csv header
HEADER = 'filename' 
for i in range(12):
    HEADER += ' stft'+str(i+1)
HEADER += ' spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1,21):
	HEADER += ' mfcc' + str(i)
HEADER += ' label'
HEADER =  HEADER.split()

def printSample(song_name, y, sr, stft, cent, bw, ro, zcr, mfc, g):
    sout = sys.stdout

    with open('sample.txt', 'w') as f:
        sys.stdout = f

        print('SAMPLE AUDIO CLIP FEATURES')
        print(song_name, g)
        print('-' * 100)

        print('+ Waveform {shape} :'.format(shape = y.shape), y, end='\n\n')
        print('+ Sampling rate :', sr, end='\n\n')
        print('+ Short time fourier transform {shape} :'.format(shape = stft.shape), stft, end='\n\n')
        print('+ Spectral centroid {shape} :'.format(shape = cent.shape), cent, end='\n\n')
        print('+ Spectral bandwidth {shape} :'.format(shape = bw.shape), bw, end='\n\n')
        print('+ Spectral rolloff {shape} :'.format(shape = ro.shape), ro, end='\n\n')
        print('+ Spectral zero crossing rate {shape} :'.format(shape = zcr.shape), zcr, end='\n\n')
        print('+ Mel Frequency Cepstral Coefficents {shape} :'.format(shape = mfc.shape), zcr, end='\n\n')

        # indices where the amplitude is zero
        zero_idx = np.where(y == 0)
        
        plt.figure(figsize=(15, 8))
        plt.plot(list(range(0, y.shape[0])), y, zorder=0)
        plt.autoscale(False)
        plt.scatter(zero_idx, y[zero_idx], c='red', zorder=1)
        plt.title(song_name + ' : ' + g)
        plt.xlabel('time quanta')
        plt.ylabel('amplitude')
        plt.savefig('sample_audio_clip.png')

        print('-' * 100)
        sys.stdout = sout

def extract(filepath, genre, df):
    global HEADER

    if df.empty:
        df = pd.DataFrame(columns=HEADER)
        print(df.columns)

    # sample the clip at a certain sample rate
    y, sr = librosa.load(filepath, mono=True, duration=30)
    '''
        Here, y represents the aplitude of the waveform. The atmospheric variations
        around the microphone, with respect to time
    '''

    stft = librosa.feature.chroma_stft(y, sr)
    stft_mean = np.mean(stft, axis = 1) # mean for each bin

    cent = librosa.feature.spectral_centroid(y, sr)
    cent_mean = np.mean(cent)

    bw = librosa.feature.spectral_bandwidth(y, sr)
    bw_mean = np.mean(bw)
    
    ro = librosa.feature.spectral_rolloff(y, sr)
    ro_mean = np.mean(ro)

    zcr = librosa.feature.zero_crossing_rate(y, sr)
    zcr_mean = np.mean(zcr)

    mfc = librosa.feature.mfcc(y, sr)
    mfc_mean = np.mean(mfc, axis=1)

    # append to csv
    feature_list = [filepath] + stft_mean.tolist() + [cent_mean, bw_mean, ro_mean, zcr_mean] + mfc_mean.tolist() + [genre]
    feature_list = pd.Series(feature_list, index = df.columns)
    df = df.append(feature_list, ignore_index=True)         

    return df, [filepath, y, sr, stft, cent, bw, ro, zcr, mfc, genre]

def extractFeatures():
    start = time.time()

    # empty dataframe
    df = pd.DataFrame()

    sample = True

    # extract features for audio clips in each genre
    genres = "blues classical country disco hiphop jazz metal pop reggae rock".split()
    # genres = "blues".split()
    
    for i, g in enumerate(genres):
        print('\nextracting features for {genre}({ind})'.format(genre = g, ind = i))
        
        fp = os.path.join(DATA_FOLDER, DATASET, g)
        clips = os.listdir(fp)  # get all audio files from genre
        print('{count} clips'.format(count = len(clips)))

        # extract features for each clip
        for j, clip in enumerate(clips):
            song_name = os.path.join(fp, clip)
            print("\t {song}({ind})".format(song = song_name, ind = j))

            try:
                df, _ = extract(song_name, g, df)
                
                if sample == True:
                    sample = False
                    printSample(*_)

            except Exception as e:
                print(e)
                # if there is an error during feature extraction, skip the clip
                print('exception handling file {clipname}. SKipping.'.format(clipname = song_name))
                pass
            
        print(df.sample(2))     
        print()
    
    # feature extraction complete
    print('that took {t}s'.format(t = time.time() - start))
    return df

# extract features and store
try:
    df = extractFeatures()
    print(df)
    df.to_csv(FEATURES_FILE, index=False) 

except FileNotFoundError:
    os.mkdir(DATA_FOLDER)
    extractFeatures()
