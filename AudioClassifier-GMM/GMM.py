# IMPORTS
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

import pyaudio
import os
import wave
import numpy as np
import python_speech_features as mfccc
import pickle
import time
from scipy.io.wavfile import read

# STORAGE (TRAINING FOLDER)
training_path = 'training_set'
if not os.path.exists(training_path):
    os.mkdir(training_path)
# STORAGE (TESTING FOLDER)
test_path = 'testing_set'
if not os.path.exists(test_path):
    os.mkdir(test_path)

def record_audio_train(Name, samples=10):
    # ENTER THE USER NAME TO STORE THE DATA UNDER
    # Name =(input("Please Enter Your Name:")) 
    # RECORDING 5 TIMES
    Name = Name.lower()
    for count in range (samples):
        # RECORDING CONFIG
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        print("--------record device list--------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
                if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print ("Input Device id", i, "-", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("--------------------------------------------------")
        index = int(input())
        print("recording via index "+str(index))
        stream = audio.open(
            format=FORMAT, 
            channels=CHANNELS,
            rate=RATE, 
            input=True,
            input_device_index = index,
            frames_per_buffer=CHUNK
        )
        
        print("recording started")
        
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        # SAVING AUDIO FILE
        OUTPUT_FILENAME=Name+"-sample"+str(count)+".wav"
        WAVE_OUTPUT_FILENAME=os.path.join("training_set",OUTPUT_FILENAME)
        trainedfilelist = open("training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME+"\n")
        trainedfilelist.close()
        waveFile = wave.open (WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

def calculate_delta(array):
    
    rows,cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first =0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j
            index.append((second, first))
            j+=1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    
    mfcc_feature = mfccc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta))
    print("Features Extracted...")
    return combined


def train_model(Name,samples=10):
    Name = Name.lower()
    
    source = "training_set" #training set path
    dest = "trained_models" #destination path to store trained models
    
    if not os.path.exists(dest):
        os.mkdir(dest)
    train_file = "training_set_addition.txt" #training file wiith sample names
    file_paths = open(train_file,'r')
    
    count = 1
    features = np.asarray(())
    for path in file_paths:

        path = path.strip()
        print(path)
        if Name in path:
            sr,audio = read(os.path.join(source, path))
            print(sr)
            vector = extract_features(audio,sr)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            # Keep Training Until we are trainined on 5 Recordings
            if count == samples:
                gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag' ,n_init = 3)
                gmm.fit(features)
                # dumping the trained gaussian model
                picklefile = path.split("-")[0]+".gmm"
                pickle.dump(gmm,open(os.path.join(dest,picklefile), 'wb'))
                print('+ modeling completed for speaker:' ,picklefile,"with data point = ",features.shape)
                features = np.asarray(())
                count = 0
            count += 1


def record_audio_test():
    test_name = (input("Please Enter a Testing Name:"))
    test_name = test_name.lower()
    # RECORDING CONFIG
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    audio = pyaudio.PyAudio()
    print("--------record device list--------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print ("Input Device id", i, "-", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("--------------------------------------------------")
    index = int(input())
    print("recording via index "+str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = index,
                    frames_per_buffer=CHUNK)
    print("recording started")

    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # SAVING AUDIO FILE
    OUTPUT_FILENAME=test_name+"-test"+".wav"
    WAVE_OUTPUT_FILENAME=os.path.join("testing_set",OUTPUT_FILENAME)
    testfilelist = open("testing_set_addition.txt", 'a')
    testfilelist.write(OUTPUT_FILENAME+"\n")
    testfilelist.close()
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()


def test_model():
    source = "testing_set" # Path of test samples
    modelpath = "trained_models" # path for trained models
    test_file = "testing_set_addition.txt" #test samples name
    file_paths = open(test_file,'r')
    
    gmm_files = [os.path. join(modelpath,fname) for fname in
                    os.listdir(modelpath) if (fname.endswith('.gmm'))]
    print(f"Files: {gmm_files}")
    #Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [os.path.basename(fname).split(".gmm")[0] for fname in gmm_files]
    _s = '\t'.join(speakers)
    print(f"Speakers : {_s}")
    # Read the test directory and get the list of test audio files
    for path in file_paths:
        
        path = path.strip()
        print("="*50)
        print(f"Testing: {path}")
        sr,audio = read(os.path.join(source,path))
        vector = extract_features(audio,sr)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i] #checking with each model by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        try:
            print("\tDetected as -", speakers[winner])
        except:
            print("You are committing a serious felony!!!!!!")
        time.sleep(1.0)
