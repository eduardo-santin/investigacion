from random import sample, shuffle
import glob
import sys
import pandas as pd 
import os
import numpy as np
#import skimage.io as io
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float32_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
def load_audio(audio):
    raw_audio = tf.io.read_file(audio)
    raw_audio = tf.audio.decode_wav(
                    raw_audio,
                    desired_channels=1,  # mono
                    desired_samples= 48000 * 5)
    
    # x = raw_audio.sample_rate.numpy()
    return raw_audio.audio

def load_labels(labels):
    string_tensor = tf.constant(labels)
    return string_tensor



def createDataRecord(out_filename, addrs, labels, especie_id, punto_id, length, matrix, rate, is_tp):
    # open the TFRecords file
    writer = tf.io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        #if i is a multiple of 100 then print the number
        if not i % 100:
            print('Writing to tfrec\t' + str(i) )
            sys.stdout.flush()
        # Load the image
        wav = load_audio(addrs[i])

        
        

        if wav is None:
            continue

        # Create a feature
        # print(_bytes_feature(b'test_string'))
        # print(_bytes_feature(u'test_bytes'.encode('utf-8')))
        feature = {
            'label': _int64_feature(labels[i]),
            'audio_wav': _float32_feature(wav.numpy().flatten().tolist()),
            'species_id': _int64_feature(especie_id[i]),
            'punto_id': _int64_feature(punto_id[i]),
            'length': _int64_feature(length[i]),
            'matrix': _float32_feature(matrix[i]),
            'rate': _float32_feature(rate[i]),
            'is_tp': _int64_feature(is_tp[i])
            
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()


#read the csv
csv = pd.read_csv(os.path.abspath('./arbimon.csv'))
#print(csv)
#audio files is a list of all the paths to the audio files found in the csv
audio_files = csv['archivo']
audio_paths = audio_files.tolist()

recording_id = csv['Unnamed: 0']
especie_id = csv['especie_id']
punto_id = csv['punto_id']
length = csv['length']
matrix = csv['matrix']
rate = csv['rate']

recording_id = recording_id.tolist()
especie_id = especie_id.tolist()
punto_id = punto_id.tolist()
length = length.tolist()
matrix = matrix.tolist()
rate = rate.tolist()
is_tp = 0
#list 
labels = []

print(len(audio_paths))
for i in range(666):
    audio_paths[i] = os.path.join(os.path.abspath('./grabaciones5seg'), audio_paths[i])
    #convert scalar to vector
    recording_id[i] = [recording_id[i]]
    recording_id[i] = np.array(recording_id[i])
    especie_id[i] = [especie_id[i]]
    especie_id[i] = np.array(especie_id[i])
    punto_id[i] = [punto_id[i]]
    punto_id[i] = np.array(punto_id[i])
    length[i] = [length[i]]
    length[i] = np.array(length[i])
    matrix[i] = [matrix[i]]
    matrix[i] = np.array(matrix[i])
    rate[i] = [rate[i]]
    rate[i] = np.array(rate[i])
    is_tp = [[1]]
    is_tp = np.array(is_tp)

    

createDataRecord('finalonemaybe.tfrec', recording_id, audio_paths, label_info )