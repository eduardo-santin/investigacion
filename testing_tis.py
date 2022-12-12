# from scipy.io import wavfile
# from scipy.fftpack import fft
# import numpy as np
# import scipy as sp

# # Read the wav file and get the sample rate and audio data
# audio_file_name = 'audios/000316da7.wav'
# # remove file extension
# audio_file_name = audio_file_name.split('.')[0]
# sample_rate, audio_data = wavfile.read('audios/000316da7.wav')


# #specify the frequency ranges for the filter 
# # starting with 512 scaling up to 24,000
# low_freq1 = 512
# high_freq1 = 5480
# low_freq2 = 5480
# high_freq2 = 10800
# low_freq3 = 10800
# high_freq3 = 21600
# low_freq4 = 21600
# high_freq4 = 24000

# # apply the butterworth filter
# a, b = sp.signal.butter(5, [low_freq1, high_freq1], btype='bandpass', fs=sample_rate)

# Import the necessary libraries
# scipy wavfile
import scipy.io.wavfile as wave
import scipy.signal as signal

# Load the WAV file
sample_rate, audio = wave.read('audios/000316da7.wav')
file_name = 'audios/000316da7.wav'
file_name = file_name.split('.')[0]

# how many dimensions does the audio have?
# if audio.ndim == 2:
#     mono_or_stereo = True
# else:
#     mono_or_stereo = False

# print (mono_or_stereo)

# check the channels
# if audio.channels == 1:
#     mono_or_stereo = False

# if mono_or_stereo:
#     tracks = audio.split_to_mono()
# else:
    # create a copy of the audio
track1 = audio
track2 = audio
track3 = audio
track4 = audio

# get half the sample rate
nyquist = 24000


# use the equalizer to filter the audio
track1 = signal.lfilter(*signal.butter(5, [512/nyquist, 5480/nyquist], "bandpass"), track1)
track2 = signal.lfilter(*signal.butter(5, [5480/nyquist, 10800/nyquist], "bandpass"), track2)
track3 = signal.lfilter(*signal.butter(5, [10800/nyquist, 15600/nyquist], "bandpass"), track3)
track4 = signal.lfilter(*signal.butter(5, [15600/nyquist, 23000/nyquist], "bandpass"), track4)


# Save the tracks as separate WAV files but have keep the original file name
wave.write("track1.wav",sample_rate, track1)
wave.write("track2.wav",sample_rate, track2)
wave.write("track3.wav",sample_rate, track3)
wave.write("track4.wav",sample_rate, track4)






