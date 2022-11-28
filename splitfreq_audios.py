import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display
import soundfile as sf

# load the audio file
y, sr = lb.load('audios/000316da7.wav', sr=48000)

# get the highest frequency of the audio file
max_freq = sr/2

print(max_freq)

#creat mel spectrogram
mel_spec = lb.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)

# plot the mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', fmax=max_freq, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
# plt.show()

print(mel_spec.shape)
# split the audio file into 3 frequency bands
mel_spec_low = mel_spec[0:43, :]
mel_spec_mid = mel_spec[43:85, :]
mel_spec_high = mel_spec[85:128, :]


# print(mel_spec_low.shape)
# print(mel_spec_mid.shape)
# print(mel_spec_high.shape)

# convert the frequency bands into audio files
y_low = lb.feature.inverse.mel_to_audio(mel_spec_low, sr=sr, n_fft=2048, hop_length=512)
y_mid = lb.feature.inverse.mel_to_audio(mel_spec_mid, sr=sr, n_fft=2048, hop_length=512)
y_high = lb.feature.inverse.mel_to_audio(mel_spec_high, sr=sr, n_fft=2048, hop_length=512)


# # save the audio files
sf.write('audios/000316da7_low.wav', y_low, sr)
sf.write('audios/000316da7_mid.wav', y_mid, sr)
sf.write('audios/000316da7_high.wav', y_high, sr)
