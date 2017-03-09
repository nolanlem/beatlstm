#!/usr/bin/env python

import librosa 
import matplotlib.pyplot as plt 
import numpy as np


# free jazz drums
audio_file = './audio/nommo_60.wav'

# librosa example track
#audio_file = librosa.util.example_audio_file()

hop_length = 512
y, sr = librosa.load(audio_file, duration=15, sr=None)

timewindowst = hop_length*400
timewindowend = hop_length*500
y = y[timewindowst:timewindowend]
frames = len(y)//hop_length 


print('sr: %r , frames: %r, samps in audio: %r' % (sr,frames,len(y)))

onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, 
aggregate=np.median) 

# calculate autocorrelation 

ac = librosa.autocorrelate(onset_env, 2* sr // hop_length)

'''
plt.figure() 
plt.plot(onset_env) 
plt.show()
plt.close()
'''

peaks = librosa.util.peak_pick(onset_env,3,3,3,5,0.5,10) 
print('num peaks: %r', len(peaks)) 



plt.subplot(2,1,1) 
plt.plot(onset_env, linewidth='0.5') 
plt.vlines(peaks, 0, onset_env.max(), color='r', alpha=0.8, linewidth='0.5')
plt.xlabel('frame num') 
plt.ylabel('spectral onset envelope')

plt.axis('tight') 
plt.tight_layout() 

plt.subplot(2,1,2) 
plt.plot(ac, linewidth='0.7')
plt.title('autocorrelation or env_onset')


plt.show() 
plt.close()


'''
times = librosa_frames_to_time(np.arange(len(onset_env)), sr =sr, hop_length=512)
plt.figure() 
ax = plt.subplot
'''