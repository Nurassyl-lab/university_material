# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:15:40 2021

@author: Nurassyl Askar student id: 0810981
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftshift

f = 5
f_s = 100
time = np.linspace(0, 1, 2 * f_s)
amplitude = np.sin(f * 2 * np.pi * time)

plt.figure(1)
plt.title("Figure A (Time domain)")
plt.plot(time, amplitude)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')

freq = fft(amplitude)
freq = fftshift(freq)
freq = abs(freq)

plt.figure(2)
plt.title("Figure B (Frequency domain)")
plt.plot(freq)
plt.xlabel('Frequency')
plt.ylabel('Magnitude ')


#if you want to record and process audio uncomment code bellow 

# import scipy.io.wavfile
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.fftpack import fft
# import sounddevice as sd
# import time
# SAMPLE_FREQ = 44100 
# SAMPLE_DUR = 1

# print("Recording starts in 1 second")
# time.sleep(1) 
# recording = sd.rec(SAMPLE_DUR * SAMPLE_FREQ, samplerate=SAMPLE_FREQ,
#                    channels=1,dtype='float64')
# print("Recording audio 1 second")
# sd.wait()

# sd.play(recording, SAMPLE_FREQ)
# print("Playing audio")
# sd.wait()
# scipy.io.wavfile.write('example1.wav', SAMPLE_FREQ, recording)  
# sample_frequency, recording = scipy.io.wavfile.read("example1.wav")
# sample_duration = len(recording)/sample_frequency
# timeX = np.arange(0, sample_frequency/2, sample_frequency/len(recording))
# absFreqSpectrum = abs(fft(recording))
# print(absFreqSpectrum)
# plt.figure(1)
# plt.title("Audio Signal 432Hz")
# time_signal = np.arange(0,sample_duration, 1/sample_frequency)
# plt.xlabel("time")
# plt.ylabel("signal")
# plt.plot(time_signal, recording)
# plt.figure(2)
# plt.title("Spectogram of the signal")
# plt.plot(timeX, absFreqSpectrum[:len(recording)//2])
# plt.ylabel('|X(n)|')
# plt.xlabel('frequency[Hz]')
# plt.show()