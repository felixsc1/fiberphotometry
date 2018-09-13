from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import peakutils
from collections import Counter
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def openfiledialogue(whichending):
    root = Tk()
    ftypes = [('LVM file',f"*{whichending}"),("All Files","*.*")]
    ttl = "Select file"

    if os.name == 'nt':
        dir1 = 'C:\\'   # Windows
    else:
        dir1 = '/Volumes/Zhivaz/'

    root.focus_force()
    # root.withdraw()
    file_path = askopenfilename(filetypes = ftypes, initialdir = dir1, title = ttl)
    root.deiconify(), root.update(), root.destroy()
    return file_path

def detect_stim_paradigm(stimchannel, fs):
    '''
    only works for evenly spaced block paradigms with constant ISI
    '''
    peakindexes = peakutils.indexes(stimchannel, thres=0.6, min_dist=0.01*fs)
    peakindexes_sec=peakindexes/fs
    delays = np.around(np.diff(peakindexes_sec),decimals=2)
    x = Counter(delays)   #counts elements in the input, and then shows which is most frequent, how often it appears etc...
    x1 = x.most_common(1)
    n_pulses = x1[0][1]
    x2 = x.most_common()[:-2:-1]  #find the longest delay (i.e. the least common value)
    n_blocks = x2[0][1] + 1
    stim_freq = np.round(1 / x1[0][0])  #remove round() for stim <1Hz
    pulses_block = n_pulses / n_blocks + 1
    block_duration = np.round(pulses_block / stim_freq)
    off_period = x2[0][0] - x1[0][0]
    print('stim freq: ' + str(stim_freq) + 'Hz')
    print('ON period: ' + str(block_duration) + 's')
    print('OFF period: ' + str(off_period) + 's')
    period = (block_duration + off_period) * 2
    print('Signals slower than ' + str(period) + 's will be removed (later)')
    return n_pulses, n_blocks, stim_freq, pulses_block, block_duration, off_period, peakindexes_sec, peakindexes


def FFT_calculate_plot(calcium, fs, plotsize):
    f1, Pxx_den1 = signal.welch(calcium['channel1','raw'], fs, nperseg=65536)

    fig = plt.figure(figsize=plotsize)
    ax = plt.axes()
    ax.semilogy(f1, Pxx_den1, label='channel 1')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    ax.set_title('Frequency spectra of raw signals')

    f2, Pxx_den2 = signal.welch(calcium['channel2','raw'],fs,nperseg=65536)
    ax.semilogy(f2, Pxx_den2, label='channel 2')
    ax.set_xlim(500, 0.2*fs)
    ax.legend()

    # Detect modulation frequencies automatically by getting highest peak
    if 'mod_ch1' not in locals():
        mod_ch1 = np.around(f1[np.argmax(Pxx_den1)],decimals=0)
    if 'mod_ch2' not in locals():
        mod_ch2 = np.around(f2[np.argmax(Pxx_den2)],decimals=0)
    return mod_ch1, mod_ch2
    