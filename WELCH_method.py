import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

#load CSV
file_path = "impact_data/final_shot/impact_003.csv"
df = pd.read_csv(file_path)
signal = df['Mic3'].values
fs = 44100  # Hz

impact_start = 0
impact_end = 512
segment = signal[impact_start:impact_end]

#welch's method
frequencies, psd = welch(
    segment,
    fs=fs,
    window='hann',
    nperseg=128,
    noverlap=64,
    scaling='density'
)

#plot
plt.figure(figsize=(6, 4))
plt.semilogy(frequencies, psd)
plt.xlabel("Frequency / Hz", fontsize=20)
plt.ylabel(r"PSD / dB$\cdot$Hz$^{-1}$", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, which='both')
plt.tight_layout()
plt.show()
