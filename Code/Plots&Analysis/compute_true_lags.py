import numpy as np

#parameters
c = 34300  #in cm/s
fs = 44100  

# --- Microphone positions (in cm) ---
'''mic1_pos = np.array([22.5, 118.0, -2.5])   # Mic1 (reference - left top)
mic2_pos = np.array([167.5, 118.0, -2.5])  # Mic2 (right top)
mic3_pos = np.array([95.0, 17.5, -2.5])    # Mic3 (bottom)'''

# --- Microphone positions (in cm) STAGE 4 ---
mic1_pos = np.array([30, 117.5, -11.5])   # Mic1 (reference - left top)
mic2_pos = np.array([160.5, 117.5, -11.5])  # Mic2 (right top)
mic3_pos = np.array([95.0, 8.5, -11.5])    # Mic3 (bottom)

def compute_theoretical_delays(source, mic1_pos, mic2_pos, mic3_pos, fs):
    #distances mics
    d1 = np.linalg.norm(source - mic1_pos)
    d2 = np.linalg.norm(source - mic2_pos)
    d3 = np.linalg.norm(source - mic3_pos)

    r21 = d2 - d1
    r31 = d3 - d1

    tdoa_21 = r21 / c
    tdoa_31 = r31 / c

    lag_21 = round(tdoa_21 * fs)
    lag_31 = round(tdoa_31 * fs)

    #AOA from Mic1
    dx = source[0] - mic1_pos[0]
    dy = source[1] - mic1_pos[1]
    aoa_rad = np.arctan2(dy, dx)
    aoa_deg = np.degrees(aoa_rad)

    return lag_21, lag_31, aoa_rad, aoa_deg

#positions

sources = {"#1 (63/44)": np.array([63, 44, 0]),
           "#3 (114/66)": np.array([114, 66, 0]),
           "#4 (59/36)": np.array([59, 36, 0]),
           "#5 (88/70)": np.array([88, 70, 0]),
           "#6 (151/60)": np.array([151, 60, 0]),
           "#7 (175/55)": np.array([175, 55, 0]),
           "#8 (54/52)": np.array([54, 52, 0]),
           "#9 (60/50)": np.array([60, 50, 0]),
           "#11 (60/67)": np.array([60, 67, 0]),
           "#14 (118/89)": np.array([118, 89, 0]),
           "#15 (114/65)": np.array([114, 65, 0]),
           "#16 (39/71)": np.array([39, 71, 0])
}

#calculation
for label, src in sources.items():
    lag_21, lag_31, aoa_rad, aoa_deg = compute_theoretical_delays(src, mic1_pos, mic2_pos, mic3_pos, fs)
    print(f"{label}")
    print(f"Delay Mic2-Mic1: {lag_21} samples")
    print(f"Delay Mic3-Mic1: {lag_31} samples")
    print(f"AOA from Mic1: {aoa_rad:.4f} rad / {aoa_deg:.2f}°\n")

for label, src in sources.items():
    lag_21, lag_31, aoa_rad, aoa_deg = compute_theoretical_delays(src, mic1_pos, mic2_pos, mic3_pos, fs)
    print(f"{label}")
    print(f"Delay Mic2-Mic1: {lag_21} samples")
    print(f"Delay Mic3-Mic1: {lag_31} samples")
    print(f"AOA from Mic1: {aoa_rad:.4f} rad / {aoa_deg:.2f}°\n")
