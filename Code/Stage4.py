import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import butter, filtfilt, correlate
from datetime import datetime
from scipy.interpolate import CubicSpline

plt.close('all')

#Load CSV
file_path = "impact_data/final_shot/impact_003.csv"

df = pd.read_csv(file_path)
if not all(col in df.columns for col in ['Mic1', 'Mic2', 'Mic3']):
    raise ValueError("CSV file does not contain the required columns: 'Mic1', 'Mic2', 'Mic3'")

#real coordinates for error calculation
true_x = 114
true_y = 66

#Parameters
c = 34300  #in cm/s
fs = 44100  
lowcut = 800 
highcut = 6000 
order = 4


# --- Microphone positions (in cm) STAGE 4 ---
mic1_pos = np.array([30, 117.5, -11.5])   # Mic1 (reference - left top)
mic2_pos = np.array([160.5, 117.5, -11.5])  # Mic2 (right top)
mic3_pos = np.array([95.0, 8.5, -11.5])    # Mic3 (bottom)


# --- Functions ---
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def normalize(sig):
    return (sig - np.mean(sig)) / np.std(sig)

#cross correlation
def get_delay_range(sig_a, sig_ref, min_lag=-186, max_lag=186):
    corr = correlate(sig_a, sig_ref, mode='full', method='fft')
    lags = np.arange(-len(sig_a) + 1, len(sig_ref))
    # Find indices within the desired lag range
    valid = (lags >= min_lag) & (lags <= max_lag)
    # Only consider correlation values within the lag range
    corr_valid = corr[valid]
    lags_valid = lags[valid]
    lag = lags_valid[np.argmax(corr_valid)]
    return lag

#AOA estimation
def aoa_est(x, y, ref_pos):
    dx = x - ref_pos[0]
    dy = y - ref_pos[1]
    return np.arctan2(dy, dx)

def estimate_aoa13_from_delay(tdoa_31, mic1, mic3):
    # Compute vector from mic1 to mic3
    dx = mic3[0] - mic1[0]
    dy = mic3[1] - mic1[1]
    mic_distance = np.hypot(dx, dy)

    # Angle of the mic-to-mic line
    mic_line_angle = np.arctan2(dy, dx)

    # Small TDOA -> small offset along mic axis -> adjust angle
    angle_offset = tdoa_31 * 34300 / mic_distance  # cm/s * s / cm = scalar

    # Clamp to [-1, 1] for arcsin stability
    angle_offset = np.clip(angle_offset, -1.0, 1.0)

    # Adjust angle based on delay
    aoa13_meas = mic_line_angle + np.arcsin(angle_offset)
    return aoa13_meas

#convert coordinates to source position
def source_pos(x, y):
    return np.array([x, y, 0])

#cost function
def cost_fn(pos, mic2, mic3, ref_mic1, r21, r31, aoa31_meas=None, sigma=np.radians(20)):
    x, y = pos
    src = np.array([x, y, 0])  # constrain to z=0

    d1 = np.linalg.norm(src - ref_mic1)
    d2 = np.linalg.norm(src - mic2)
    d3 = np.linalg.norm(src - mic3)

    term1 = (d2 - d1 - r21) ** 2 # scale by 100 to increase contribution
    term2 = (d3 - d1 - r31) ** 2

    if aoa31_meas is not None:
        est_angle = aoa_est(x, y, ref_mic1)
        angle_error = (aoa31_meas - est_angle + np.pi) % (2 * np.pi) - np.pi
        term3 = (angle_error / sigma) ** 2
    else:
        term3 = 0

    return term1 + term2 + term3

#window detection
def detect_window_first_impact(signal, diff_thresh=0.3, amp_thresh=0.2, pre=100, post=100, min_sample=100):
    # Smooth the signal to reduce noise
    smooth = np.convolve(signal, np.ones(5)/5, mode='same')
    diff = np.diff(smooth)
    # Only consider after min_sample samples
    candidates = np.where(
        (np.arange(len(diff)) > min_sample) &
        (diff > diff_thresh) &
        (smooth[1:] > amp_thresh)
    )[0]
    if len(candidates) == 0:
        idx = np.argmax(signal)
    else:
        idx = candidates[0]
    window_start = max(0, idx - pre)
    window_end = min(len(signal), idx + post)
    return window_start, window_end


#clipping
def detect_clipping(signal, threshold=0.99, flat_threshold=1e-6, min_length=2):

    #clipping detection near ±1
    clipped_high = np.where(signal >= threshold)[0]
    clipped_low = np.where(signal <= -threshold)[0]

    #flat region detection
    diffs = np.abs(np.diff(signal))
    flat = np.where(diffs < flat_threshold)[0]
    flat_regions = []

    if len(flat) > 0:
        start = flat[0]
        for i in range(1, len(flat)):
            if flat[i] != flat[i - 1] + 1:
                end = flat[i - 1]
                if end - start + 1 >= min_length:
                    flat_regions.extend(range(start, end + 1))
                start = flat[i]
        end = flat[-1]
        if end - start + 1 >= min_length:
            flat_regions.extend(range(start, end + 1))

    # Combine all detected clipped regions
    all_clipped = np.unique(np.concatenate((clipped_high, clipped_low, flat_regions)))
    return np.array(all_clipped, dtype=int)

def find_clipped_regions(clipped_indices):
    if len(clipped_indices) == 0:
        return []

    regions = []
    start = clipped_indices[0]
    for i in range(1, len(clipped_indices)):
        if clipped_indices[i] != clipped_indices[i - 1] + 1:
            regions.append((start, clipped_indices[i - 1]))
            start = clipped_indices[i]
    regions.append((start, clipped_indices[-1]))
    return regions

def interpolate_clipped_regions(signal, regions, padding=3):
    repaired = signal.copy()
    for start, end in regions:
        start = int(start)
        end = int(end)
        left = max(0, start - padding)
        right = min(len(signal), end + padding + 1)

        # If the region touches the start or end, skip interpolation
        if start == 0 or end == len(signal) - 1:
            # Optionally, set to np.nan or a constant if you want to mark these
            # repaired[start:end+1] = np.nan
            continue

        known_x = np.concatenate((np.arange(left, start), np.arange(end + 1, right)))
        known_y = np.concatenate((signal[left:start], signal[end + 1:right]))

        if len(known_x) < 2:
            continue

        cs = CubicSpline(known_x, known_y)
        interp_x = np.arange(start, end + 1)
        repaired[interp_x] = cs(interp_x)

    # Cut off the first 50 samples
    #repaired[512:] = 0  
    return repaired


# --- PROCESSING ---
#filter and normalize signals
b, a = butter_bandpass(lowcut, highcut, fs, order=order)
normalized_signals= {}

for mic in ['Mic1', 'Mic2', 'Mic3']:
    raw = df[mic].values
    norm = (raw - np.min(raw)) / (np.max(raw) - np.min(raw))  # Scale to [0, 1]
    norm = 2 * norm - 1  # Scale to [-1, 1]
    normalized_signals[mic] = norm

#repair signals
filtered = {}
repaired_signals = {}
for mic in ['Mic1', 'Mic2', 'Mic3']:
    clipped = detect_clipping(normalized_signals[mic], threshold=0.99)
    regions = find_clipped_regions(clipped)
    repaired = interpolate_clipped_regions(normalized_signals[mic], regions)
    repaired_signals[mic] = repaired
    filtered[mic] = filtfilt(b, a, repaired)
    
#detect window from reference mic (Mic1)
window_start, window_end = detect_window_first_impact(filtered['Mic1'])

#slice and normalize windowed signals
windowed = {}
for mic in ['Mic1', 'Mic2', 'Mic3']:
    windowed[mic] = filtered[mic][window_start:window_end]
    
#TDOA Calculation

delay_21 = get_delay_range(windowed['Mic2'], windowed['Mic1'], min_lag=-186, max_lag=186)  # Mic2 - Mic1
delay_31 = get_delay_range(windowed['Mic3'], windowed['Mic1'], min_lag=-186, max_lag=186)  # Mic3 - Mic1

tdoa_21 = delay_21 / fs
tdoa_31 = delay_31 / fs

r21 = c * tdoa_21
r31 = c * tdoa_31

#including AOA?
if abs(delay_31) < 10: # If delay is too small, skip AOA estimation
    aoa31_meas = None
    sigma = np.radians(999) # disable AOA term in cost function
else:
    aoa31_meas = estimate_aoa13_from_delay(tdoa_31, mic1_pos, mic3_pos)
    #aoa31_meas = -1.3761 # For testing purposes, set a fixed value
    sigma = np.radians(0.5)

if aoa31_meas is not None:
    aoa31_deg = np.degrees(aoa31_meas)
    print(f"AOA from Mic1 to source: {aoa31_deg:.2f} degrees and {aoa31_meas:.4f} radians")
else:
    print("AOA estimation skipped due to insufficient delay (|delay_31| < 10 samples)")

wall_dim = [(0, 190), (0, 126.5)]
result = differential_evolution(cost_fn, wall_dim, args=(mic2_pos, mic3_pos, mic1_pos, r21, r31, aoa31_meas), seed=42, maxiter=1000, tol=1e-6)


x_est, y_est = result.x
print(f"Estimated position: x = {x_est:.2f}, y = {y_est:.2f}, z = 0 (constrained)")
#print(f"(aoa): {np.degrees(aoa31_meas)}")
print("delay_21 (samples) =", delay_21)
#print("tdoa_21 (s) =", tdoa_21)
print("r21 (cm) =", r21)
print("delay_31 (samples) =", delay_31)
#print("tdoa_31 (s) =", tdoa_31)
print("r31 (cm) =", r31)


true_pos = np.array([true_x, true_y])
est_pos = np.array([x_est, y_est])
error = np.linalg.norm(est_pos - true_pos)
print(f"Localization error: {error:.2f} cm")


mic_linestyles = {'Mic1': '-', 'Mic2': '--', 'Mic3': ':'}


# --- Plot 1: Raw Signals ---
plt.figure(figsize=(10, 4))
for mic in ['Mic1', 'Mic2', 'Mic3']:
    raw = df[mic].values
    plt.plot(raw, label=f'Raw {mic}', linestyle=mic_linestyles[mic], linewidth=1)
plt.ylabel("Amplitude")
plt.xlabel("Sample Index")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
#plt.show()

# --- Plot 2: Repaired Signals ---
plt.figure(figsize=(10, 4))
for mic in ['Mic1', 'Mic2', 'Mic3']:
    plt.plot(repaired_signals[mic], label=f'Repaired {mic}', linestyle=mic_linestyles[mic], linewidth=1)
plt.ylabel("Normalized Amplitude")
plt.xlabel("Sample Index")
plt.legend(loc ='upper left')
plt.grid(True)
plt.tight_layout()
#plt.show()

# --- Plot 3: Filtered Signals ---
plt.figure(figsize=(10, 4))
for mic in ['Mic1', 'Mic2', 'Mic3']:
    plt.plot(filtered[mic], label=f'Filtered {mic}', linestyle=mic_linestyles[mic], linewidth=1)
plt.ylabel("Amplitude")
plt.xlabel("Sample Index")
plt.legend(loc ='upper left')
plt.grid(True)
plt.tight_layout()
#plt.show()

# --- Plot 4: Detected Window on Mic1 ---
plt.figure(figsize=(10, 4))
plt.plot(filtered['Mic1'], linewidth=1)
plt.axvline(window_start, color='red', linestyle='--', label='Window Start')
plt.axvline(window_end, color='green', linestyle=':', label='Window End')
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend(loc ='upper left')
plt.grid(True)
plt.tight_layout()
#plt.show()

# --- Plot 5: Cross-Correlation (Mic2 vs Mic1) ---
plt.figure(figsize=(10, 4))
corr_21 = correlate(filtered['Mic2'], windowed['Mic1'], mode='full', method='fft')
lags_21 = np.arange(-len(filtered['Mic2']) + 1, len(windowed['Mic1']))
plt.plot(lags_21, corr_21, linewidth=1)
plt.axvline(delay_21, color='red', linestyle='--', label=f'Peak lag: {delay_21}')
plt.legend()
plt.xlabel("Lag")
plt.ylabel("Cross-Correlation")
plt.grid(True)
plt.tight_layout()
#plt.show()

# --- Plot 6: Cross-Correlation (Mic3 vs Mic1) ---
plt.figure(figsize=(10, 4))
corr_31 = correlate(filtered['Mic3'], windowed['Mic1'], mode='full', method='fft')  
lags_31 = np.arange(-len(filtered['Mic3']) + 1, len(windowed['Mic1']))
plt.plot(lags_31, corr_31, linewidth=1)
plt.axvline(delay_31, color='red', linestyle='--', label=f'Peak lag: {delay_31}')
plt.legend()
plt.xlabel("Lag")
plt.ylabel("Cross-Correlation")
plt.grid(True)
plt.tight_layout()
#plt.show()

# --- Plot: Cross-Correlation (Mic2 vs Mic1 and Mic3 vs Mic1) ---
plt.figure(figsize=(6, 4))
corr_21 = correlate(filtered['Mic2'], windowed['Mic1'], mode='full', method='fft')
lags_21 = np.arange(-len(filtered['Mic2']) + 1, len(windowed['Mic1']))
corr_31 = correlate(filtered['Mic3'], windowed['Mic1'], mode='full', method='fft')  
lags_31 = np.arange(-len(filtered['Mic3']) + 1, len(windowed['Mic1']))
plt.plot(lags_21, corr_21, linestyle='-', linewidth=1, label="Mic2 vs Mic1")
plt.plot(lags_31, corr_31, linestyle='--', linewidth=1, label="Mic3 vs Mic1")
plt.axvline(delay_21, color='red', linestyle=':', label=f'Peak lag Mic2: {delay_21}')
plt.axvline(delay_31, color='green', linestyle=':', label=f'Peak lag Mic3: {delay_31}')
plt.legend()
plt.xlabel("Lag")
plt.ylabel("Cross-Correlation")
plt.grid(True)
plt.tight_layout()
#plt.show()

def plot_cost_function_heatmap(mic1_pos, mic2_pos, mic3_pos, r21, r31, aoa31_meas=None, sigma=np.radians(20)):
    x_range = np.linspace(0, 190, 50)
    y_range = np.linspace(0, 126.5, 50)
    cost_grid = np.zeros((len(x_range), len(y_range)))

    #calculate cost for each point in the grid
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            cost_grid[i, j] = cost_fn(
                [x, y],
                mic2_pos, mic3_pos, mic1_pos,
                r21, r31, aoa31_meas
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    contour = plt.contourf(x_range, y_range, cost_grid.T, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour)  # capture the colorbar object
    cbar.set_label("Cost", fontsize=20, labelpad=10)
    cbar.ax.tick_params(labelsize=16)  # adjust tick label font size

    # Plot wall
    wall_top_left = (0, 126.5)
    wall_top_right = (190, 126.5)
    wall_bottom_left = (0, 0)
    wall_bottom_right = (190, 0)
    
    ax.plot([wall_top_left[0], wall_top_right[0]], [wall_top_left[1], wall_top_right[1]], 'k', linewidth=3, label="Wall Outline")
    ax.plot([wall_bottom_left[0], wall_bottom_right[0]], [wall_bottom_left[1], wall_bottom_right[1]], 'k', linewidth=3)
    ax.plot([wall_top_left[0], wall_bottom_left[0]], [wall_top_left[1], wall_bottom_left[1]], 'k', linewidth=3)
    ax.plot([wall_top_right[0], wall_bottom_right[0]], [wall_top_right[1], wall_bottom_right[1]], 'k', linewidth=3)
    
    
    ax.text(mic1_pos[0], mic1_pos[1], r"$x_1$", fontsize=18, ha='center', va='center', color='red')
    ax.text(mic2_pos[0], mic2_pos[1], r"$x_2$", fontsize=18, ha='center', va='center', color='red')
    ax.text(mic3_pos[0], mic3_pos[1], r"$x_3$", fontsize=18, ha='center', va='center', color='red')
    ax.scatter(x_est, y_est, color='red', marker='o', s=100, label='Estimated Position (x)')
    
    #legend
    mic1_legend = plt.Line2D([0], [0], linestyle="none", marker='x', color= 'red' ,label=f"$mic_1$ ({mic1_pos[0]}, {mic1_pos[1]}, {mic1_pos[2]})")
    mic2_legend = plt.Line2D([0], [0], linestyle="none", marker='x', color= 'red' ,label=f"$mic_2$ ({mic2_pos[0]}, {mic2_pos[1]}, {mic2_pos[2]})")
    mic3_legend = plt.Line2D([0], [0], linestyle="none", marker='x', color= 'red' ,label=f"$mic_3$ ({mic3_pos[0]}, {mic3_pos[1]}, {mic3_pos[2]})")
    est_pos_legend = plt.Line2D([0], [0], linestyle="none", marker='o', color='red', label=f"Estimated Position ({x_est:.2f}, {y_est:.2f})")


    
    plt.xlabel("X /cm")
    plt.ylabel("Y /cm")
    plt.legend(handles=[mic1_legend, mic2_legend, mic3_legend, est_pos_legend], loc = 'best', fontsize= 14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

# --- Plot Cost Function Heatmap ---
plot_cost_function_heatmap(mic1_pos, mic2_pos, mic3_pos, r21, r31, aoa31_meas, sigma)


