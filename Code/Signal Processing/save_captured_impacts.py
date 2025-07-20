import serial
import os
import csv
from datetime import datetime

PORT = 'COM4'
BAUD = 500000
SAMPLES_PER_IMPACT = 512
SAVE_FOLDER = 'impact_data/final_shot'

ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Connected to {PORT} at {BAUD} baud.")

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

recording = False
current_block = []
teensy_impact_number = 0

try:
    while True:
        line = ser.readline().decode(errors='ignore').strip()

        # Detect impact trigger
        if line.startswith("# Impact"):
            try:
                teensy_impact_number = int(line.split()[-1])
            except:
                teensy_impact_number += 1

            print(f"\n Detected {line}")
            recording = True
            current_block = []
            continue

        # End
        if line == "# End":
            print(" Reached end marker")
            recording = False

        # validate and append during recording
        if recording:
            if line.count(',') == 2:
                parts = line.split(',')
                if all(part.strip().isdigit() for part in parts):
                    current_block.append([int(p) for p in parts])
                else:
                    print(f" Skipped non-numeric line: {line}")
            else:
                print(f" Skipped malformed line: {line}")

        #save and stop after reached sample length
        if len(current_block) == SAMPLES_PER_IMPACT:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"impact_{teensy_impact_number:03d}_{timestamp}.csv"
            filepath = os.path.join(SAVE_FOLDER, filename)

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Mic1", "Mic2", "Mic3"])
                writer.writerows(current_block)

            print(f" Saved {filename}")
            recording = False
            current_block = []

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    ser.close()
