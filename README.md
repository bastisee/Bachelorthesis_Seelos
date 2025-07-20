# Bachelorthesis_Seelos
Code and data for my bachelor thesis: Acoustic Localization of Puck Impacts on an Ice Hockey Goal Wall using Source Localization Algorithms. 

## Thesis Goal
This thesis presents the design, development, and evaluation of an acoustic sound source localization approach in order to locate the impact of an ice hockey puck after
a shot on a vertical wooden wall. This should build the foundation to further develop a smart training device to improve timing and shot accuracy in ice hockey.

## Development Environment

### Software & Languages
- **C++** for microcontroller programming using **PlatformIO** inside **Visual Studio Code**
- **Python 3.10** for signal processing, TDOA estimation, and visualization

### Hardware
- **Microcontroller**: Teensy 4.1
- **Sensors**: 3 MAX9814 microphone boards (connected directly to ADC pins)

### Development Platform
- **IDE**: Visual Studio Code with [PlatformIO](https://platformio.org/)
- **Python environment**: with packages installed via `pip`
- **OS**: Windows 10

## Repository Structure

- Bachelorthesis_Seelos/
  - Code/
    - Teensy/
      - src/
        - main.cpp                    (C++ code for dual ADC microphone sampling on Teensy)
    - Signal Processing/              (SSL and processing visualization in every prototype stage)
      - Stage1_2.py                  
      - Stage3.py
      - Stage4.py
      - save_captured_impacts          (Script to store serial received data in CSV files)
    - Plots&Analysis/                  (Visualize and analyze results)
      - compute_true_lags.py            (theoretical lag calculation for further analysis)
      - goalplot.py
      - boxplots.py
      - WELCH_method.py
      
  - Data/
    - Test Series Shoot/                (CSV files of captured mic data for each impact)
      - ...            
    - Test Series Throw/
      - ...
    - data_testings.py                  (Assignment of real coordinates to impact number)
  
  - Docs/
    - Bachelorthesis_SebastianSeelos.pdf  (Final PDF of the bachelor thesis)

  - README.md                       (Project overview and documentation)

