import numpy as np
import gwpy
from gwpy.timeseries import TimeSeries
import h5py
import json5
import argparse
import pyfstat
from pyfstat.utils import get_sft_as_arrays


# Importing timestamps
file_name = "ASCII_timestamp_data.json"
file = open(file_name, "r")
ASCII = json5.load(file)
file.close()

# Setting up argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--min_sft', type=int, help="Specify the smallest SFT")
parser.add_argument('--max_sft', type=int, help="Specify the largest SFT")
parser.add_argument('--F0', type=float, help="Specify the synthetic signal frequency")
parser.add_argument('--h0', type=float, help="Specify the synthetic signal amplitude")

args = parser.parse_args()

def inject_signal(detector: str = 'H1', min_sft: int = 0, max_sft: int = None, window: str = 'tukey',
                 sqrtSX: float = 0.0, F0: float = 201, F1: float = -1e-9, Alpha: float = 2.0, 
                 Delta: float = 1.0, h0: float = 1e-22, cosi: float = 1.0):

    timestamps = []
    for i, gps in enumerate(ASCII[detector][::3]): # The times when the detector starts running.
        for ii in range(int(ASCII[detector][3*i+2] / 1800)): # Runs the possible number of 30 minute segments.
            timestamps.append(int(gps)+1800*ii)

    timestamps = timestamps[min_sft:max_sft]
    timestamps_dict = {detector: np.array(timestamps)}

    noise_kwargs = {
        "sqrtSX": sqrtSX,
        "detectors": detector,
        "Tsft": 1800,
        "F0": 201, # Skift til 350
        "Band": 2, # Skift til 400
        "SFTWindowType": window,
        "timestamps": timestamps_dict,
        }
    
    noise_writer = pyfstat.Writer(label="custom_band_noise", **noise_kwargs)
    noise_writer.make_data()

    signal_kwargs = {
        "noiseSFTs": noise_writer.sftfilepath,
        "F0": F0,
        "F1": F1,
        "Alpha": Alpha,
        "Delta": Delta,
        "h0": h0,
        "cosi": cosi,
        "psi": 0,
        "phi": 0,
        "SFTWindowType": window,
        }
    
    signal_writer = pyfstat.Writer(label="custom_band_signal", **signal_kwargs)
    signal_writer.make_data()
    
    frequency, timestamps, fourier_data = get_sft_as_arrays(signal_writer.sftfilepath)
    print(fourier_data[detector].shape, frequency)
    
    freqs = np.arange(200, 202, 0.2) # Skift til 150 til 550
    for freq in freqs:
        with h5py.File(f'SFT_files_{h0}/SFT_f0_{round(freq,1)}_.hdf5', 'a') as f:
            
            dataset_sft = f[f'_f0_{round(freq,1)}/{detector}/SFTs']
            dataset_sft[:, min_sft:max_sft] += fourier_data[detector][(frequency >= freq) & (frequency < freq+0.2)]
            
inject_signal(detector='L1', min_sft=args.min_sft, max_sft=args.max_sft, F0=args.F0, h0=args.h0*1e-23)
inject_signal(detector='H1', min_sft=args.min_sft, max_sft=args.max_sft, F0=args.F0, h0=args.h0*1e-23)