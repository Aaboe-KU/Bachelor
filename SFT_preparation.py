import numpy as np
import gwpy
from gwpy.timeseries import TimeSeries
import h5py
import json5
import argparse

# Importing timestamps
file_name = "ASCII_timestamp_data.json"
file = open(file_name, "r")
ASCII = json5.load(file)
file.close()

# Setting up argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--min_sft', type=int, help="Specify the smallest SFT")
parser.add_argument('--max_sft', type=int, help="Specify the largest SFT")
parser.add_argument('--maximum_sfts', type=int, help="Specify the maximum number of SFTs")

args = parser.parse_args()

def compute_sfts(detector: str = 'H1', min_sft: int = 0, max_sft: int = None, window: str = 'tukey',
                 maximum_sfts: int = 5000):
  
    # Preparing timestamps
    timestamps = []
    for i, gps in enumerate(ASCII[detector][::3]): # The times when the detector starts running.
        for ii in range(int(ASCII[detector][3*i+2] / 1800)): # Runs the possible number of 30 minute segments.
            timestamps.append(int(np.floor(gps)+1800*ii))
    
    timestamps = timestamps[min_sft:max_sft]
    timestamps_dict = {detector: np.array(timestamps)}

    freqs = np.arange(150, 550, 0.2)  
    for ii, gps in enumerate(timestamps_dict[detector]): # Runs the possible number of 30 minute segments.
        segment = (int(gps), int(gps)+1800)
        try:
            data = TimeSeries.fetch_open_data(detector, *segment, verbose=False)
        except:
            continue

        # Performing FFT
        fft = data.fftgram(fftlength=1800, overlap=None, window=window)
        mask = (fft.yindex.value >= 150) & (fft.yindex.value < 550)
        fft = fft[:,mask] # Only keeps frequencies between 150 and 550 Hz.

        # Splits the frequency range into 0.2 Hz bins.
        for freq in freqs:
            mask = (fft.yindex.value >= freq) & (fft.yindex.value < freq+0.2)
            fft_masked = np.transpose(fft[:,mask])
            
            # Add SFT to file
            with h5py.File(f'SFT_files/SFT_f0_{round(freq,1)}_.hdf5', 'a') as f:

                if ii == 0:
                        try:
                            # Create file
                            try:
                                initial_group = f.create_group(f'_f0_{round(freq,1)}')
                            except:
                                initial_group = f[f'_f0_{round(freq,1)}']
                            
                            data_group = initial_group.create_group(detector)
                            dataset_sft = data_group.create_dataset('SFTs', shape=(360, maximum_sfts), maxshape=(None, None), dtype=np.complex64)
                            dataset_timestamps = data_group.create_dataset('timestamps_GPS', shape=dataset_sft.shape[1], maxshape=(None, ), dtype=np.int64)
                            dataset_frequencies = initial_group.create_dataset('frequency_Hz', shape=dataset_sft.shape[0], maxshape=(None, ), dtype=np.float64)

                        except:
                            # Load datasets from file
                            dataset_sft = f[f'_f0_{round(freq,1)}/{detector}/SFTs']
                            dataset_timestamps = f[f'_f0_{round(freq,1)}/{detector}/timestamps_GPS']
                            dataset_frequencies = f[f'_f0_{round(freq,1)}/frequency_Hz']

                        finally:
                            # Add initial values
                            dataset_sft[:, min_sft] = fft_masked.value[:, 0]
                            dataset_timestamps[min_sft] = fft_masked.xindex.value[0]
                            dataset_frequencies[:] = fft_masked.yindex.value
                else:
                    # Load datasets from file
                    dataset_sft = f[f'_f0_{round(freq,1)}/{detector}/SFTs']
                    dataset_timestamps = f[f'_f0_{round(freq,1)}/{detector}/timestamps_GPS']
                    dataset_frequencies = f[f'_f0_{round(freq,1)}/frequency_Hz']

                    # Add new values
                    dataset_sft[:, min_sft+ii] = fft_masked.value[:, 0]
                    dataset_timestamps[min_sft+ii] = fft_masked.xindex.value[0]


compute_sfts(detector='L1', maximum_sfts=args.maximum_sfts, min_sft=args.min_sft, max_sft=args.max_sft)
compute_sfts(detector='H1', maximum_sfts=args.maximum_sfts, min_sft=args.min_sft, max_sft=args.max_sft)
