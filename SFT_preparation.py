import numpy as np
import matplotlib.pyplot as plt
import gwpy
from gwpy.timeseries import TimeSeries
import h5py
import time
import json5

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

def compute_sfts(detector: str = 'H1', min_sft: int = 0, max_sft: int = 5, inject_signal: bool = False,
                 sqrtSX: float = 0.0, F0: float = 150.10, F1: float = -1e-9, Alpha: float = 0.0,
                 Delta: float = 0.0, h0: float = 1e-22, cosi: float = 0.0, window: str = 'tukey'):
    """
    The amplitude of the injected signal follows the equation for rho^2 in:
    https://github.com/PyFstat/PyFstat/blob/master/examples/tutorials/1_generating_signals.ipynb

    When preparing number of SFTs, one must give the function a min_sfts and max_sfts value. 
    The timestamps are then extracted for SFT numbers in the range [min_sfts, max_sfts).
    """
    
    # Preparing timestamps
    timestamps = []
    for i, gps in enumerate(ASCII[detector][::3]): # The times when the detector starts running.
        for ii in range(int(ASCII[detector][3*i+2] / 1800)): # Runs the possible number of 30 minute segments.
            timestamps.append(int(gps)+1800*ii)
    
    maximum_num_sfts = len(timestamps)
    timestamps = timestamps[min_sft:max_sft]
    timestamps_dict = {detector: np.array(timestamps)}

    # Initializing generated signal parameters
    if inject_signal:
        import pyfstat
        from pyfstat.utils import get_sft_as_arrays
        
        noise_kwargs = {
            "sqrtSX": sqrtSX,
            "detectors": detector,
            "Tsft": 1800,
            "F0": 350,
            "Band": 400,
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
        
        freqs, timestamps, fourier_data = get_sft_as_arrays(signal_writer.sftfilepath)

    freqs = np.arange(150, 550, 0.2)         
    n_sfts = 0
    for ii, gps in enumerate(timestamps_dict[detector]): # Runs the possible number of 30 minute segments.
        segment = (int(gps), int(gps)+1800)
        data = TimeSeries.fetch_open_data(detector, *segment, verbose=False)

        # Performing FFT
        fft = data.fftgram(fftlength=1800, overlap=None, window=window)
        mask = (fft.yindex.value >= 150) & (fft.yindex.value < 550)
        fft = fft[:,mask] # Only keeps frequencies between 150 and 550 Hz.
        
        # Inject signal if True
        if inject_signal:
            fft.value[0,:] += fourier_data[detector][:,ii]

        # Splits the frequency range into 0.2 Hz bins.
        for freq in freqs:
            mask = (fft.yindex.value >= freq) & (fft.yindex.value < freq+0.2)
            fft_masked = np.transpose(fft[:,mask])
            
            # Add SFT to file
            with h5py.File(f'../SFT_files/SFT_f0_{round(freq,1)}_.hdf5', 'a') as f:

                if ii == 0:
                        try:
                            # Create file
                            try:
                                initial_group = f.create_group(f'_f0_{round(freq,1)}')
                            except:
                                initial_group = f[f'_f0_{round(freq,1)}']
                            
                            data_group = initial_group.create_group(detector)
                            dataset_sft = data_group.create_dataset('SFTs', shape=(360, maximum_num_sfts), maxshape=(None, None), dtype=np.complex64)
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


compute_sfts(detector='L1', min_sft=6, max_sft=1000, F0=306.5, inject_signal=False)
