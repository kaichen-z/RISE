import numpy as np
import pdb
import math
import torch
# ----------------------------------------- Range ----------------------------------------------
class _range_fft_:
    def __init__(self, hyper, size_dict):
        '''
        For_FFT: rangeWindowCoeffVec, rangeFFTSize, 
        For_Visual: rangeBinSize, minRangeBinKeep, rightRangeBinDiscard,
        '''
        self.hyper = hyper
        self.size_dict = size_dict

    def apply_fft(self, volume):
        numrange = volume.shape[0]
        numdoppler = volume.shape[1]
        numrx = volume.shape[2]
        numtx = volume.shape[3]
        self.output = torch.zeros((numrange, numdoppler, numrx, numtx), dtype=torch.cfloat)
        self.output_snr = torch.zeros((numrange, numdoppler, numrx, numtx), dtype=torch.cfloat)
        # for loop mainly for storage consideration.
        for t in range(numtx):
            for r in range(numrx):
                mat = volume[:, :, r, t]
                mat = mat - torch.mean(mat, dim=0, keepdim=True)
                mat = mat * self.hyper['rangeWindowCoeffVec'][:, None]
                fftOutput = torch.fft.fft(mat, n=self.hyper["rangeFFTSize"], dim=0)
                self.output[:, :, r, t] = fftOutput
                """
                noise_power = torch.mean(range_angle_map[:10, :])  # Example: mean of the first few rows
                snr_map = range_angle_map / noise_power
                snr_db_map = 10 * torch.log10(snr_map)
                """
    
# ----------------------------------------- Doppler ----------------------------------------------
class _dopper_fft_:
    def __init__(self, hyper, size_dict):
        '''
        For_FFT: dopplerWindowCoeffVec, dopplerFFTSize, velocityBinSize,
        '''
        self.hyper = hyper
        self.size_dict = size_dict
    def apply_fft(self, volume):
        numrange = volume.shape[0]
        numdoppler = volume.shape[1]
        numrx = volume.shape[2]
        numtx = volume.shape[3]
        self.output = torch.zeros((numrange, numdoppler, numrx, numtx), dtype=torch.cfloat)
        self.output_shift = torch.zeros((numrange, numdoppler, numrx, numtx), dtype=torch.cfloat)
        for t in range(numtx):
            for r in range(numrx):
                mat = volume[:, :, r, t]
                mat = mat * self.hyper['dopplerWindowCoeffVec'][None, :]
                fftOutput = torch.fft.fft(mat, n=self.hyper["dopplerFFTSize"], dim=1)
                fftOutput_shifted = torch.fft.fftshift(fftOutput, dim=1)
                self.output[:, :, r, t] = fftOutput
                self.output_shift[:, :, r, t] = fftOutput_shifted
        
# ----------------------------------------- Angle ----------------------------------------------
class _angle_fft_:
    def __init__(self, hyper, size_dict):
        ''' For_FFT: antenna_azimuthonly '''
        self.hyper = hyper
        self.size_dict = size_dict
        # antenna_azimuthonly index starting from 0
        # ---- Double check the zero start or one start
        self.hyper['antenna_azimuthonly'] = self.hyper['antenna_azimuthonly'] - 1
        self.hyper['azimuthFFTSize'] = 256

    def apply_fft_virtual(self, volume):
        numrange = volume.shape[0]
        numdoppler = volume.shape[1]
        numrx = volume.shape[2]
        numtx = volume.shape[3]
        # Very Important: Never delete!!!!!
        # Due to different storage method of python and matlab!!!!!
        volume = volume.permute(0, 1, 3, 2).reshape(numrange, numdoppler, numtx*numrx) # 
        volume = volume[:, :, self.hyper['antenna_azimuthonly']]
        self.output_1d = torch.fft.fft(volume, n=self.hyper['azimuthFFTSize'], dim=2)
        self.output_1d_shift = torch.fft.fftshift(self.output_1d, dim=2)

class _angle_mag_eight_:
    def __init__(self, hyper, size_dict):
        self.hyper = hyper
        self.size_dict = size_dict
        if self.hyper['antenna_azimuthonly'][0] == -1:
            self.hyper_antenna_azimuthonly = self.hyper['antenna_azimuthonly'] - 1
        else:
            self.hyper_antenna_azimuthonly = self.hyper['antenna_azimuthonly']
        self.hyper_antenna_azimuth_elev = self.hyper['antenna_azimuth_elev']
        tx_mask = torch.zeros(33); tx_mask[::4] = 1
        tx_num = torch.arange(1, len(tx_mask)+1)*tx_mask
        
        rx_mask = torch.cat((torch.ones(4), torch.zeros(7), torch.ones(4), torch.zeros(31), torch.ones(8)))
        rx_num = torch.arange(1, len(rx_mask)+1)*rx_mask
        self.tx_num = tx_num[tx_num != 0] - 1 # [ 0, 4, 8, 12, 16, 20, 24, 28, 32] # unit wavelength
        self.rx_num = rx_num[rx_num != 0] # [ 1, 2, 3, 4, 12, 13, 14, 15, 47, 48, 49, 50, 51, 52, 53, 54] # unit wavelength
        self.mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                            [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
        self.mask = self.mask.permute(1,0)
        self.pi = 3.141592653589793
        self.pi = 3.1415926535
        """
        [ 0  1  2  3  0  0  0  0  0  0  0  4  5  6  7  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  8  9
            10 11 12 13 14 15]
        [0. 0. 0. 0. 1. 0. 0. 0. 2. 0. 0. 0. 3. 0. 0. 0. 4. 0. 0. 0. 5. 0. 0. 0.
             6. 0. 0. 0. 7. 0. 0. 0. 8.]
        [ 0  1  2  3 (7*0) 4  5  6  7  (31*0) 8  9 10 11 12 13 14 15  )0
          0  1  2  3 (7*0) 4  5  6  7  (31*0)            12 13 14 15  )1
          0  1  2    (7*0) 4  5  6  7  (31*0)            12 13 14 15  )2
                     (7*0) 4  5  6  7  (31*0)            12 13 14 15  )3
                     (7*0) 4  5  6  7  (31*0)            12 13 14 15  )4
                     (7*0) 4  5  6  7  (31*0)            12 13 14 15  )5
                     (7*0) 4  5  6  7  (31*0)            12 13 14 15  )7
                     (7*0) 4  5  6     (31*0)            12 13 14 15] )8
        [  0   1   2   3  16  17  18  19  32  33  34   4   5   6   7  20  21  22
            23  36  37  38  39  52  53  54  55  68  69  70  71  84  85  86  87 100
            101 102 103 116 117 118 119 132 133 134   8   9  10  11  12  13  14  15
            28  29  30  31  44  45  46  47  60  61  62  63  76  77  78  79  92  93
            94  95 108 109 110 111 124 125 126 127 140 141 142 143]
        """

    def accelerate_power(self, volume, d_wavelength, theta, full = False):
        # directly using power equation for aoa and aod
        volume = volume[:, :, :, :-3]
        if full:
            rx_use, tx_use = [i for i in range(volume.shape[-2])], [i for i in range(volume.shape[-1])]
        else:
            rx_use = [8, 9, 10, 11, 12, 13, 14, 15]
            tx_use = [1, 3, 5, 7]
            #tx_use = [0, 2, 4, 6]
            # We dont have to use all
        # Both works.
        volume = volume[:, :, rx_use, :][:, :, :, tx_use]
        self.rx_num = self.rx_num[rx_use]; self.tx_num = self.tx_num[tx_use]

        numrange, numdoppler, numrx, numtx = volume.shape[0], volume.shape[1], volume.shape[2], volume.shape[3]
        wavelength = 1  
        d = wavelength * d_wavelength
        self.t_matrix = torch.exp(1j * (2 * self.pi * self.tx_num[:, None] * d * torch.cos(theta[None, :]) / wavelength)) #[t_dim, angles]
        self.r_matrix = torch.exp(1j * (2 * self.pi * self.rx_num[:, None] * d * torch.cos(theta[None, :]) / wavelength)) #[r_dim, angles]
        r_power_spectrum = self.r_matrix[:, :, None] * volume.permute(2,0,1,3).reshape(numrx, -1)[:, None, :] 
        r_power_spectrum = torch.sum(r_power_spectrum, dim=0) # angles, numrange*numdoppler*numtx
        r_power_spectrum = r_power_spectrum.reshape(len(theta), numrange, numdoppler, numtx).permute(3, 0, 1, 2).reshape(numtx, -1)
        t_power_spectrum = self.t_matrix[:, :, None] * r_power_spectrum[:, None, :] 
        t_power_spectrum = torch.sum(t_power_spectrum, dim=0) # angles_tx, angles_rx*numrange*numdoppler
        t_power_spectrum = t_power_spectrum.reshape(len(theta), len(theta), numrange, numdoppler)
        t_power_spectrum = t_power_spectrum.permute(2, 3, 1, 0)/144
        return t_power_spectrum
    
import scipy.fftpack as fft
from scipy.constants import c

class MVDR:
    def __init__(self, doppler_dim=64):
        # Get hyperparameter from Lua file:
        self.start_freq = 77e9
        self.slope = 79*40 
        self.duration = 40
        self.B = self.slope * self.duration * 1/100 * 10**9
        self.end_freq = self.start_freq + self.B
        self.compute_freq = (self.start_freq + self.end_freq) / 2
        self.N_range_fft = 256
        self.N_angle_fft = 256
        self.N_doppler = doppler_dim 
        # This one may be different from real
        # This one denotes dimension, we want to use 
        self.N_rx = 8
        self.N_tx = 12
        self.Use_tx = 0
        self.use_mvdr = True
        self.ranges = np.arange(1, self.N_range_fft + 1) * c / (2 * self.B) * 256 / self.N_range_fft
        self.tx_number = 0  # Assuming first TX for simplicity
        self.angles = np.arange(-90, 91)  # Angle scan
        self.Lambda = (c / self.compute_freq)
        self.D_lambda = 1/2 # The ratio D/Lambda
        self.steering_vectors = np.exp(-1j * 2 * np.pi * self.D_lambda * np.outer(np.sin(np.radians(self.angles)), np.arange(self.N_rx)))
        # 256, 8

    def acc_process_2(self, range_fft):
        #range_fft = range_fft[:, :self.N_doppler, 8:, self.Use_tx]
        range_fft = range_fft[:, :, :, 0]
        range_fft -= np.mean(range_fft, axis=1, keepdims=True)
        invR = np.array([np.linalg.inv(np.cov(batch, rowvar=False) + np.eye(self.N_rx) * 1e-6) for batch in range_fft])
        # invR: N_range, RX, RX; self.steering_vectors: N_angle, RX
        if self.use_mvdr:
            spectrum = np.array([1 / np.einsum('na,na->n', np.conj(self.steering_vectors) @ _invR_, self.steering_vectors)  for _invR_ in invR])
        else: # FFT
            spectrum = fft.fft(range_fft[:, 0, :], self.N_angle_fft, axis=1)
            spectrum = np.fft.fftshift(spectrum, axes=1); spectrum = spectrum[:, ::-1]
            spectrum = spectrum[:, int(self.N_angle_fft/2 - 90) : int(self.N_angle_fft/2 + 91)]
        return np.abs(spectrum)
