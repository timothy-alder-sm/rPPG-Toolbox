"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, power2db etc.
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags

from scipy.interpolate import Akima1DInterpolator

import heartpy as hp
from copy import deepcopy

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def compute_power_spectrum(signal, fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


def _calculate_hr(signal, fs, low_pass=0.75, high_pass=2.5, method='FFT'):
    if method == 'FFT':
        """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""       
        signal = np.expand_dims(signal, 0)
        N = _next_power_of_2(signal.shape[1])
        f_ppg, pxx_ppg = scipy.signal.periodogram(signal, fs=fs, nfft=N, detrend=False)
        fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
        mask_ppg = np.take(f_ppg, fmask_ppg)
        mask_pxx = np.take(pxx_ppg, fmask_ppg)
        fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
        return fft_hr
    elif method == 'Peak':
        """Calculate heart rate based on PPG using peak detection."""
        ppg_peaks, _ = scipy.signal.find_peaks(signal)
        hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
        return hr_peak
    elif method == 'HeartPy':
        print(signal, signal.shape)
        _, hp_process = hp.process(np.array(signal), fs)
        
        return hp_process['breathingrate'] * 60
    else:
        raise ValueError('Unsupported method for HR calculation.')


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)
def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1**2)
    signal_power_hm2 = np.sum(pxx_harmonic2**2)
    signal_power_rem = np.sum(pxx_remainder**2)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
        
    return SNR


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)

    # bandpass filter between [0.75, 2.5] Hz
    # equals [45, 150] beats per min
    low_pass = 0.13
    high_pass = 0.5
    
    if use_bandpass:
        [b, a] = butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    
    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, SNR, macc


# Resp Metrics
def _calculate_fft_rr(resp_signal, fs=30, low_pass=0.13, high_pass=0.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    resp_signal = np.expand_dims(resp_signal, 0)
    N = _next_power_of_2(resp_signal.shape[1])
    f_resp, pxx_resp = scipy.signal.periodogram(resp_signal, fs=fs, nfft=N, detrend=False)
    fmask_resp = np.argwhere((f_resp >= low_pass) & (f_resp <= high_pass))
    mask_resp = np.take(f_resp, fmask_resp)
    mask_pxx = np.take(pxx_resp, fmask_resp)
    fft_rr = np.take(mask_resp, np.argmax(mask_pxx, 0))[0] * 60
    return fft_rr


def _calculate_peak_rr(resp_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    resp_peaks, _ = scipy.signal.find_peaks(resp_signal)
    rr_peak = 60 / (np.mean(np.diff(resp_peaks)) / fs)
    return rr_peak


def calculate_resp_metrics_per_video(predictions, labels, fs=30, diff_flag=False, use_bandpass=True, rr_method='FFT'):
    """Calculate video-level RR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of Resp signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
        
    
    if use_bandpass:
        low_pass = 0.13
        high_pass = 0.5
        # bandpass filter between [0.13, 0.5] Hz
        # equals [8, 30] breaths per min
        [b, a] = butter(1, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    if rr_method == 'FFT':
        rr_pred = _calculate_fft_rr(predictions, fs=fs)
        rr_label = _calculate_fft_rr(labels, fs=fs)
    elif rr_method == 'Peak':
        rr_pred = _calculate_peak_rr(predictions, fs=fs)
        rr_label = _calculate_peak_rr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your RR.')
    SNR = _calculate_SNR(predictions, rr_label, fs=fs, low_pass=0.13, high_pass=0.5)
    return rr_label, rr_pred, SNR