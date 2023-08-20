from matplotlib import pyplot as plt
import numpy as np

class BE_Viz:
    def __init__(self, BE_waveform):
        for key, value in BE_waveform.__dict__.items():
            setattr(self, key, value)
            
    def plot_fft(self, signal=None, x_range = None, AI_rate = None, y_range = None):
        """
        Compute and plot the FFT magnitude of a signal.

        Parameters:
        - signal: Time-domain signal
        - fs: sampling frequency (Hz)
        """

        if signal is None:
            signal = self.BE_wave

        if AI_rate is None:
            rate = self.AO_rate
        else:
            rate = AI_rate

        N = len(signal)

        # Compute the FFT
        freqs = np.fft.fftfreq(N, 1/rate)
        fft_vals = np.fft.fft(signal)

        # Shift the FFT values so that the center frequency is at 0 Hz
        freqs = np.fft.fftshift(freqs)
        fft_magnitude = np.fft.fftshift(np.abs(fft_vals))
        
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, fft_magnitude)
        plt.title("Magnitude Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        if x_range is not None:
                    plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(0, y_range)
        plt.show()

    def plot_waveform(self, signal=None):
        if signal is None:
            signal = self.BE_wave

        plt.figure(figsize=(10, 6))
        plt.plot(signal)
        plt.title("BE Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def plot_merged_waveform(self, signal=None):
        if signal is None:
            signal = self.SS_wave

        plt.figure(figsize=(10, 6))
        plt.plot(signal)
        plt.title("BE Merged_Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()