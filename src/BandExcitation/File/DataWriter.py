from dataclasses import dataclass 
import numpy as np

@dataclass
class DataConverter:
    be_measurement: object
    
    def __post_init__(self):
        pass
    
    def get_spectroscopic_dimension(self):
        # this calls all the subfunctions to get the spectroscopic dimension
        pass

    def update_binning(self, signal, **kwargs):

        N = len(signal)
        
        # get BE frequencies
        freqs = self.BE_frequencies(N, self.be_measurement.AI_sample_rate)
        
        if self.be_measurement.BE_num_bins is not None:
            # get the masked regions
            self.inds = self.extract_freq_range(freqs, self.be_measurement.BE_freq_range, self.be_measurement.BE_num_bins)
            # Verify if number of bins is too large
            if self.be_measurement.BE_num_bins > len(self.inds):
                raise ValueError("Number of bins is greater than the number of FFT points within the frequency range.")
        else:
            self.inds = np.where(freqs>=0)
            
        freqs_filtered = freqs[self.inds]
        
        self.binned_freqs = self.BE_bin(freqs_filtered, self.be_measurement.BE_num_bins)
            
    
    def bin_signal(self, signal, **kwargs):
        
        if not hasattr(self, 'inds'):
            self.update_binning(signal, **kwargs)
        
        # Compute the Normalized FFT
        FFT_ = self.BE_FFT(signal)

        # Filter the FFT
        FFT_filtered = FFT_[self.inds]
        
        binned_complex = self.BE_bin(FFT_filtered, self.be_measurement.BE_num_bins)

        return binned_complex
       
    @staticmethod
    def BE_bin(signal, num_bins):

        """
        Downsample a 1D array signal to num_bins, including endpoints.

        Parameters:
            signal (array-like): The input 1D signal
            num_bins (int): The number of bins to downsample to
        
        Returns:
            np.ndarray: The downsampled signal
        """
        # Validate inputs
        if num_bins <= 0:
            raise ValueError("The number of bins must be greater than 0.")
        if len(signal) < num_bins:
            raise ValueError("The number of bins must be less than or equal to the length of the signal.")

        len_signal = len(signal)
        bin_size = len_signal / num_bins
        
        # Generate index array for bin edges
        idx = np.linspace(0, len_signal, num_bins + 1)
        
        # Truncate index array to integers and ensure last index is exactly len(signal)
        idx = np.floor(idx).astype(int)
        idx[-1] = len_signal
        
        # Compute means using NumPy's advanced indexing and diff to find bin sizes
        ave_signal = np.add.reduceat(signal, idx[:-1]) / np.diff(idx)
    
        return ave_signal
        
        # # Reshape and average
        # reshaped_signal = signal.reshape(num_bins, -1)
        # avg_signal = np.mean(reshaped_signal, axis=1)
        
        # return avg_signal

    @staticmethod
    def BE_frequencies(number_of_points, sampling_frequency):
        
        freqs = np.fft.fftfreq(number_of_points, 1/sampling_frequency)
        freqs = np.fft.fftshift(freqs)
        
        return freqs
    
    @staticmethod
    def BE_FFT(signal):
        
        FFT_ = np.fft.fft(signal) / len(signal)  # FFT and normalization
        FFT_ = np.fft.fftshift(FFT_) # Shift FFT and keep one side
        return FFT_
    
    

    # def BE_freq_bins(self, **kwargs):
    #     BE_wave = self.be_measurement.get_simulated_BE_measurement()
    #     freqs = np.fft.fftfreq(len(BE_wave), 1/self.be_measurement.AI_sample_rate)
    #     self.ind = self.extract_freq_range(freqs, **kwargs)
    #     self.freqs = freqs[self.ind]

    @staticmethod
    def extract_freq_range(freqs, _range, num_bins):

        range_ = (_range[0], _range[1])
        print(range_)
        indices = np.where((freqs >= range_[0]) & (freqs <= range_[1]))[0]
        return indices
        
    @property
    def frequency_dimension(self):
        # this gets the frequency bins from the FFT
        pass
    
    @property
    def DC_offset_dimension(self):
        pass
    
    def get_field_dimension(self):
        pass
    
    @property
    def cycle_dimension(self):
        pass
    
    @property
    def binned_freqs(self):
        return self._binned_freqs
    
    @binned_freqs.setter
    def binned_freqs(self, value):
        self._binned_freqs = value
    

    
