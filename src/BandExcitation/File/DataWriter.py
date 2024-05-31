from dataclasses import dataclass 
import numpy as np

@dataclass
class DataConverter:
    be_measurement: object
    
    def __post_init__(self):
        pass
    
    def get_spectroscopic_dimension(self):
        frequency = self.get_spectroscopic_freq()
        DC_field = self.get_DC_field()
        Voltage_state = self.get_measurement_voltage_state()
        pass

    def get_spectroscopic_freq(self):
        self.update_binning(self.be_measurement.get_simulated_BE_measurement())
        return np.tile(self.binned_freqs,self.be_measurement.BE_num_bins*self.be_measurement.spectroscopic_points)

    def get_DC_field(self):
        if self.be_measurement.spectroscopic_measurement_state == "on and off":
            multiple = 2
        else: multiple = 1

        return self.tile_and_offset(np.zeros(self.be_measurement.BE_num_bins*multiple),
                                    self.be_measurement.be_spectroscopy.DC_waveform)

    def get_measurement_voltage_state(self):
        if self.be_measurement.spectroscopic_measurement_state == "on and off":
            vec = np.array([1,0])
        elif self.be_measurement.spectroscopic_measurement_state == "on":
            vec = np.array([1])
        elif self.be_measurement.spectroscopic_measurement_state == "off":
            vec = np.array([0])
        vec_binned = self.tile_and_offset(np.zeros(self.be_measurement.BE_num_bins), vec)
        return(self.tile_and_offset(vec_binned, np.zeros(len(self.be_measurement.be_spectroscopy.DC_waveform))))


    @staticmethod
    def tile_and_offset(a, b):
        """
        Tile vector `a` at each index specified by values in vector `b`.
        The length of the resulting array will be len(a) * len(b).

        Parameters:
        - a (numpy.ndarray): Vector to be tiled.
        - b (numpy.ndarray): Vector specifying the offsets.

        Returns:
        - numpy.ndarray: The resulting tiled and offset array.
        """
        
        # Step 1 & 2: Tile 'a' len(b) times
        tiled_a = np.tile(a, len(b))

        # Step 3: Reshape to a 2D array with each row being a tiled 'a'
        reshaped_a = tiled_a.reshape(len(b), len(a))

        # Step 4: Create an array by repeating each element in 'b', len(a) times
        repeated_b = np.repeat(b, len(a)).reshape(len(b), len(a))

        # Step 5: Add the reshaped_a and repeated_b
        result_2D = reshaped_a + repeated_b

        # Step 6: Flatten the resulting 2D array
        result = result_2D.flatten()
        
        return result
    
    @staticmethod
    def cycle(bandwidth):
        half_bandwidth = int(bandwidth//2)
        cycle = np.concatenate((np.zeros(half_bandwidth),np.ones(half_bandwidth)))
        return cycle


    def update_binning(self, signal, **kwargs):

        N = len(signal)
        
        # get BE frequencies
        freqs = self.BE_frequencies(N, self.be_measurement.AI_sample_rate)
        
        if self.be_measurement.BE_num_bins is not None:
            # get the masked regions
            #self.inds = self.extract_freq_range(freqs, self.be_measurement.BE_freq_range, self.be_measurement.BE_num_bins)
            self.inds = self.extract_freq_range(freqs, self.be_measurement.BE_freq_range)

            
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

    @staticmethod
    def BE_frequencies(number_of_points, sampling_frequency):
        
        freqs = np.fft.fftfreq(number_of_points, 1/sampling_frequency)
        freqs = np.fft.fftshift(freqs)
        
        return freqs
    
    @staticmethod
    def BE_FFT(signal, transfer_function = None):
        
        if transfer_function is None:
            FFT_ = np.fft.fft(signal) / len(signal)  # FFT and normalization
        else: 
            FFT_ = np.fft.fft(signal) / np.fft.fft(transfer_function) / len(signal)
        FFT_ = np.fft.fftshift(FFT_) # Shift FFT and keep one side
        return FFT_
    
    @staticmethod
    def extract_freq_range(freqs, _range):

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
    

    # def get_noise_floor(fft_data, tolerance):
    #     """
    #     Calculate the noise floor from the FFT data. Algorithm originally written by Mahmut Okatan Baris

    #     Parameters
    #     ----------
    #     fft_data : 1D or 2D complex numpy array
    #         Signal in frequency space (ie - after FFT shifting) arranged as (channel or repetition, signal)
    #     tolerance : unsigned float
    #         Tolerance to noise. A smaller value gets rid of more noise.
            
    #     Returns
    #     -------
    #     noise_floor : 1D array-like
    #         One value per channel / repetition

    #     """

    #     fft_data = np.atleast_2d(fft_data)
    #     # Noise calculated on the second axis

    #     noise_floor = []

    #     fft_data = np.abs(fft_data)
    #     num_pts = fft_data.shape[1]

    #     for amp in fft_data:

    #         prev_val = np.sqrt(np.sum(amp ** 2) / (2 * num_pts))
    #         threshold = np.sqrt((2 * prev_val ** 2) * (-np.log(tolerance)))

    #         residual = 1
    #         iterations = 1

    #         while (residual > 10 ** -2) and iterations < 50:
    #             amp[amp > threshold] = 0
    #             new_val = np.sqrt(np.sum(amp ** 2) / (2 * num_pts))
    #             residual = np.abs(new_val - prev_val)
    #             threshold = np.sqrt((2 * new_val ** 2) * (-np.log(tolerance)))
    #             prev_val = new_val
    #             iterations += 1

    #         noise_floor.append(threshold)

    #     return noise_floor
