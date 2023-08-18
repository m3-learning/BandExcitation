import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import warnings
from BEWaveform import BE_Spectroscopy, BE_Viz


class AWG:
    def __init__(self, platform="PXI-5413") -> None:
        self.platform = platform


# class DAQ:
#     def __init__(self, platform="PXI-5412", AO_rates=None) -> None:
#         # sets the DAQ platform
#         self.platform = platform
#         self.AWG_Freq = AWG_Freq

#     def set_AO_rate(self, BE_ppw):
#         AO_rate_req = (
#             BE_ppw / self.AWG_Freq
#         )  # requested frequency of the function generator
#         self.AO_rate = self.AO_rates[np.argmin(abs(self.AO_rates - AO_rate_req))]
#         self.BE_step_t = BE_ppw / self.AO_rate
#         self.w_vec = np.arange(
#             -self.AO_rate / 2,
#             self.AO_rate / 2 + self.AO_rate / (BE_ppw - 1),
#             self.AO_rate / (BE_ppw - 1),
#         )


class BEWaveform(AWG):
    def __init__(
        self,
        BE_ppw,
        BE_rep,
        center_freq=500e3,
        bandwidth=60e3,
        wave="chirp",
        platform="PXI-5412",
        waveform_time=4e-3,
        BE_smoothing=125,
        BE_phase_var=None,
        chirp_direction="up",
    ) -> None:
        super().__init__(platform=platform)

        self.BE_ppw = 2**BE_ppw  # number of points per waveform in orders of 2
        self.BE_rep = BE_rep
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.wave = wave
        self.BE_smoothing = BE_smoothing
        self.chirp_direction = chirp_direction
        self.BE_phase_var = BE_phase_var
        self.waveform_time = waveform_time
        self.AO_rate = self.BE_ppw / self.waveform_time
        self.build_BE()

    def build_BE(self):
        freq1 = self.center_freq - self.bandwidth / 2
        freq2 = self.center_freq + self.bandwidth / 2

        if self.wave == "chirp":
            return self.chirp(freq1, freq2)
        elif self.wave == "sinc":
            return self.sinc(freq1, freq2)
        else:
            raise ValueError("Invalid waveform type")

    def chirp(self, freq1, freq2):
        t_vector = np.linspace(0, self.waveform_time, self.BE_ppw)
        m = ((freq2 - freq1) / self.waveform_time) / 2  # slope of frequency change
        w_chirp = m * t_vector + freq1  # vector for linear frequency change with time
        chirp_smoothing = (
            4 * self.waveform_time * self.BE_smoothing / 1e4
        )  # smoothing factor for error function
        envelope_a = (
            1 + erf((t_vector - chirp_smoothing * 2) / chirp_smoothing)
        ) / 2  # first half of erf window
        envelope_b = (
            1
            + erf(
                (t_vector + chirp_smoothing * 2 - self.waveform_time) / chirp_smoothing
            )
        ) / 2  # second half of erf window
        envelope = envelope_a - envelope_b  # erf window
        self.BE_wave = envelope * np.sin(2 * np.pi * t_vector * w_chirp)

        if not isinstance(self.BE_rep, int):
            raise ValueError("BE_rep must be an integer")
        elif self.BE_rep > 1:
            self.BE_wave = np.tile(self.BE_wave, self.BE_rep)
        elif self.BE_rep == 1:
            self.BE_wave = self.BE_wave
        else:
            raise ValueError("BE_rep must be greater than 0")

        if self.chirp_direction == "up":
            pass
        elif self.chirp_direction == "down":
            self.BE_wave = self.BE_wave[::-1]
        else:
            raise ValueError("Invalid chirp direction")

        return self.BE_wave

    def sinc(self, freq1, freq2):
        raise NotImplementedError("Sinc waveform not implemented yet")

    @property
    def BE_ppw(self):
        return self._BE_ppw

    @BE_ppw.setter
    def BE_ppw(self, value):
        self._BE_ppw = value

    @property
    def waveform_time(self):
        return self._waveform_time

    @waveform_time.setter
    def waveform_time(self, value):
        self._waveform_time = value


class BE_Viz:
    def __init__(self, BE_waveform):
        for key, value in BE_waveform.__dict__.items():
            setattr(self, key, value)
            
    def plot_fft(self, signal=None, x_range = None):
        """
        Compute and plot the FFT magnitude of a signal.

        Parameters:
        - signal: Time-domain signal
        - fs: sampling frequency (Hz)
        """

        if signal is None:
            signal = self.BE_wave

        N = len(signal)

        # Compute the FFT
        freqs = np.fft.fftfreq(N, 1/self.AO_rate)
        fft_vals = np.fft.fft(signal)

        # Shift the FFT values so that the center frequency is at 0 Hz
        freqs = np.fft.fftshift(freqs)
        fft_magnitude = np.fft.fftshift(np.abs(fft_vals))

        plt.figure(figsize=(10, 6))
        plt.plot(freqs, fft_magnitude)
        plt.title("Magnitude Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)

        if x_range is not None:
            plt.xlim(x_range)

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

def add_kwargs(obj, check = True, **kwargs):
    for key, value in kwargs.items():
        if not hasattr(obj, key) and check == True:
            warnings.warn(f"Attribute '{key}' does not exist in the object. Setting it now.")
        setattr(obj, key, value)
        
def inherit_attributes(source_obj, target_obj):
    for key, value in source_obj.__dict__.items():
        setattr(target_obj, key, value)
        
def phase_shift_waveform(waveform, shift_radians):
    # get length of waveform
    n = len(waveform)
    
    # Convert the phase shift from radians to points
    shift_points = int(shift_radians / (2 * np.pi) * n) % n
    
    print(shift_points)

    # Apply the phase shift
    return np.roll(waveform, shift_points)

class Spectroscopy():
    def __init__(
        self,
        **kwargs
    ) -> None:
        
        add_kwargs(self, check=False, **kwargs)
        super().__init__()
        
        # self.BE_ppw = 2**BE_ppw
        # self.n_read = n_read
        # self.SS_steps_per_cycle = SS_steps_per_cycle
        # self.build_SS()
        
    def build_DC_wave(self):
        if self.type == "switching spectroscopy":
            self.switching_spectroscopy()
        else:
            raise ValueError("Invalid spectroscopy type")

    def switching_spectroscopy(self, **kwargs):
        
        add_kwargs(self, **kwargs)
        
        n = self.points_per_cycle + 1

        # Lengths for each segment within the cycle
        length_up = (self.max - self.start)
        length_down = (self.max - self.min)
        length_return = (self.start - self.min)

        # Points for each segment within the cycle
        x_up = np.linspace(0, length_up, int(n * length_up / (length_up + length_down + length_return)))
        y_up = self.start + x_up

        x_down = np.linspace(0, length_down, int(n * length_down / (length_up + length_down + length_return)))
        y_down = self.max - x_down

        x_return = np.linspace(0, length_return, int(n * length_return / (length_up + length_down + length_return)))
        y_return = self.min + x_return

        # Concatenate the points
        x_cycle = np.concatenate([x_up, x_up[-1] + x_down, x_up[-1] + x_down[-1] + x_return])
        y_cycle = np.concatenate([y_up, y_down, y_return])[:-1]
        
        if self.phase_shift is not None:
            y_cycle = phase_shift_waveform(y_cycle, self.phase_shift)

        # Repeat the cycle for cycles
        self.DC_waveform = np.tile(y_cycle, self.cycles)

            
            # n_cycle = (2*self.n_read)  * self.SS_steps_per_cycle
            # interp_factor = (2*self.BE_ppw )/ self.n_read
            # n_step = 2*self.n_read 
            # n_write_vec = np.arange(self.n_read, n_cycle // 4, n_step)

            # dc_amp_vec_1 = np.arange(0, self.SS_steps_per_cycle/4, 1)
            # dc_amp_vec_2 = np.arange(self.SS_steps_per_cycle/4, -1,-1)
            
            # y_positive,y_negative = np.zeros(n_cycle // 4 - 1),np.zeros(n_cycle // 4 - 1)
            
            # for step_count in range(int(self.SS_steps_per_cycle/4)):
            #     n_sub_shifted = np.arange(1, n_cycle // 4) - n_write_vec[step_count]
            #     yk = 0.5 * (erf(n_sub_shifted ) - erf((n_sub_shifted - self.n_read)))
            #     y_positive += dc_amp_vec_1[step_count] * yk
            #     y_negative += dc_amp_vec_2[step_count] * yk       
            # n = np.arange(n_cycle - 4)
            # self.SS_wave  = np.interp(np.arange(1, int(n_cycle * interp_factor) + 1) / interp_factor, n, np.concatenate((y_positive,y_negative, -y_positive, -y_negative)))

            # # getting the points where BE will be placed
            # n_write_vec = np.concatenate((np.arange(self.n_read, n_cycle // 2, n_step),np.arange(self.n_read, n_cycle // 2 - self.n_read - 1, n_step) + n_cycle // 2))
            # n_read_vec = np.concatenate(([1 / interp_factor], n_write_vec + (self.n_read )))
            # ni_read_vec = ((self.SS_steps_per_cycle*100)+n_read_vec * interp_factor)  
            # ni_write_vec = ((self.SS_steps_per_cycle*100)+n_write_vec * interp_factor) 

            # self.SS_read_vec = np.round(ni_read_vec + np.round(interp_factor / 2))
            # SS_write_vec = np.round(ni_write_vec + np.round( interp_factor / 2))
            # self.SS_write_vec = np.concatenate((SS_write_vec, [SS_write_vec[-1] + np.round(n_step * interp_factor)]))
            
class BE_Spectroscopy(BEWaveform,Spectroscopy):

    def __init__(
        self,
        BE_ppw,
        BE_rep,
        type = "switching spectroscopy",
        start = None, 
        max = None,
        min = None,
        cycles = None,
        points_per_cycle = None,
        phase_shift = None,
        center_freq=500e3,
        bandwidth=60e3,
        wave="chirp",
        platform="PXI-5412",
        waveform_time=4e-3,
        BE_smoothing=125,
        BE_phase_var=None,
        chirp_direction="up",
        measurement_state = "on and off", 
        measurement_state_offset = 0,
        
    ) -> None:
        
        super().__init__(
                    BE_ppw,
                    BE_rep,
                    center_freq,
                    bandwidth,
                    wave,
                    platform,
                    waveform_time,
                    BE_smoothing,
                    BE_phase_var,
                    chirp_direction,
                    )
        
        self.measurement_state = measurement_state
        self.measurement_state_offset = measurement_state_offset
        
        
        # self.spectroscopic_waveform = spectroscopic_waveform
        Spectroscopy_ = Spectroscopy(type=type, start=start, max=max, min=min, cycles=cycles, points_per_cycle=points_per_cycle, phase_shift=phase_shift)
        
        inherit_attributes(Spectroscopy_, self)
        
        self.build_DC_wave()
        self.build_spectroscopy_waveform()
        
        self.merge_low_and_high_freq(self.DC_wave, self.BE_wave)
        
    @staticmethod
    def insert_constant(arr, constant):
        result = []
        for i, item in enumerate(arr):
            result.append(item)
            result.append(constant)
        return result
    
    def merge_low_and_high_freq(self, DC_wave, AC_wave):
        # Create a result array with enough space to accommodate all replications
        result = np.zeros(len(DC_wave) * len(AC_wave))

        # Replicate the entire array AC_wave at each DC value
        for i, offset in enumerate(DC_wave):
            start_idx = i * len(AC_wave)
            end_idx = start_idx + len(AC_wave)
            result[start_idx:end_idx] = AC_wave + offset
            
        self.cantilever_excitation_waveform = result
    
    @property
    def cantilever_excitation_waveform(self):
        return self._cantilever_excitation_waveform
    
    @property
    def cantilever_excitation_length(self):
        return self._cantilever_excitation_length
        
    @cantilever_excitation_waveform.setter
    def cantilever_excitation_waveform(self, value):
        self._cantilever_excitation_waveform = value
        self._cantilever_excitation_length = len(value)
    
    
    def build_spectroscopy_waveform(self):
    
        if self.measurement_state == "on and off":
            self.DC_wave = self.insert_constant(self.DC_waveform, self.measurement_state_offset)
        else:
            self.DC_wave = self.DC_waveform
