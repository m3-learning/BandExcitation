import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
import warnings
from . import AWG
from ..Util.core import add_kwargs, inherit_attributes

class BEWaveform():
    def __init__(
        self,
        BE_ppw,
        BE_rep,
        BE_amplitude=1,
        center_freq=500e3,
        bandwidth=60e3,
        wave="chirp",
        waveform_time=4e-3,
        BE_smoothing=125,
        BE_phase_var=None,
        chirp_direction="up",
    ) -> None:
        
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
        self.BE_amplitude = BE_amplitude
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
        self.BE_wave = self.BE_amplitude*envelope * np.sin(2 * np.pi * t_vector * w_chirp)

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
        
    def build_DC_wave(self):
        if self.type == "switching spectroscopy":
            self.switching_spectroscopy()
        if self.type == "BE Line":
            self.BE_line()
        else:
            raise ValueError("Invalid spectroscopy type")
        
    def BE_line(self, **kwargs):
        
        add_kwargs(self, **kwargs)
        
        # creates the waveform by replicating the start value.
        self.DC_waveform = [self.start] * self.points_per_cycle
        

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
