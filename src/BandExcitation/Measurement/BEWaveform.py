import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from BandExcitation.Util.Core import add_kwargs, inherit_attributes


class BEWaveform:
    """
    BEWaveform Object for the BEWaveform
    """

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
        chirp_direction="up",
        delay=(0, 0),
    ):
        """
        __init__ Initialization function

        Args:
            BE_ppw (int): Points per BE wave
            BE_rep (int): Number of repetitions per BE wave
            BE_amplitude (int, optional): Amplitude of the BE Wave. Defaults to 1.
            center_freq (int, optional): Center resonance frequency of the BE wave. Defaults to 500e3.
            bandwidth (int, optional): Bandwidth of the BE wave. Defaults to 60e3.
            wave (str, optional): Type of the BE Wave. Defaults to "chirp".
            waveform_time (float, optional): time of the BE wave in seconds. Defaults to 4e-3.
            BE_smoothing (int, optional): Factor that smooths the BE band. Defaults to 125.
            chirp_direction (str, optional): sets the direction of the chirp excitation. Defaults to "up".
            delay (tuple, optional): delay added before or after the BE wave. Defaults to (0,0).
        """

        # initializations
        self.BE_ppw = BE_ppw
        self.BE_rep = BE_rep
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.wave = wave
        self.BE_smoothing = BE_smoothing
        self.chirp_direction = chirp_direction
        self.waveform_time = waveform_time
        self.delay = delay
        self.AO_rate = self.BE_ppw / (self.waveform_time + np.sum(self.delay))
        self.BE_amplitude = BE_amplitude

        # builds ths BE wave
        self.build_BE()

    def build_BE(self):
        """
        build_BE function that builds the BE wave

        Raises:
            ValueError: error if the type of wave is not in the available types
        """

        freq1 = self.center_freq - self.bandwidth / 2
        freq2 = self.center_freq + self.bandwidth / 2

        if self.wave == "chirp":
            wave = self.chirp(freq1, freq2)
        elif self.wave == "sinc":
            wave = self.sinc(freq1, freq2)
        else:
            raise ValueError("Invalid waveform type")

        # adds the delay to the waveform if required
        if self.delay != (0, 0):
            wave = np.concatenate(
                (
                    np.zeros(int(self.delay[0] * self.AO_rate)),
                    wave,
                    np.zeros(int(self.delay[1] * self.AO_rate)),
                )
            )

        self.BE_wave = wave

    def chirp(self, freq1, freq2):
        """
        chirp function to create a BE chirp

        Args:
            freq1 (int): lower bound frequency of the BE chirp
            freq2 (int): upper bound frequency of the BE chirp

        Raises:
            ValueError: BE repetitions must be an integer
            ValueError: BE repetitions must be greater than 0
            ValueError: Invalid chirp direction

        Returns:
            np.array: BE waveform as an array
        """

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
        self.BE_wave = (
            self.BE_amplitude * envelope * np.sin(2 * np.pi * t_vector * w_chirp)
        )

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
        """
         Sinc function to create a BE sinc

        Raises:
            NotImplementedError: Currently not implemented
        """

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
    """
    phase_shift_waveform function that shifts the phase of a waveform

    Returns:
        np.array : waveform shifter by the phase shift
    """

    # get length of waveform
    n = len(waveform)

    # Convert the phase shift from radians to points
    shift_points = int(shift_radians / (2 * np.pi) * n) % n

    print(shift_points)

    # Apply the phase shift
    return np.roll(waveform, shift_points)


class Spectroscopy:
    def __init__(self, **kwargs) -> None:
        """
        Class that builds a Band Excitation Spectroscopy waveform
        """

        add_kwargs(self, check=False, **kwargs)
        super().__init__()

    def build_DC_wave(self):
        """
        build_DC_wave Builds the DC waveform for the spectroscopy

        Raises:
            ValueError: Invalid spectroscopy type
        """

        if self.type == "switching spectroscopy":
            self.switching_spectroscopy()
        elif self.type == "BE Line":
            self.BE_line()
        else:
            raise ValueError("Invalid spectroscopy type")

    def BE_line(self, **kwargs):
        """
        Function that builds the DC waveform for a BE line
        """

        add_kwargs(self, **kwargs)

        # creates the waveform by replicating the start value.
        self.DC_waveform = [self.start] * self.points_per_cycle

    def switching_spectroscopy(self, **kwargs):
        """
         Function that builds a bipolar triangle waveform for a switching spectroscopy
         
         .. math::
            f(x) =
            \\begin{cases} 
                \\text{{self.start}} + x & \\text{{if }} 0 \leq x < \\text{{length\_up}} \\\\
                \\text{{self.max}} - (x - \\text{{length\_up}}) & \\text{{if }} \\text{{length\_up}} \leq x < \\text{{length\_up}} + \\text{{length\_down}} \\\\
                \\text{{self.min}} + (x - \\text{{length\_up}} - \\text{{length\_down}}) & \\text{{if }} \\text{{length\_up}} + \\text{{length\_down}} \leq x < \\text{{length\_up}} + \\text{{length\_down}} + \\text{{length\_return}} \\\\
                f(x \\mod (\\text{{length\_up}} + \\text{{length\_down}} + \\text{{length\_return}})) & \\text{{otherwise}}
            \\end{cases}

        Returns:
            np.array: waveform as an np array
        """

        add_kwargs(self, **kwargs)

        n = self.points_per_cycle + 1

        # Lengths for each segment within the cycle
        length_up = self.max - self.start
        length_down = self.max - self.min
        length_return = self.start - self.min

        # Points for each segment within the cycle
        x_up = np.linspace(
            0, length_up, int(n * length_up / (length_up + length_down + length_return))
        )
        y_up = self.start + x_up

        x_down = np.linspace(
            0,
            length_down,
            int(n * length_down / (length_up + length_down + length_return)),
        )
        y_down = self.max - x_down

        x_return = np.linspace(
            0,
            length_return,
            int(n * length_return / (length_up + length_down + length_return)),
        )
        y_return = self.min + x_return

        # Concatenate the points
        x_cycle = np.concatenate(
            [x_up, x_up[-1] + x_down, x_up[-1] + x_down[-1] + x_return]
        )
        y_cycle = np.concatenate([y_up, y_down, y_return])[:-1]

        if self.phase_shift is not None:
            y_cycle = phase_shift_waveform(y_cycle, self.phase_shift)

        # Repeat the cycle for cycles
        self.DC_waveform = np.tile(y_cycle, self.cycles)


class BE_Spectroscopy(BEWaveform, Spectroscopy):

    """
    Function that build the entire BE spectroscopy waveform
    """

    def __init__(
        self,
        BE_ppw,
        BE_rep,
        BE_amplitude = 1,
        type="switching spectroscopy",
        start=None,
        max=None,
        min=None,
        cycles=None,
        points_per_cycle=None,
        phase_shift=None,
        center_freq=500e3,
        bandwidth=60e3,
        wave="chirp",
        waveform_time=4e-3,
        BE_smoothing=125,
        chirp_direction="up",
        measurement_state="on and off",
        measurement_state_offset=0,
        **kwargs
    ):
        """
        __init__ Init for BE Spectroscopy class

        Args:
            BE_ppw (int): points per wave
            BE_rep (int): repetitions of the BE wave
            BE_amplitude (int): amplitude of the BE wave
            type (str, optional): type of DC waveform. Defaults to "switching spectroscopy".
            start (float, optional): starting voltage. Defaults to None.
            max (float, optional): maximum voltage. Defaults to None.
            min (float, optional): minimum voltage. Defaults to None.
            cycles (int, optional): number of cycles through the waveform. Defaults to None.
            points_per_cycle (int, optional): number of points per DC cycle. Defaults to None.
            phase_shift (float, optional): DC waveform phase shift. Defaults to None.
            center_freq (float, optional): center drive frequency of the cantilever. Defaults to 500e3.
            bandwidth (float, optional): bandwidth of the BE excitation. Defaults to 60e3.
            wave (str, optional): type of BE wave. Defaults to "chirp".
            waveform_time (float, optional): time for the BE wave. Defaults to 4e-3.
            BE_smoothing (float, optional): Smoothing factor applied to smooth the BE band. Defaults to 125.
            chirp_direction (str, optional): direction of the chirp wave. Defaults to "up".
            measurement_state (str, optional): sets if measurements are conducted at on and off voltage states. Defaults to "on and off".
            measurement_state_offset (int, optional): Sets the voltage offset for the off measurement state. Defaults to 0.
        """

        super().__init__(
            BE_ppw,
            BE_rep,
            BE_amplitude,
            center_freq=center_freq,
            bandwidth=bandwidth,
            wave = wave,
            waveform_time = waveform_time,
            BE_smoothing=BE_smoothing,
            chirp_direction=chirp_direction,
        )

        self.measurement_state = measurement_state
        self.measurement_state_offset = measurement_state_offset

        # self.spectroscopic_waveform = spectroscopic_waveform
        Spectroscopy_ = Spectroscopy(
            type=type,
            start=start,
            max=max,
            min=min,
            cycles=cycles,
            points_per_cycle=points_per_cycle,
            phase_shift=phase_shift,
        )

        inherit_attributes(Spectroscopy_, self)

        self.build_DC_wave()
        
        
        # If BE Line, set the BE_wave to the DC_waveform
        if type == "BE Line":
            self.cantilever_excitation_waveform = self.BE_wave
            
        # If switching spectroscopy, merge the DC and BE waveforms
        else:
            self.build_spectroscopy_waveform()
            self.merge_low_and_high_freq(self.DC_wave, self.BE_wave)

    @staticmethod
    def insert_constant(arr, constant):
        """
        insert_constant Utility function that inserts a constant between each element of an array

        Args:
            arr (np.array): initial array
            constant (float): constant value to intersperse between each element of an array

        Returns:
            np.array: array with interspersed constant
        """

        result = []
        for i, item in enumerate(arr):
            result.append(item)
            result.append(constant)
        return result

    def merge_low_and_high_freq(self, DC_wave, AC_wave):
        """
        merge_low_and_high_freq Function that combines the DC and AC waveforms

        Args:
            DC_wave (np.array): switching spectroscopy waveform
            AC_wave (np.array): BE waveform used to measure the cantilever response
        """
        
        # Create a result array with enough space to accommodate all replications
        result = np.zeros(len(DC_wave) * len(AC_wave))

        # Replicate the entire array AC_wave at each DC value
        for i, offset in enumerate(DC_wave):
            start_idx = i * len(AC_wave)
            end_idx = start_idx + len(AC_wave)
            result[start_idx:end_idx] = AC_wave + offset

        self.cantilever_excitation_waveform = result

    def build_spectroscopy_waveform(self):
        """
        build_spectroscopy_waveform function that builds the DC wave
        """

        if self.measurement_state == "on and off":
            self.DC_wave = self.insert_constant(
                self.DC_waveform, self.measurement_state_offset
            )
        else:
            self.DC_wave = self.DC_waveform

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

        
    @property
    def cantilever_excitation_length(self):
        return self._cantilever_excitation_length
    
    @property
    def cantilever_excitation_time(self):
        return self.cantilever_excitation_length/self.AO_rate
    
    @property
    def max_voltage(self):
        return np.abs(self.cantilever_excitation_waveform).max() #np.abs(self.DC_wave).max()
