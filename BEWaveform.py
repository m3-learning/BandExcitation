import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


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

        if ~isinstance(self.BE_rep, int):
            ValueError("BE_rep must be an integer")
        elif self.BE_rep > 1:
            self.BE_wave = np.repeat(self.BE_wave, self.BE_rep)
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

        self.BE_wave = BE_waveform.BE_wave
        self.SS_wave = BE_waveform.SS_wave

    def plot_fft(self, signal=None):
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
        freqs = np.fft.fftfreq(N, self.AO_rate)
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

class Spectroscopy():
    def __init__(
        self,
        BE_ppw,
        n_read = 256,
        SS_steps_per_cycle= 64,
               
    ) -> None:
        
        super().__init__()
        self.BE_ppw = 2**BE_ppw
        self.n_read = n_read
        self.SS_steps_per_cycle = SS_steps_per_cycle
        self.build_SS()

    def build_SS(self):
            
            n_cycle = (2*self.n_read)  * self.SS_steps_per_cycle
            interp_factor = (2*self.BE_ppw )/ self.n_read
            n_step = 2*self.n_read 
            n_write_vec = np.arange(self.n_read, n_cycle // 4, n_step)

            dc_amp_vec_1 = np.arange(0, self.SS_steps_per_cycle/4, 1)
            dc_amp_vec_2 = np.arange(self.SS_steps_per_cycle/4, -1,-1)
            
            y_positive,y_negative = np.zeros(n_cycle // 4 - 1),np.zeros(n_cycle // 4 - 1)
            
            for step_count in range(int(self.SS_steps_per_cycle/4)):
                n_sub_shifted = np.arange(1, n_cycle // 4) - n_write_vec[step_count]
                yk = 0.5 * (erf(n_sub_shifted ) - erf((n_sub_shifted - self.n_read)))
                y_positive += dc_amp_vec_1[step_count] * yk
                y_negative += dc_amp_vec_2[step_count] * yk       
            n = np.arange(n_cycle - 4)
            self.SS_wave  = np.interp(np.arange(1, int(n_cycle * interp_factor) + 1) / interp_factor, n, np.concatenate((y_positive,y_negative, -y_positive, -y_negative)))

            # getting the points where BE will be placed
            n_write_vec = np.concatenate((np.arange(self.n_read, n_cycle // 2, n_step),np.arange(self.n_read, n_cycle // 2 - self.n_read - 1, n_step) + n_cycle // 2))
            n_read_vec = np.concatenate(([1 / interp_factor], n_write_vec + (self.n_read )))
            ni_read_vec = ((self.SS_steps_per_cycle*100)+n_read_vec * interp_factor)  
            ni_write_vec = ((self.SS_steps_per_cycle*100)+n_write_vec * interp_factor) 

            self.SS_read_vec = np.round(ni_read_vec + np.round(interp_factor / 2))
            SS_write_vec = np.round(ni_write_vec + np.round( interp_factor / 2))
            self.SS_write_vec = np.concatenate((SS_write_vec, [SS_write_vec[-1] + np.round(n_step * interp_factor)]))
            
class BE_Spectroscopy(BEWaveform,Spectroscopy):

    def __init__(
        self,
        BE_ppw,
        BE_rep,
        center_freq=500e3,
        bandwidth=60e3,
        wave="chirp",
        spectroscopic_waveform = "Bipolar",
        platform="PXI-5412",
        waveform_time=4e-3,
        BE_smoothing=125,
        BE_phase_var=None,
        chirp_direction="up",
        
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
        self.spectroscopic_waveform = spectroscopic_waveform
        Spectroscopy.__init__(self, BE_ppw)
        self.build_spectroscopy_waveform = self.build_spectroscopy_waveform()
        
    def build_spectroscopy_waveform(self):
        if self.spectroscopic_waveform == "Bipolar":
            # merging BE and SS waveforms
            n_step = len(self.BE_wave)
            self.SS_wave[:n_step] += self.BE_wave
            for step_count in range(len(self.SS_read_vec)):
                start_idx = int(self.SS_read_vec[step_count]) 
                end_idx = start_idx + n_step - 1
                self.SS_wave[start_idx:end_idx] += self.BE_wave[: n_step - 1]
            
            for step_count in range(len(self.SS_write_vec)):
                start_idx = int(self.SS_write_vec[step_count]) 
                end_idx = start_idx + n_step - 1
                self.SS_wave[start_idx:end_idx] += self.BE_wave[: n_step - 1]
        else:
            raise ValueError("Invalid spectroscopic waveform type")
        return(self.SS_wave)


# class BEWaveform:
#     def __init__(
#         self,
#         BE_parms_1,
#         BE_parms_2,
#         SS_parm_vec,
#         assembly_parm_vec,
#         DAQ_platform="PXI-5412",
#     ) -> None:
#         self.BE_parms_1 = BE_parms_1
#         self.BE_parms_2 = BE_parms_2
#         self.SS_parm_vec = SS_parm_vec
#         self.assembly_parm_vec = assembly_parm_vec
#         self.DAQ_platform = DAQ_platform

#     # def determine_AO_rate(self, BE_ppw, function_generator_freq=0.01):
#     #     # coerce IO rate to acceptable value for PXI-5412
#     #     if self.DAQ_platform == "PXI-5412":
#     #         AO_rate_req = (
#     #             BE_ppw / function_generator_freq
#     #         )  # requested frequency of the function generator
#     #         M, N = np.meshgrid(range(4), range(4))  # matrix of acceptable rates
#     #         av_mat = (5**M) * (2**N)
#     #         av_vec = av_mat.reshape(16)
#     #         av_vec = np.delete(av_vec, 15)
#     #         av_vec = np.sort(av_vec)
#     #         av_vec = 100e6 / av_vec
#     #         AO_rate = av_vec[np.argmin(abs(av_vec - AO_rate_req))]
#     #         SS_step_t = BE_ppw / AO_rate

#     #     return AO_rate, SS_step_t

#     def build_BE(self, BE_parms, chirp_direction=0, **kwargs):
#         # BE_ppw = 2 ** BE_parms["BE_ppw"]
#         # BE_rep = 2 ** BE_parms["BE_rep"]
#         AO_rate, SS_step_t = self.determine_AO_rate(BE_ppw)

#         w1 = BE_parms["BE_w_center"] - BE_parms["BE_w_width"] / 2
#         w2 = BE_parms["BE_w_center"] + BE_parms["BE_w_width"] / 2

#         if BE_parms["BE_wave_type"] == "chirp":
#             chirp_t = SS_step_t / BE_rep
#             t = np.arange(0, chirp_t, 1 / (AO_rate - 1))  # time vector
#             m = (w2 - w1) / chirp_t / 2  # slope of frequency change
#             w_chirp = m * t + w1  # vector for linear frequency change with time
#             chirp_smoothing = (
#                 4 * chirp_t * BE_parms["BE_smoothing"] / 1e4
#             )  # smoothing factor for error function
#             envelope_a = (
#                 1 + erf((t - chirp_smoothing * 2) / chirp_smoothing)
#             ) / 2  # first half of erf window
#             envelope_b = (
#                 1 + erf((t + chirp_smoothing * 2 - chirp_t) / chirp_smoothing)
#             ) / 2  # second half of erf window
#             envelope = envelope_a - envelope_b  # erf window
#             A = envelope * np.sin(2 * np.pi * t * w_chirp)

#             for k in range(int(np.log2(BE_rep))):
#                 A = np.concatenate((A, A))

#             if chirp_direction == 0:
#                 BE_wave = A[::-1]
#             if chirp_direction == 1:
#                 BE_wave = A
#             BE_band = np.fft.fftshift(np.fft.fft(A))

#         elif BE_parms["BE_wave_type"] == "sinc":
#             N = int(np.log2(BE_ppw))
#             t_max = SS_step_t
#             IO_rate = 2**N / t_max
#             bw = w2 - w1

#             w_vec = np.arange(
#                 -IO_rate / 2,
#                 IO_rate / 2 + IO_rate / (2**N - 1),
#                 IO_rate / (2**N - 1),
#             )
#             f_resolution = 1 / t_max
#             bin_ind_1 = round(2 ** (N - 1) + w1 / f_resolution)
#             bin_ind_2 = round(2 ** (N - 1) + w2 / f_resolution)
#             bin_ind = np.arange(bin_ind_1, bin_ind_2 + 1)

#             x1 = np.arange(0, 1, 1 / (len(bin_ind) - 1))

#             Yp_chirp1 = -1 * ((x1) ** 2) * bw * np.pi * t_max * 1
#             Yp1 = -1 * ((x1) ** 2) * bw * np.pi * t_max * self.BE_phase_var
#             sigma = BE_parms["BE_smoothing"]
#             a = erf((w_vec - w1 - 2 * sigma) / sigma)
#             b = erf((w_vec - w2 + 2 * sigma) / sigma)
#             Ya = np.sqrt(2**N * IO_rate / bw) * 1 / 2 * (a - b)
#             Yp = np.zeros_like(Ya)
#             Yp[bin_ind[: len(Yp1)]] = Yp1
#             Yp_chirp = np.zeros_like(Ya)
#             Yp_chirp[bin_ind[: len(Yp1)]] = Yp_chirp1

#             Y = Ya * np.exp(1j * Yp)
#             Y_chirp = Ya * np.exp(1j * Yp_chirp)

#             A = np.real(np.fft.ifft(np.fft.fftshift(Y)))
#             A = np.roll(A, round((2**N) * (1 - self.BE_phase_var) / 2))

#             B = np.real(np.fft.ifft(np.fft.fftshift(Y_chirp)))

#             if 1:
#                 cut_fraction = (BE_rep - 1) / (2 * BE_rep)
#                 keep = slice(
#                     int(BE_ppw * cut_fraction), int(BE_ppw * (1 - cut_fraction))
#                 )
#                 A = A[keep]
#                 for k in range(int(np.log2(BE_rep))):
#                     A = np.concatenate((A, A))

#             BE_wave = A / max(A)
#             BE_band = np.fft.fftshift(np.fft.fft(A))

#         # Return the results (modify accordingly)
#         return BE_wave, BE_band

#     # Plotting BE_wave
#     def plot_BE_wave(fig_num, BE_wave, BE_band, w_vec_full, SS_step_t):
#         fh = plt.figure(fig_num)

#         sph1a = plt.subplot(3, 2, 1)
#         sph1a.tick_params(axis="both", which="both", labelsize=7)
#         plt.plot(np.arange(0, SS_step_t, SS_step_t / (len(BE_wave))), BE_wave)

#         sph1b = plt.subplot(3, 2, 2)
#         sph1b.tick_params(axis="both", which="both", labelsize=7)
#         plt.plot(w_vec_full, np.abs(BE_band))

#     # Plotting SS_wave
#     def plot_SS_wave(fig_num, SS_wave, AO_rate, SS_read_vec, SS_write_vec):
#         fh = plt.figure(fig_num)
#         t_vec = np.arange(len(SS_wave)) / AO_rate

#         plt.plot(t_vec, SS_wave)  # plotting the SS_wave lines

#         ph1 = plt.plot(
#             t_vec[np.array(SS_read_vec, dtype=int)],
#             SS_wave[np.array(SS_read_vec, dtype=int)],
#             "ro",
#         )
#         plt.setp(ph1, markersize=1.5, markerfacecolor=[1, 0, 0])

#         ph2 = plt.plot(
#             t_vec[np.array(SS_write_vec, dtype=int)],
#             SS_wave[np.array(SS_write_vec, dtype=int)],
#             "go",
#         )
#         plt.setp(ph2, markersize=1.5, markerfacecolor=[0, 1, 0])

#         fh.set_facecolor([1, 1, 1])
#         plt.title("SS_wave")

#     # Plotting merge BE and SS wave
#     def plot_BEPS_wave(fig_num, BEPS_wave, AO_rate, SS_read_vec, SS_write_vec):
#         fh = plt.figure(fig_num)
#         t_vec = np.arange(len(BEPS_wave)) / AO_rate

#         sph1 = plt.subplot(2, 2, 1)
#         plt.plot(t_vec, BEPS_wave)
#         plt.xlim([0.4, 0.5])

#         ph1 = plt.plot(
#             t_vec[np.array(SS_read_vec, dtype=int)],
#             BEPS_wave[np.array(SS_read_vec, dtype=int)],
#             "ro",
#         )
#         ph2 = plt.plot(
#             t_vec[np.array(SS_write_vec, dtype=int)],
#             BEPS_wave[np.array(SS_write_vec, dtype=int)],
#             "go",
#         )

#         sph1.tick_params(axis="both", which="both", labelsize=7)

#         for line in ph1:
#             line.set_markersize(1.5)
#             line.set_markerfacecolor("r")
#         for line in ph2:
#             line.set_markersize(1.5)
#             line.set_markerfacecolor("g")

#         sph2 = plt.subplot(2, 2, 2)
#         plt.plot(t_vec, BEPS_wave)

#         ph1 = plt.plot(
#             t_vec[SS_read_vec.astype(int)], BEPS_wave[SS_read_vec.astype(int)], "ro"
#         )
#         ph2 = plt.plot(
#             t_vec[SS_write_vec.astype(int)], BEPS_wave[SS_write_vec.astype(int)], "go"
#         )

#         sph2.tick_params(axis="both", which="both", labelsize=7)

#         for line in ph1:
#             line.set_markersize(1.5)
#             line.set_markerfacecolor("r")
#         for line in ph2:
#             line.set_markersize(1.5)
#             line.set_markerfacecolor("g")
#         max_t = max(t_vec)
#         if not np.isfinite(max_t):
#             max_t = 1.0  # Set a default value for NaN or Inf

#         plt.axis([0, max_t / 20, -1.0, 1.0])  # Adj
#         plt.setp(fh, "facecolor", [1, 1, 1])
#         plt.show()

#     def build_SS(self, **kwargs):
#         BE_ppw = 2 ** self.BE_parms_1["BE_ppw"]

#         self.n_read_final = BE_ppw  # points per read step actual
#         if self.assembly_parm_vec["num_band_ring"] == 1:  # excite two bands
#             if self.assembly_parm_vec["par_ser_ring"] == 1:  # excite them in series
#                 self.n_read_final = 2 * BE_ppw  # then double the width
#         AO_rate, SS_step_t = self.determine_AO_rate(BE_ppw)
#         # AO_length = AO_rate * SS_step_t

#         self.n_read = 256  # points per read reduced in order to speed up calculation

#         SS_smooth = AO_rate * self.SS_parm_vec["SS_smoothing"]  # smoothing factor
#         n_trans = round(SS_smooth * 5)

#         n_pfm = self.n_read_final  # AO_rate*SS_step_t*2;
#         n_setpulse = AO_rate * self.SS_parm_vec["SS_set_pulse_t"]

#         if self.assembly_parm_vec["meas_high_ring"] == 0:
#             n_write = int(self.n_read / self.SS_parm_vec["SS_RW_ratio"])  # points per write
#         if self.assembly_parm_vec["meas_high_ring"] == 1:
#             n_write = self.n_read

#         if self.SS_parm_vec["SS_mode_ring"] == "standard_spectrum":
#             n_write += n_trans
#             n_cycle = (self.n_read + n_trans + n_write) * self.SS_parm_vec[
#                 "self.SS_steps_per_cycle"
#             ]  # points per cycle
#             interp_factor = self.n_read_final / self.n_read

#             n_step = n_write + n_trans + self.n_read  # data points per steps
#             n_write_vec = np.arange(
#                 self.n_read, n_cycle // 4, n_step
#             )  # vector indices when writing starts
#             self.n_read_vec = n_write_vec + n_write  # vector of indices when writing stops

#             dc_amp_vec_1 = np.arange(
#                 self.SS_parm_vec["SS_max_offset_amp"]
#                 / (self.SS_parm_vec["self.SS_steps_per_cycle"] / 4),
#                 self.SS_parm_vec["SS_max_offset_amp"] + 1e-10,
#                 self.SS_parm_vec["SS_max_offset_amp"]
#                 / (self.SS_parm_vec["self.SS_steps_per_cycle"] / 4),
#             )  # vector of offset values for first quarter wave
#             dc_amp_vec_2 = np.arange(
#                 self.SS_parm_vec["SS_max_offset_amp"]
#                 - self.SS_parm_vec["SS_max_offset_amp"]
#                 / (self.SS_parm_vec["self.SS_steps_per_cycle"] / 4),
#                 -self.SS_parm_vec["SS_max_offset_amp"]
#                 / (self.SS_parm_vec["self.SS_steps_per_cycle"] / 4),
#                 -self.SS_parm_vec["SS_max_offset_amp"]
#                 / (self.SS_parm_vec["self.SS_steps_per_cycle"] / 4),
#             )  # vector of offset values for second quarter wave
#             dc_amp_vec_3 = -dc_amp_vec_1
#             dc_amp_vec_4 = -dc_amp_vec_2
#             dc_amp_vec_1 = dc_amp_vec_1 - self.SS_parm_vec["SS_read_voltage"]
#             dc_amp_vec_2 = dc_amp_vec_2 - self.SS_parm_vec["SS_read_voltage"]
#             dc_amp_vec_3 = dc_amp_vec_3 - self.SS_parm_vec["SS_read_voltage"]
#             dc_amp_vec_4 = dc_amp_vec_4 - self.SS_parm_vec["SS_read_voltage"]

#             plt.figure(55)
#             plt.plot(dc_amp_vec_1, "b.-")
#             plt.plot(dc_amp_vec_2, "r.-")
#             plt.plot(dc_amp_vec_3, "k.-")
#             plt.plot(dc_amp_vec_4, "g.-")

#             # build quarter waves
#             n_sub = np.arange(1, n_cycle // 4)
#             y1 = np.zeros_like(n_sub)
#             y2 = np.zeros_like(n_sub)
#             y3 = np.zeros_like(n_sub)
#             y4 = np.zeros_like(n_sub)

#             for step_count in range((self.SS_parm_vec["self.SS_steps_per_cycle"] // 4)):
#                 yk1 = (
#                     dc_amp_vec_1[step_count]
#                     * 0.5
#                     * (
#                         erf((n_sub - n_write_vec[step_count]) / SS_smooth)
#                         - erf((n_sub - self.n_read_vec[step_count]) / SS_smooth)
#                     )
#                 )

#                 yk2 = (
#                     dc_amp_vec_2[step_count]
#                     * 0.5
#                     * (
#                         erf((n_sub - n_write_vec[step_count]) / SS_smooth)
#                         - erf((n_sub - self.n_read_vec[step_count]) / SS_smooth)
#                     )
#                 )

#                 yk3 = (
#                     dc_amp_vec_3[step_count]
#                     * 0.5
#                     * (
#                         erf((n_sub - n_write_vec[step_count]) / SS_smooth)
#                         - erf((n_sub - self.n_read_vec[step_count]) / SS_smooth)
#                     )
#                 )
#                 yk4 = (
#                     dc_amp_vec_4[step_count]
#                     * 0.5
#                     * (
#                         erf((n_sub - n_write_vec[step_count]) / SS_smooth)
#                         - erf((n_sub - self.n_read_vec[step_count]) / SS_smooth)
#                     )
#                 )

#                 y1 = y1 + yk1
#                 y2 = y2 + yk2
#                 y3 = y3 + yk3
#                 y4 = y4 + yk4

#             # combine quarter waves to build full cycle
#             n = np.arange(n_cycle - 4)  # fix
#             y = np.concatenate((y1, y2, y3, y4))
#             dc_amp_vec_single = np.concatenate(
#                 (dc_amp_vec_1, dc_amp_vec_2, dc_amp_vec_3, dc_amp_vec_4)
#             )
#             # interpolate wave and read/write indices to achieve desired number of points per read step
#             ni = np.arange(1, int(n_cycle * interp_factor) + 1) / interp_factor
#             yi = np.interp(ni, n, y)  # offset output wave

#             ni *= interp_factor
#             n_write_vec = np.concatenate(
#                 (
#                     np.arange(self.n_read, n_cycle // 2, n_step),
#                     n_cycle // 2
#                     + np.arange(self.n_read, n_cycle // 2 - self.n_read - 1, -n_step),
#                 )
#             )
#             self.n_read_vec = np.concatenate(([1 / interp_factor], n_write_vec + n_write))
#             ni_read_vec = self.n_read_vec * interp_factor  # vector of indices for reading
#             ni_write_vec = n_write_vec * interp_factor  # vector of indices for writing
#             ni_read_vec = ni_read_vec[:-1]
#             ni_write_vec = ni_write_vec[:-1]

#             # repeat full cycle
#             yi0 = yi.copy()
#             ni_write_vec0 = ni_write_vec.copy()
#             ni_read_vec0 = ni_read_vec.copy()
#             dc_amp_vec_full = dc_amp_vec_single.copy()

#             for k in range(self.SS_parm_vec["SS_num_loops"] - 1):
#                 yi = np.concatenate((yi, yi0))
#                 ly = len(yi) - len(yi0)
#                 final_read = ni_read_vec[-1]
#                 ni_write_vec = np.concatenate(
#                     (
#                         ni_write_vec,
#                         [final_read + self.n_read * interp_factor],
#                         ni_write_vec0 + ly,
#                     )
#                 )
#                 ni_read_vec = np.concatenate((ni_read_vec, ni_read_vec0 + ly))
#                 dc_amp_vec_full = np.concatenate((dc_amp_vec_full, dc_amp_vec_single))

#             n_sp = np.arange(1, n_setpulse + 2 * n_trans + 1)
#             y_sp = (
#                 0.5
#                 * self.SS_parm_vec["SS_set_pulse_amp"]
#                 * (
#                     erf((n_sp - n_trans * interp_factor) / (SS_smooth * interp_factor))
#                     - erf(
#                         (n_sp - n_setpulse + n_trans * interp_factor)
#                         / (SS_smooth * interp_factor)
#                     )
#                 )
#             )

#             # Add PFM read and setpulse
#             ni = np.arange(
#                 1, len(ni) * self.SS_parm_vec["SS_num_loops"] + n_pfm + n_setpulse + 1
#             )
#             yi = np.concatenate((np.zeros(n_pfm), y_sp, yi))
#             ni_read_vec = ni_read_vec + n_pfm + n_setpulse
#             ni_write_vec = ni_write_vec + n_pfm + n_setpulse
#             SS_read_vec = np.round(ni_read_vec + np.round(n_trans * interp_factor / 2))
#             SS_write_vec = np.round(
#                 ni_write_vec + np.round(n_trans * interp_factor / 2)
#             )
#             SS_write_vec = np.concatenate(
#                 (SS_write_vec, [SS_write_vec[-1] + np.round(n_step * interp_factor)])
#             )
#             SS_read_vec[0] = np.round(ni_read_vec[0]) - interp_factor * n_trans / 2
#             SS_wave = yi + self.SS_parm_vec["SS_read_voltage"]
#             dc_amp_vec_full = dc_amp_vec_full + self.SS_parm_vec["SS_read_voltage"]
#             SS_wave_nan = np.where(np.isnan(SS_wave))[0]
#             SS_wave[SS_wave_nan] = 0
#             SS_parm_out = np.arange(2, 12)

#         return SS_wave, SS_read_vec, SS_write_vec, SS_parm_out

#     def merge_BE_SS(self, BE_wave_1, BE_wave_2, SS_wave, SS_read_vec, SS_write_vec):
#         if self.assembly_parm_vec["num_band_ring"] == 0:  # excite one band
#             BE_wave = BE_wave_1

#         if self.assembly_parm_vec["num_band_ring"] == 1:  # excite two bands
#             if self.assembly_parm_vec["par_ser_ring"] == 0:  # parallel combination
#                 BE_wave = BE_wave_1 + BE_wave_2
#             if self.assembly_parm_vec["par_ser_ring"] == 1:  # parallel combination
#                 BE_wave = np.concatenate((BE_wave_1, BE_wave_2))

#         n_step = len(BE_wave)
#         if self.SS_parm_vec["SS_mode_ring"] == "standard_spectrum":
#             BEPS_wave = SS_wave
#             BEPS_wave_dc = SS_wave
#             BEPS_wave_ac = np.zeros_like(SS_wave)
#             BEPS_wave[:n_step] += BE_wave

#             for step_count in range(len(SS_read_vec)):
#                 start_idx = int(SS_read_vec[step_count])
#                 end_idx = int(SS_read_vec[step_count]) + n_step - 1
#                 BEPS_wave[start_idx:end_idx] += BE_wave[: n_step - 1]
#                 BEPS_wave_ac[start_idx:end_idx] = BE_wave[: n_step - 1]

#             if self.assembly_parm_vec["meas_high_ring"] == 1:
#                 for step_count in range(len(SS_write_vec)):
#                     start_idx = int(SS_write_vec[step_count])
#                     end_idx = int(SS_write_vec[step_count]) + n_step - 1
#                     BEPS_wave[start_idx:end_idx] += BE_wave[: n_step - 1]
#                     BEPS_wave_ac[start_idx:end_idx] = BE_wave[: n_step - 1]
#         ly = len(BEPS_wave)
#         if ly % 4 == 1:
#             BEPS_wave = BEPS_wave[:-1]
#         elif ly % 4 == 2:
#             BEPS_wave = BEPS_wave[:-2]
#         elif ly % 4 == 3:
#             BEPS_wave = BEPS_wave[:-3]

#         return BEPS_wave, BEPS_wave_ac, BEPS_wave_dc

#     def BEPS_wave_build(self, plot_cond_vec=1, num_band_ring=1, **kwargs):
#         BE_ppw = 2 ** self.BE_parms_1["BE_ppw"]
#         BE_rep = 2 ** self.BE_parms_1["BE_rep"]

#         # Determine IO rate
#         AO_rate, SS_step_t = self.determine_AO_rate(BE_ppw)
#         AO_length = AO_rate * SS_step_t

#         w_vec_full = np.arange(
#             -AO_rate / 2,
#             AO_rate / 2 + AO_rate / (AO_length - 1),
#             AO_rate / (AO_length - 1),
#         )

#         BE_wave_1, BE_band_1 = BEWaveform.build_BE(self, self.BE_parms_1)

#         BE_wave_1 = BE_wave_1 * self.BE_parms_1["BE_amp"]
#         F_BE_wave_1 = np.fft.fftshift(np.fft.fft(BE_wave_1))
#         F_BE_wave_1 = F_BE_wave_1[len(F_BE_wave_1) // 2 :]
#         F2_BE_wave_1 = np.fft.fftshift(np.fft.fft(BE_wave_1**2))
#         F2_BE_wave_1 = F2_BE_wave_1[len(F2_BE_wave_1) // 2 :]
#         if plot_cond_vec == 1:
#             BEWaveform.plot_BE_wave(1, BE_wave_1, BE_band_1, w_vec_full, SS_step_t)

#         if num_band_ring == 1:
#             BE_wave_2, BE_band_2 = BEWaveform.build_BE(self, self.BE_parms_2)
#             BE_wave_2 = BE_wave_2 * self.BE_parms_2["BE_amp"]
#             F_BE_wave_2 = np.fft.fftshift(np.fft.fft(BE_wave_2))
#             F_BE_wave_2 = F_BE_wave_2[: len(F_BE_wave_2) // 2]
#             if plot_cond_vec == 1:
#                 BEWaveform.plot_BE_wave(2, BE_wave_2, BE_band_2, w_vec_full, SS_step_t)

#         # Build SS waveform
#         SS_wave, SS_read_vec, SS_write_vec, SS_parm_out = BEWaveform.build_SS(self)
#         # if 0:
#         if plot_cond_vec == 1:
#             BEWaveform.plot_SS_wave(3, SS_wave, AO_rate, SS_read_vec, SS_write_vec)
#         BEPS_wave, BEPS_wave_ac, BEPS_wave_dc = BEWaveform.merge_BE_SS(
#             self, BE_wave_1, BE_wave_2, SS_wave, SS_read_vec, SS_write_vec
#         )
#         # BEPS_wave = single(BEPS_wave);
#         if plot_cond_vec == 1:
#             BEWaveform.plot_BEPS_wave(4, BEPS_wave, AO_rate, SS_read_vec, SS_write_vec)
#         return BEPS_wave, SS_parm_out, SS_read_vec, SS_write_vec
