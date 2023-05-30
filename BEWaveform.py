import numpy as np
from scipy.special import erf

class BEWaveform:
    
    def __init__(self, BE_parms, DAQ_platform = "PXI-5412") -> None:
        self.BE_parms = BE_parms
        self.DAQ_platform = DAQ_platform
        
    def determine_AO_rate(self, BE_ppw, function_generator_freq = 0.01):
        # coerce IO rate to acceptable value for PXI-5412
        if self.DAQ_platform == "PXI-5412":
            AO_rate_req = BE_ppw / function_generator_freq  # requested frequency of the function generator
            M, N = np.meshgrid(range(4), range(4))  # matrix of acceptable rates
            av_mat = (5 ** M) * (2 ** N)
            av_vec = av_mat.reshape(16)
            av_vec = np.delete(av_vec, 15)
            av_vec = np.sort(av_vec)
            av_vec = 100E6 / av_vec
            AO_rate = av_vec[np.argmin(abs(av_vec - AO_rate_req))]
            SS_step_t = BE_ppw / AO_rate

        # # coerce IO rate to acceptable value for PXI-6115
        # if DAQ_platform_cond == 1:
        #     AO_rate_req = BE_ppw / function_generator_freq  # requested frequency of the function generator
        #     av_vec = 20E6 / np.arange(1, 1001)  # vector of acceptable rates
        #     AO_rate = av_vec[np.argmin(abs(av_vec - AO_rate_req))]
        #     SS_step_t = BE_ppw / AO_rate

        return AO_rate, SS_step_t
    
def build_BE(self, BE_parm_vec, AO_rate, SS_step_t, chirp_direction = 0):
    # BE_wave_type = BE_parm_vec[0]
    # self.BE_parms['BE_w_center'] = BE_parm_vec[1]
    # self.BE_parms['BE_w_width'] = BE_parm_vec[2]
    # BE_amp = BE_parm_vec[3]
    # BE_smoothing = BE_parm_vec[4]
    # BE_phase_var = BE_parm_vec[5]
    # BE_window_adj = BE_parm_vec[6]
    BE_ppw = 2**self.BE_parms["BE_ppw"]
    BE_rep = 2**self.BE_parms["BE_rep"]

    w1 = self.BE_parms['BE_w_center'] - self.BE_parms['BE_w_width']/2
    w2 = self.BE_parms['BE_w_center'] + self.BE_parms['BE_w_width']/2

    if self.BE_parms['BE_wave_type'] == "chirp":
      chirp_t = SS_step_t / BE_rep
      t = np.arange(0, chirp_t, 1 / (AO_rate - 1))  # time vector
      m = (w2 - w1) / chirp_t / 2  # slope of frequency change
      w_chirp = m * t + w1  # vector for linear frequency change with time
      chirp_smoothing = 4 * chirp_t * self.BE_parms["BE_smoothing"] / 1E4  # smoothing factor for error function
      envelope_a = (1 + erf((t - chirp_smoothing * 2) / chirp_smoothing)) / 2  # first half of erf window
      envelope_b = (1 + erf((t + chirp_smoothing * 2 - chirp_t) / chirp_smoothing)) / 2  # second half of erf window
      envelope = envelope_a - envelope_b  # erf window
      A = envelope * np.sin(2 * np.pi * t * w_chirp)

      for k in range(int(np.log2(BE_rep))):
          A = np.concatenate((A, A))

      if chirp_direction == 0:
          BE_wave = A[::-1]
      if chirp_direction == 1:
          BE_wave = A
      BE_band = np.fft.fftshift(np.fft.fft(A))

    elif self.BE_parms['BE_wave_type'] == "sinc":
        NotImplementedError ("sinc BE wave not implemented yet")
        # N = int(np.log2(BE_ppw))
        # t_max = SS_step_t
        # IO_rate = 2**N / t_max
        # bw = w2 - w1

        # w_vec = np.arange(-IO_rate/2, IO_rate/2 + IO_rate/(2**N-1), IO_rate/(2**N-1))
        # f_resolution = 1 / t_max
        # bin_ind_1 = round(2**(N-1) + w1 / f_resolution)
        # bin_ind_2 = round(2**(N-1) + w2 / f_resolution)
        # bin_ind = np.arange(bin_ind_1, bin_ind_2 + 1)
        
        # points_per_band = bw * t_max
        # f_rate = bw / t_max

        # x1 = np.arange(0, 1, 1 / (len(bin_ind) - 1))

        # Yp_chirp1 = -1 * ((x1) ** 2) * bw * np.pi * t_max * 1
        # Yp1 = -1 * ((x1) ** 2) * bw * np.pi * t_max * BE_phase_var
        # sigma = BE_smoothing
        # a = erf((w_vec - w1 - 2 * sigma) / sigma)
        # b = erf((w_vec - w2 + 2 * sigma) / sigma)
        # Ya = np.sqrt(2 ** N * IO_rate / bw) * 1 / 2 * (a - b)
        # Yp = np.zeros_like(Ya)
        # Yp[bin_ind] = Yp1
        # Yp_chirp = np.zeros_like(Ya)
        # Yp_chirp[bin_ind] = Yp_chirp1

        # Y = Ya * np.exp(1j * Yp)
        # Y_chirp = Ya * np.exp(1j * Yp_chirp)

        # A = np.real(np.fft.ifft(np.fft.fftshift(Y)))
        # A = np.roll(A, round((2**N) * (1 - BE_phase_var) / 2))

        # B = np.real(np.fft.ifft(np.fft.fftshift(Y_chirp)))

        # if 1:
        #     cut_fraction = (BE_rep - 1) / (2 * BE_rep)
        #     keep = slice(int(BE_ppw * cut_fraction) + 1, int(BE_ppw * (1 - cut_fraction)))
        #     A = A[keep]
        #     for k in range(int(np.log2(BE_rep))):
        #         A = np.concatenate((A, A))

        # BE_wave = A / max(A)
        # BE_band = np.fft.fftshift(np.fft.fft(A))

        # BE_wave_chirp = B / max(B)
        # BE_band_chirp = np.fft.fftshift(np.fft.fft(B))


    # Return the results (modify accordingly)
    return BE_wave, BE_band