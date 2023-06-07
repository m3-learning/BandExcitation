import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

class BEWaveform:
    
    def __init__(self, BE_parms_1,BE_parms_2, SS_parm_vec, assembly_parm_vec, DAQ_platform = "PXI-5412") -> None:
        self.BE_parms_1 = BE_parms_1
        self.BE_parms_2 = BE_parms_2
        self.SS_parm_vec = SS_parm_vec
        self.assembly_parm_vec = assembly_parm_vec
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
    
    def build_BE(self, BE_parms,chirp_direction = 0, **kwargs):
        BE_ppw = 2**BE_parms["BE_ppw"]
        BE_rep = 2**BE_parms["BE_rep"]
        AO_rate, SS_step_t = self.determine_AO_rate(BE_ppw)

        w1 = BE_parms['BE_w_center'] - BE_parms['BE_w_width']/2
        w2 = BE_parms['BE_w_center'] + BE_parms['BE_w_width']/2

        if BE_parms['BE_wave_type'] == "chirp":
            
            chirp_t = SS_step_t / BE_rep
            t = np.arange(0, chirp_t, 1 / (AO_rate - 1))  # time vector
            m = (w2 - w1) / chirp_t / 2  # slope of frequency change
            w_chirp = m * t + w1  # vector for linear frequency change with time
            chirp_smoothing = 4 * chirp_t * BE_parms["BE_smoothing"] / 1E4  # smoothing factor for error function
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

        elif BE_parms['BE_wave_type'] == "sinc":
            BE_wave = None
            BE_band = None
            #NotImplementedError ()
            N = int(np.log2(BE_ppw))
            t_max = SS_step_t
            IO_rate = 2**N / t_max
            bw = w2 - w1

            w_vec = np.arange(-IO_rate/2, IO_rate/2 + IO_rate/(2**N-1), IO_rate/(2**N-1))
            f_resolution = 1 / t_max
            bin_ind_1 = round(2**(N-1) + w1 / f_resolution)
            bin_ind_2 = round(2**(N-1) + w2 / f_resolution)
            bin_ind = np.arange(bin_ind_1, bin_ind_2 + 1)
           
            points_per_band = bw * t_max
            f_rate = bw / t_max

            x1 = np.arange(0, 1, 1 / (len(bin_ind) - 1))

            Yp_chirp1 = -1 * ((x1) ** 2) * bw * np.pi * t_max * 1
            Yp1 = -1 * ((x1) ** 2) * bw * np.pi * t_max * BE_parms["BE_phase_var"]
            sigma = BE_parms["BE_smoothing"]
            a = erf((w_vec - w1 - 2 * sigma) / sigma)
            b = erf((w_vec - w2 + 2 * sigma) / sigma)
            Ya = np.sqrt(2 ** N * IO_rate / bw) * 1 / 2 * (a - b)
            Yp = np.zeros_like(Ya)
            Yp[bin_ind[:len(Yp1)]] = Yp1
            Yp_chirp = np.zeros_like(Ya)
            Yp_chirp[bin_ind[:len(Yp1)]] = Yp_chirp1

            Y = Ya * np.exp(1j * Yp)
            Y_chirp = Ya * np.exp(1j * Yp_chirp)

            A = np.real(np.fft.ifft(np.fft.fftshift(Y)))
            A = np.roll(A, round((2**N) * (1 -BE_parms["BE_phase_var"]) / 2))

            B = np.real(np.fft.ifft(np.fft.fftshift(Y_chirp)))

            if 1:
                cut_fraction = (BE_rep - 1) / (2 * BE_rep)
                keep = slice(int(BE_ppw * cut_fraction), int(BE_ppw * (1 - cut_fraction)))
                A = A[keep]
                for k in range(int(np.log2(BE_rep))):
                    A = np.concatenate((A, A))

            BE_wave = A / max(A)
            BE_band = np.fft.fftshift(np.fft.fft(A))

            BE_wave_chirp = B / max(B)
            BE_band_chirp = np.fft.fftshift(np.fft.fft(B))
            
        # Return the results (modify accordingly)
        return BE_wave, BE_band


    def plot_BE_wave(fig_num,BE_wave, BE_band, w_ind_band, w_vec_full, SS_step_t):
        fh = plt.figure(fig_num)

        sph1a = plt.subplot(3, 2, 1)
        sph1a.tick_params(axis='both', which='both', labelsize=7)
        plt.plot(np.arange(0, SS_step_t, SS_step_t / (len(BE_wave) )), BE_wave)
                
        sph1b = plt.subplot(3, 2, 2)
        sph1b.tick_params(axis='both', which='both', labelsize=7)
        plt.plot(w_vec_full, np.abs(BE_band))

    def plot_SS_wave(fig_num, SS_wave, AO_rate, SS_read_vec, SS_write_vec):
        fh = plt.figure(fig_num)
        t_vec = np.arange(len(SS_wave)) / AO_rate
       
        plt.plot(t_vec, SS_wave)

        ph1 = plt.plot(t_vec[SS_read_vec], SS_wave[SS_read_vec], 'ro')
        ph2 = plt.plot(t_vec[SS_write_vec], SS_wave[SS_write_vec], 'go')
        plt.setp(ph1, markersize=1.5, markerfacecolor=[1, 0, 0])
        plt.setp(ph2, markersize=1.5, markerfacecolor=[0, 1, 0])
        fh.set_facecolor([1, 1, 1])
        
        fh.set_facecolor([1, 1, 1])

    def build_SS(self, chirp_direction = 0, **kwargs):
        BE_ppw = 2**self.BE_parms_1["BE_ppw"]

        n_read_final = BE_ppw  # points per read step actual
        if self.assembly_parm_vec["num_band_ring"] == 1:  # excite two bands
            if self.assembly_parm_vec["par_ser_ring"] == 1:  # excite them in series
                n_read_final = 2 * BE_ppw  # then double the width
        AO_rate, SS_step_t = self.determine_AO_rate(BE_ppw)
        AO_length = AO_rate * SS_step_t

        n_read = 256  # points per read reduced in order to speed up calculation

        SS_smooth = AO_rate * self.SS_parm_vec["SS_smoothing"]  # smoothing factor
        n_trans = round(SS_smooth * 5)
        # read_delay = 3; %ensures that the reading starts(stops) after(before) smoothing

        n_pfm = n_read_final  # AO_rate*SS_step_t*2;
        n_setpulse = AO_rate * self.SS_parm_vec["SS_set_pulse_t"]

        if self.assembly_parm_vec["meas_high_ring"] == 0:
            n_write = int(n_read / self.SS_parm_vec["SS_RW_ratio"])  # points per write
        if self.assembly_parm_vec["meas_high_ring"] == 1:
            n_write = n_read
        

        if self.SS_parm_vec["SS_mode_ring"] == 0:

            n_write = n_write + n_trans
            n_cycle = (n_read + n_trans + n_write) * self.SS_parm_vec["SS_steps_per_cycle"]  # points per cycle
            interp_factor = n_read_final / n_read

            n_step = n_write + n_trans + n_read  # data points per steps
            n_write_vec = np.arange(n_read, n_cycle // 4, n_step)  # vector indices when writing starts
            n_read_vec = n_write_vec + n_write  # vector of indices when writing stops

            dc_amp_vec_1 = np.arange(self.SS_parm_vec["SS_max_offset_amp"] / ( self.SS_parm_vec["SS_steps_per_cycle"] / 4),self.SS_parm_vec["SS_max_offset_amp"] + 1e-10,
                                    self.SS_parm_vec["SS_max_offset_amp"] / ( self.SS_parm_vec["SS_steps_per_cycle"] / 4))  # vector of offset values for first quarter wave
            dc_amp_vec_2 = np.arange( self.SS_parm_vec["SS_max_offset_amp"]  - self.SS_parm_vec["SS_max_offset_amp"] /( self.SS_parm_vec["SS_steps_per_cycle"]/4), 
                                     - self.SS_parm_vec["SS_max_offset_amp"] /( self.SS_parm_vec["SS_steps_per_cycle"]/4), - self.SS_parm_vec["SS_max_offset_amp"] /( self.SS_parm_vec["SS_steps_per_cycle"]/4)) # vector of offset values for second quarter wave
            dc_amp_vec_3 = -dc_amp_vec_1
            dc_amp_vec_4 = -dc_amp_vec_2
            dc_amp_vec_1 = dc_amp_vec_1 - self.SS_parm_vec["SS_read_voltage"]
            dc_amp_vec_2 = dc_amp_vec_2 - self.SS_parm_vec["SS_read_voltage"]
            dc_amp_vec_3 = dc_amp_vec_3 - self.SS_parm_vec["SS_read_voltage"]
            dc_amp_vec_4 = dc_amp_vec_4 - self.SS_parm_vec["SS_read_voltage"]

            plt.figure(55)
            plt.clf()
            plt.plot(dc_amp_vec_1, 'b.-')
            plt.plot(dc_amp_vec_2, 'r.-')
            plt.plot(dc_amp_vec_3, 'k.-')
            plt.plot(dc_amp_vec_4, 'g.-')

            # build quarter waves
            n_sub = np.arange(1, n_cycle // 4 )
            y1 = np.zeros_like(n_sub)
            y2 = np.zeros_like(n_sub)
            y3 = np.zeros_like(n_sub)
            y4 = np.zeros_like(n_sub)
        
            for step_count in range((self.SS_parm_vec["SS_steps_per_cycle"]//4)):
                yk1 = dc_amp_vec_1[step_count] * 0.5 * (
                    erf((n_sub - n_write_vec[step_count]) / SS_smooth) - erf((n_sub - n_read_vec[step_count]) / SS_smooth))
                
                yk2 = dc_amp_vec_2[step_count] * 0.5 * (
                    erf((n_sub - n_write_vec[step_count]) / SS_smooth) - erf((n_sub - n_read_vec[step_count]) / SS_smooth))
                
                yk3 = dc_amp_vec_3[step_count] * 0.5 * (
                    erf((n_sub - n_write_vec[step_count]) / SS_smooth) - erf((n_sub - n_read_vec[step_count]) / SS_smooth))
                yk4 = dc_amp_vec_4[step_count] * 0.5 * (
                    erf((n_sub - n_write_vec[step_count]) / SS_smooth) - erf((n_sub - n_read_vec[step_count]) / SS_smooth))

                y1 = y1 + yk1
                y2 = y2 + yk2
                y3 = y3 + yk3
                y4 = y4 + yk4

            # combine quarter waves to build full cycle
            n = np.arange(n_cycle-4 ) #fix
            y = np.concatenate((y1, y2, y3, y4))
            dc_amp_vec_single = np.concatenate(
                (dc_amp_vec_1, dc_amp_vec_2, dc_amp_vec_3, dc_amp_vec_4))
            # interpolate wave and read/write indices to achieve desired number of points per read step
            ni = np.arange(1, int(n_cycle * interp_factor) + 1) / interp_factor
            yi = np.interp(ni, n, y)  # offset output wave

            ni *= interp_factor
            n_write_vec = np.concatenate(
                (np.arange(n_read, n_cycle // 2, n_step),
                n_cycle // 2 + np.arange(n_read, n_cycle // 2 - n_read - 1, -n_step)))
            n_read_vec = np.concatenate(([1 / interp_factor], n_write_vec + n_write))
            ni_read_vec = n_read_vec * interp_factor  # vector of indices for reading
            ni_write_vec = n_write_vec * interp_factor  # vector of indices for writing
            ni_read_vec = ni_read_vec[:-1]
            ni_write_vec = ni_write_vec[:-1]

            # repeat full cycle
            yi0 = yi.copy()
            ni_write_vec0 = ni_write_vec.copy()
            ni_read_vec0 = ni_read_vec.copy()
            dc_amp_vec_full = dc_amp_vec_single.copy()

            for k in range(self.SS_parm_vec["SS_num_loops"] - 1):
                yi = np.concatenate((yi, yi0))
                ly = len(yi) - len(yi0)
                final_read = ni_read_vec[-1]
                ni_write_vec = np.concatenate(
                    (ni_write_vec, [final_read + n_read * interp_factor], ni_write_vec0 + ly))
                ni_read_vec = np.concatenate((ni_read_vec, ni_read_vec0 + ly))
                dc_amp_vec_full = np.concatenate((dc_amp_vec_full, dc_amp_vec_single))
            
            n_sp = np.arange(1, n_setpulse + 2 * n_trans + 1)
            y_sp = 0.5 * self.SS_parm_vec["SS_set_pulse_amp"] * (erf((n_sp - n_trans * interp_factor) / (SS_smooth * interp_factor))
                                            - erf((n_sp - n_setpulse + n_trans * interp_factor) / (SS_smooth * interp_factor)))

            # Add PFM read and setpulse
            ni = np.arange(1, len(ni) * self.SS_parm_vec["SS_num_loops"] + n_pfm + n_setpulse + 1)
            yi = np.concatenate((np.zeros(n_pfm), y_sp, yi))
            ni_read_vec = ni_read_vec + n_pfm + n_setpulse
            ni_write_vec = ni_write_vec + n_pfm + n_setpulse
            SS_read_vec = np.round(ni_read_vec + np.round(n_trans * interp_factor / 2))
            SS_write_vec = np.round(ni_write_vec + np.round(n_trans * interp_factor / 2))
            SS_write_vec = np.concatenate((SS_write_vec, [SS_write_vec[-1] + np.round(n_step * interp_factor)]))
            SS_read_vec[0] = np.round(ni_read_vec[0]) - interp_factor * n_trans / 2
            SS_wave = yi + self.SS_parm_vec["SS_read_voltage"]
            dc_amp_vec_full = dc_amp_vec_full + self.SS_parm_vec["SS_read_voltage"]
            SS_wave_nan = np.where(np.isnan(SS_wave))[0]
            SS_wave[SS_wave_nan] = 0
            SS_parm_out = np.arange(2, 12)
            
        return SS_wave,SS_read_vec,SS_write_vec,SS_parm_out

    def BEPS_wave_build(self,plot_cond_vec = 1, num_band_ring = 1,**kwargs):
        BE_ppw = 2**self.BE_parms_1["BE_ppw"]
        BE_rep = 2**self.BE_parms_1["BE_rep"]

        # Determine IO rate
        AO_rate, SS_step_t = self.determine_AO_rate(BE_ppw)
        AO_length = AO_rate * SS_step_t

        w_vec_full = np.arange(-AO_rate/2, AO_rate/2 + AO_rate/(AO_length-1) , AO_rate/(AO_length-1))
        w1_1 = self.BE_parms_1["BE_w_center"] - self.BE_parms_1["BE_w_width"]/2
        w2_1 = self.BE_parms_1["BE_w_center"] + self.BE_parms_1["BE_w_width"]/2
        w1_2 = self.BE_parms_2["BE_w_center"] - self.BE_parms_2["BE_w_width"]/2
        w2_2 = self.BE_parms_2["BE_w_center"] + self.BE_parms_2["BE_w_width"]/2
        w_ind_band_1 = np.where((w_vec_full >= w1_1) & (w_vec_full <= w2_1))[0]
        w_ind_band_2 = np.where((w_vec_full >= w1_2) & (w_vec_full <= w2_2))[0]


        BE_wave_1, BE_band_1 = BEWaveform.build_BE(self,self.BE_parms_1)
        
        BE_wave_1 = BE_wave_1 * self.BE_parms_1["BE_amp"]
        F_BE_wave_1 = np.fft.fftshift(np.fft.fft(BE_wave_1))
        F_BE_wave_1 = F_BE_wave_1[len(F_BE_wave_1)//2:]
        F2_BE_wave_1 = np.fft.fftshift(np.fft.fft(BE_wave_1**2))
        F2_BE_wave_1 = F2_BE_wave_1[len(F2_BE_wave_1)//2:]
        if plot_cond_vec== 1:
            BEWaveform.plot_BE_wave(1,BE_wave_1, BE_band_1, w_ind_band_1, w_vec_full, SS_step_t)

        if num_band_ring == 1:
            BE_wave_2, BE_band_2 = BEWaveform.build_BE(self,self.BE_parms_2)
            BE_wave_2 = BE_wave_2 * self.BE_parms_2["BE_amp"]
            F_BE_wave_2 = np.fft.fftshift(np.fft.fft(BE_wave_2))
            F_BE_wave_2 = F_BE_wave_2[:len(F_BE_wave_2)//2]
            if plot_cond_vec == 1:
                BEWaveform.plot_BE_wave(2, BE_wave_2, BE_band_2, w_ind_band_2, w_vec_full, SS_step_t)
        
        # Build SS waveform
        SS_wave,SS_read_vec,SS_write_vec,SS_parm_out = BEWaveform.build_SS(self)
        if 0:
            BEWaveform.plot_SS_wave(3,SS_wave,AO_rate,SS_read_vec,SS_write_vec)
                
        


                    

    
