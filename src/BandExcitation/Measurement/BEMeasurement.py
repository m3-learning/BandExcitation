from dataclasses import dataclass
from BandExcitation.Measurement.BEWaveform import BE_Spectroscopy, BEWaveform
from BandExcitation.Hardware import AO
from BandExcitation.Measurement.NI import FunctionGenerator, Oscilloscope, PXI
import numpy as np

@dataclass
class BEMeasurement:
    BE_time: float = 1e-3
    BE_ampl: float = 1
    BE_smoothing: float = 125
    BE_bandwidth: float = 60e3
    BE_center_freq: float = 500e3
    BE_ppw: int = 2**12
    BE_rep: int = 1
    BE_wave_type: str = 'chirp'
    BE_delay: tuple[float, float] = (0, 0)
    BE_chirp_direction: str = 'up'
    BE_num_bins: any = None 
    spectroscopy_type: str = "switching spectroscopy"
    spectroscopic_start_voltage: float = 0
    spectroscopic_min_voltage: float = -10
    spectroscopic_max_voltage: float = 10
    spectroscopic_cycles: int = 2
    spectroscopic_points: int = 96
    spectroscopic_offset: float = 0
    spectroscopic_phase_shift: float = 0
    spectroscopic_measurement_state: str = "on and off"
    AO_platform: str = "PXI-5412"
    AO_ext_amp: float = 1
    AO_trigger_channel: str = "PXI_Trig0"
    AO_channel: int = 0
    AO_resource_name = "Dev1"
    AI_platform: str = "PXI-6115"
    AI_photodiode_channel_num: int = 0
    AI_photodiode_vertical_range: float = 12
    AI_AWG_channel_num: int = None
    AI_AWG_vertical_range: float = 12
    AI_trigger_channel: str = "PXI_Trig0"
    AI_sample_rate: float = 1e6
    AI_number_of_points: int = 15000000
    AI_ref_position: int = 0
    AI_num_records: int = 1
    AI_enforce_realtime: bool = True
    AI_resource_name: str = "PXI1Slot6"

    
    def __post_init__(self):
        print("Initializing BEparams")
        self.system_initialization()
        self.system_checks()
        
    def system_initialization(self):
        self.update_be_spectroscopy()
        self.update_PXI()
        
    def update_PXI(self):
        self.update_function_generator()
        self.update_oscilloscope()
        self.PXI = PXI(self.function_generator, self.oscilloscope)
        
    def update_function_generator(self):
        self.function_generator = FunctionGenerator(BEwave=self.be_spectroscopy, 
                                                    resource_name = self.AO_resource_name, 
                                                    channel = self.AO_channel,
                                                    platform = self.AO_platform, 
                                                     trigger_channel = self.AO_trigger_channel)
    
    @property
    def AO_sample_rate(self):
        return self.be_spectroscopy.AO_rate

    @property
    def BE_freq_range(self):
        return (self.BE_center_freq - self.BE_bandwidth/2, self.BE_center_freq + self.BE_bandwidth/2)
    
    def BE_graph_range(self, multiple):
        return (self.BE_center_freq - self.BE_bandwidth/2*multiple, self.BE_center_freq + self.BE_bandwidth/2*multiple)
        
    def update_oscilloscope(self):
        self.oscilloscope = Oscilloscope(  BEwave=self.be_spectroscopy,
                                            resource_name =self.AI_resource_name,
                                            channel_num = self.AI_photodiode_channel_num,
                                            vertical_range = self.AI_photodiode_vertical_range,
                                            AWG_channel_num = self.AI_AWG_channel_num,
                                            AWG_vertical_range = self.AI_AWG_vertical_range,
                                            trigger_channel = self.AI_trigger_channel,
                                            sample_rate = self.AI_sample_rate,
                                            number_of_points = self.AI_number_of_points,
                                            ref_position = self.AI_ref_position,
                                            num_records = self.AI_num_records,
                                            enforce_realtime = self.AI_enforce_realtime)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.update_PXI()
    
    def update_be_spectroscopy(self):

        self.be_spectroscopy = BE_Spectroscopy(
                        self.BE_ppw, 
                        self.BE_rep,
                        type=self.spectroscopy_type,
                        start=self.spectroscopic_start_voltage,
                        max=self.spectroscopic_max_voltage,
                        min=self.spectroscopic_min_voltage,
                        cycles=self.spectroscopic_cycles,
                        BE_amplitude = self.BE_ampl,
                        points_per_cycle=self.spectroscopic_points,
                        phase_shift=self.spectroscopic_phase_shift,
                        center_freq=self.BE_center_freq,
                        bandwidth=self.BE_bandwidth,
                        wave=self.BE_wave_type,
                        waveform_time=self.BE_time,
                        BE_smoothing=self.BE_smoothing,
                        chirp_direction=self.BE_chirp_direction,
                        measurement_state=self.spectroscopic_measurement_state,
                        measurement_state_offset=self.spectroscopic_offset,
                        )
        
    def get_simulated_BE_measurement(self):
        """ Builds a waveform that is the BE waveform based on the AI sampling rate
        """
        self.AI_BE_ppw = int(self.AI_sample_rate * (self.BE_time + np.sum(self.BE_delay)))

        self.BE_AI_sim = BEWaveform(
            self.AI_BE_ppw,
            self.BE_rep,
            self.BE_ampl,
            center_freq=self.BE_center_freq,
            bandwidth=self.BE_bandwidth,
            wave = self.BE_wave_type,
            waveform_time = self.BE_time,
            BE_smoothing=self.BE_smoothing,
            chirp_direction=self.BE_chirp_direction,
        )

        return self.BE_AI_sim.BE_wave
    #TODO
    def system_checks(self):
        self.AO_check()

    def AO_check(self):
        if self.AO_platform == "PXI-5412":
            if self.be_spectroscopy.max_voltage > AO.pxi_5412.max_voltage*self.AO_ext_amp:
                #TODO add a better fix for the high voltage amplifier
                raise ValueError(f"Max voltage of {self.be_spectroscopy.max_voltage} too high for PXI-5412 with a \
                    max range of {AO.pxi_5412.max_voltage*self.AO_ext_amp}, consider adding a high voltage amplifier to the AO channel")

        if self.AO_sample_rate < self.BE_center_freq*2:
            raise ValueError(f"Sample rate of {self.AO_sample_rate} is too low for the center frequency of {self.BE_center_freq}, \
                based on the nyquest frequency, consider increasing the sample rate. This can be achieved by increasing the number of points per waveform")

    def AI_check(self):
        pass


    
    
    


# # BE_band_edge_smoothing_[s] : 4832.1 <span style="color:green">(False)</span>\
# # BE_band_edge_trim : 0.094742 <span style="color:green">(False)</span>\
# # BE_bins_per_band : 0 <span style="color:green">(False)</span>\
# # BE_center_frequency_[Hz] : 1310000 <span style="color:green">(True: center_freq = 500e3)</span>\
# # BE_desired_duration_[s] : 0.004 <span style="color:green">(False)</span>\
# # BE_phase_content : chirp-sinc hybrid <span style="color:green">(True: wave = chirp)</span>\
# # BE_phase_variation : 1 <span style="color:green">(False)</span>\
# # BE_points_per_BE_wave : 0 <span style="color:green">(True: BE_ppw = input)</span>\
# # BE_repeats : 4 <span style="color:green">(True: BE_rep = input)</span>\
# # FORC_V_high1_[V] : 1 <span style="color:green">(False)</span>\
# # FORC_V_high2_[V] : 10 <span style="color:green">(False)</span>\
# # FORC_V_low1_[V] : -1 <span style="color:green">(False)</span>\
# # FORC_V_low2_[V] : -10 <span style="color:green">(False)</span>\
# # FORC_num_of_FORC_cycles : 1 <span style="color:green">(False)</span>\
# # FORC_num_of_FORC_repeats : 1 <span style="color:green">(False)</span>\
# File_MDAQ_version : MDAQ_VS_090915_01 <span style="color:green">(False)</span>\
# File_date_and_time : 18-Sep-2015 18:32:14 <span style="color:green">(False)</span>\
# File_file_name : SP128_NSO <span style="color:green">(False)</span>\
# File_file_path : C:\Users\Asylum User\Documents\Users\Agar\SP128_NSO\ <span style="color:green">(False)</span></span>\
# File_file_suffix : 99 <span style="color:green">(False)</span>\
# IO_AO_amplifier : 10 <span style="color:green">(False)</span>\
# IO_AO_range_[V] : +/- 10 <span style="color:green">(False)</span>\
# IO_Analog_Input_1 : +/- .1V, FFT <span style="color:green">(False)</span>\
# IO_Analog_Input_2 : off <span style="color:green">(False)</span>\
# IO_Analog_Input_3 : off <span style="color:green">(False)</span>\
# IO_Analog_Input_4 : off <span style="color:green">(False)</span>\
# IO_DAQ_platform : NI 6115 <span style="color:green">(True: platform="PXI-5412")</span>\
# IO_rate_[Hz] : 4000000 <span style="color:green">(False)</span>\
# # VS_amplitude_[V] : 16 <span style="color:green">(False)</span>\
# VS_cycle_fraction : full <span style="color:green">(False)</span>\
# VS_cycle_phase_shift : 0 <span style="color:green">(True: phase_shift=None)</span>\
# VS_measure_in_field_loops : in and out-of-field <span style="color:green">(False)</span>\
# VS_mode : DC modulation mode <span style="color:green">(False)</span>\
# VS_number_of_cycles : 2 <span style="color:green">(True: cycles=None)</span>\
# VS_offset_[V] : 0 <span style="color:green">(False)</span>\
# VS_read_voltage_[V] : 0 <span style="color:green">(False)</span>\
# VS_set_pulse_amplitude[V] : 0 <span style="color:green">(False)</span>\
# VS_set_pulse_duration[s] : 0.002  <span style="color:green">(False)</span>\
# VS_step_edge_smoothing_[s] : 0.001 <span style="color:green">(False)</span>\
# VS_steps_per_full_cycle : 96 <span style="color:green">(True: points_per_cycle=None)</span>\
# data_type : BEPSData <span style="color:green">(False)</span>\
# grid_/single : grid <span style="color:green">(False)</span>\
# grid_contact_set_point_[V] : 1 <span style="color:green">(False)</span>\
# grid_current_col : 1 <span style="color:green">(False)</span>\
# grid_current_row : 1 <span style="color:green">(False)</span>\
# grid_cycle_time_[s] : 10 <span style="color:green">(False)</span>\
# grid_measuring : 0 <span style="color:green">(False)</span>\
# grid_moving : 0 <span style="color:green">(False)</span>\
# grid_num_cols : 60 <span style="color:green">(False)</span>\
# grid_num_rows : 60 <span style="color:green">(False)</span>\
# grid_settle_time_[s] : 0.15 <span style="color:green">(False)</span>\
# grid_time_remaining_[h;m;s] : 10 <span style="color:green">(False)</span>\
# grid_total_time_[h;m;s] : 10 <span style="color:green">(False)</span>\
# grid_transit_set_point_[V] : 0.1 <span style="color:green">(False)</span>\
# grid_transit_time_[s] : 0.15 <span style="color:green">(False)</span>\
# num_bins : 165 <span style="color:green">(False)</span>\
# num_pix : 3600 <span style="color:green">(False)</span>\
# num_udvs_steps : 384 <span style="color:green">(False)</span>\