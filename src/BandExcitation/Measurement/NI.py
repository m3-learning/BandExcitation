import nifgen
import niscope
from nifgen import enums
from niscope import enums as scope_enums
import numpy as np


class FunctionGenerator(nifgen.Session):
    def __init__(self, 
                 BEwave, 
                 resource_name, 
                 channel = 0, 
                 platform="PXI-5413", 
                 **kwargs) -> None:
        
        super().__init__(resource_name=resource_name, **kwargs)
        
        self.BEwave = BEwave
        
        self.platform = platform
        
        # makes sure the session is aborted from previous runs
        self.abort()
        self.channel_num = channel
        
        self.output_mode = nifgen.OutputMode.ARB
        self.arb_sample_rate = self.BEwave.AO_rate
        self.channels[self.channel_num].trigger_mode = enums.TriggerMode.SINGLE
        self.analog_path = nifgen.AnalogPath.FIXED_HIGH_GAIN
        self.exported_start_trigger_output_terminal = "PXI_Trig0"
        
        self.construct_arb_waveform()
        
        self.initiate()
        
    def construct_arb_waveform(self):
    
        #gets the excitation from the BEWave object
        excitation = self.BEwave.cantilever_excitation_waveform
        
        self.scale_wave(excitation)
        
        waveform = excitation/self.gain
        
        self.arb_waveform_handle = self.create_waveform(waveform_data_array=waveform)
        
        self.channels[self.channel_num].configure_arb_waveform(waveform_handle=self.arb_waveform_handle, 
                                                               gain=self.gain)
        
    def scale_wave(self, wave):
        
        if np.max(np.abs(wave)) > 6:
            raise ValueError("Waveform outside AWG voltage range, an Amp is needed")
        
        self.gain = np.ceil(np.max(np.abs(wave)))
        
        
        