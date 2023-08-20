import nifgen
import niscope
from nifgen import enums
from niscope import enums as scope_enums
import numpy as np


class FunctionGenerator(nifgen.Session):
    def __init__(self, resource_name, platform="PXI-5413", **kwargs) -> None:
        
        super().__init__(BEwave, resource_name=resource_name, **kwargs)
        
        self.BEwave = BEwave
        
        self.platform = platform
        
        # makes sure the session is aborted from previous runs
        self.abort()
        
        fgen.output_mode = nifgen.OutputMode.ARB
        fgen.arb_sample_rate = out.AO_rate
        fgen.channels[0].trigger_mode = enums.TriggerMode.SINGLE
        # nifgen.AnalogPath.FIXED_HIGH_GAIN
        fgen.analog_path = nifgen.AnalogPath.FIXED_HIGH_GAIN

        #fgen.channels[0].trigger_mode = enums.TriggerMode.CONTINUOUS
        waveform_handle = fgen.create_waveform(waveform_data_array=waveform)

        fgen.exported_start_trigger_output_terminal = "PXI_Trig0"

        fgen.arb_waveform_handle = waveform_handle
        # fgen.streaming_waveform_handle = waveform_handle
        fgen.channels[0].configure_arb_waveform(waveform_handle=waveform_handle, gain=4, offset=0)

        fgen.initiate()
        