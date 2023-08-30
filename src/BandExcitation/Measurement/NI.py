import nifgen
import niscope
from nifgen import enums
from niscope import enums as scope_enums
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
from dataclasses import dataclass


# class FunctionGenerator(nifgen.Session):

#     """
#     Object for the function generator, inherits from nifgen.Session
#     """

#     def __init__(
#         self,
#         BEwave,
#         resource_name,
#         channel=0,
#         platform="PXI-5413",
#         trigger_channel="PXI_Trig0",
#         **kwargs,
#     ):
#         """
#         __init__ Initialization function for the function generator object

#         Args:
#             BEwave (obj): BE wave object
#             resource_name (string): location with the NI card is located
#             channel (int, optional): output channel for the waveform. Defaults to 0.
#             platform (str, optional): platform of the waveform generator. Defaults to "PXI-5413".
#             trigger_channel (str, optional): Channel where the trigger exists. Defaults to "PXI_Trig0".
#         """
#         super().__init__(resource_name=resource_name, **kwargs)

#         self.BEwave = BEwave
#         self.platform = platform
#         self.trigger_channel = trigger_channel
#         self.channel_num = channel

#         # makes sure the session is aborted from previous runs
#         self.reset()
        
#         # sets the output mode of the function generator
#         self.output_mode = nifgen.OutputMode.ARB

#         # sets the AO rate based on the BEwave object
#         self.arb_sample_rate = self.BEwave.AO_rate

#         # sets the triggering mode of the channel
#         self.channels[self.channel_num].trigger_mode = enums.TriggerMode.SINGLE

#         # sets the gain path for the function generator
#         self.analog_path = nifgen.AnalogPath.FIXED_HIGH_GAIN

#         # sets the output trigger channel
#         self.exported_start_trigger_output_terminal = self.trigger_channel

#         # constructs the waveform on the generator
#         self.construct_arb_waveform()

@dataclass
class FunctionGenerator(nifgen.Session):
    """
    Object for the function generator, inherits from nifgen.Session
    """

    BEwave: object
    resource_name: str
    channel: int = 0
    platform: str = "PXI-5412"
    trigger_channel: str = "PXI_Trig0"

    def __post_init__(self):
        """
        Initialization function for the function generator object

        Args:
            BEwave (obj): BE wave object
            resource_name (string): location with the NI card is located
            channel (int, optional): output channel for the waveform. Defaults to 0.
            platform (str, optional): platform of the waveform generator. Defaults to "PXI-5412".
            trigger_channel (str, optional): Channel where the trigger exists. Defaults to "PXI_Trig0".
        """
        super().__init__(resource_name=self.resource_name)

        self.BEwave = self.BEwave
        self.platform = self.platform
        self.trigger_channel = self.trigger_channel
        self.channel_num = self.channel

        # makes sure the session is aborted from previous runs
        self.reset()
        
        # sets the output mode of the function generator
        self.output_mode = nifgen.OutputMode.ARB

        # sets the AO rate based on the BEwave object
        self.arb_sample_rate = self.BEwave.AO_rate

        # sets the triggering mode of the channel
        self.channels[self.channel_num].trigger_mode = enums.TriggerMode.SINGLE

        # sets the gain path for the function generator
        self.analog_path = nifgen.AnalogPath.FIXED_HIGH_GAIN

        # sets the output trigger channel
        self.exported_start_trigger_output_terminal = self.trigger_channel

        # constructs the waveform on the generator
        self.construct_arb_waveform()


    def __setattr__(self, name, value): 
        object.__setattr__(self, name, value)

    def run(self, timeout=30):
        # initiates the function generator
        self.initiate()

        try:
            func_timeout(30, self.check_status)
        except FunctionTimedOut:
            print(f"Function timed out after {30} seconds.")

        self.abort()

    def check_status(self):
        while self.is_done() is False:
            pass

    def construct_arb_waveform(self):
        """
        construct_arb_waveform constructs the waveform on the function generator
        """

        # gets the excitation from the BEwave object
        excitation = self.BEwave.cantilever_excitation_waveform

        self.scale_wave(excitation)

        self.waveform = excitation / self.gain

        self.arb_waveform_handle = self.create_waveform(waveform_data_array=self.waveform)

        self.channels[self.channel_num].configure_arb_waveform(
            waveform_handle=self.arb_waveform_handle, gain=self.gain/2, offset=0,
        )

    def scale_wave(self, wave):
        """
        scale_wave sets the scale and the gain of the function generator. The waveform to the function generator must be between -1 and 1 V. The gain is used to increase the voltage.

        Args:
            wave (np.array): wave to scale

        Raises:
            ValueError: Waveform outside the maximum voltage range, an amplifier is needed.
        """

        if np.max(np.abs(wave)) > 6:
            raise ValueError("Waveform outside AWG voltage range, an Amp is needed")

        self.gain = np.ceil(np.max(np.abs(wave)))


# class Oscilloscope(niscope.Session):
#     def __init__(
#         self,
#         BEwave,
#         resource_name,
#         channel_num=0,
#         vertical_range=12,
#         AWG_channel_num=None,
#         AWG_vertical_range=12,
#         trigger_channel="PXI_Trig0",
#         sample_rate=1e6,
#         number_of_points=15000000,
#         ref_position=0,
#         num_records=1,
#         enforce_realtime=True,
#         **kwargs,
#     ):
#         super().__init__(resource_name)
        
#         self.BEwave = BEwave
        

#         self.cantilever_response_channel = self.Channel(
#             channel_num,
#             vertical_range,
#             sample_rate,
#             number_of_points,
#             ref_position,
#             num_records,
#             enforce_realtime,
#             trigger_channel,
#         )

#         if AWG_channel_num is not None:
#             self.excitation_channel = self.Channel(
#                 AWG_channel_num,
#                 AWG_vertical_range,
#                 sample_rate,
#                 number_of_points,
#                 ref_position,
#                 num_records,
#                 enforce_realtime,
#                 trigger_channel,
#             )

#         self.config_scope(self.cantilever_response_channel)
#         self.initiate()

from dataclasses import dataclass
import niscope

@dataclass
class Oscilloscope(niscope.Session):
    """Oscilloscope class encapsulating an NI-SCOPE session and configuration.

    Attributes:
        BEwave: Waveform for the Band Excitation (BE) process.
        resource_name: Resource name for the NI-SCOPE session.
        channel_num: Channel number for cantilever response (default 0).
        vertical_range: Vertical range for the cantilever response channel (default 12).
        AWG_channel_num: Channel number for excitation (default None).
        AWG_vertical_range: Vertical range for the excitation channel (default 12).
        trigger_channel: Trigger channel name (default "PXI_Trig0").
        sample_rate: Sample rate in samples per second (default 1e6).
        number_of_points: Number of points to acquire (default 15000000).
        ref_position: Reference position for acquisition (default 0).
        num_records: Number of records to acquire (default 1).
        enforce_realtime: Whether to enforce real-time acquisition (default True).
    """
    BEwave: object
    resource_name: str
    channel_num: int = 0
    vertical_range: float = 12
    AWG_channel_num: int = None
    AWG_vertical_range: float = 12
    trigger_channel: str = "PXI_Trig0"
    sample_rate: float = 1e6
    number_of_points: int = 15000000
    ref_position: int = 0
    num_records: int = 1
    enforce_realtime: bool = True

    def __post_init__(self):
        super().__init__(self.resource_name)
        
        self.cantilever_response_channel = self.Channel(
            self.channel_num,
            self.vertical_range,
            self.sample_rate,
            self.number_of_points,
            self.ref_position,
            self.num_records,
            self.enforce_realtime,
            self.trigger_channel,
        )

        if self.AWG_channel_num is not None:
            self.excitation_channel = self.Channel(
                self.AWG_channel_num,
                self.AWG_vertical_range,
                self.sample_rate,
                self.number_of_points,
                self.ref_position,
                self.num_records,
                self.enforce_realtime,
                self.trigger_channel,
            )
            self.config_scope(self.excitation_channel)

        self.config_scope(self.cantilever_response_channel)


    class Channel:
        def __init__(
            self,
            channel_num,
            vertical_range,
            sample_rate,
            number_of_points,
            ref_position,
            num_records,
            enforce_realtime,
            trigger_channel,
        ):
            self.channel_num = channel_num
            self.vertical_range = vertical_range
            self.vertical_range = vertical_range
            self.trigger_channel = trigger_channel
            self.sample_rate = sample_rate
            self.number_of_points = number_of_points
            self.ref_position = ref_position
            self.num_records = num_records
            self.enforce_realtime = enforce_realtime

    def config_scope(
        self,
        channel,
    ):
        # TODO add a kwargs update function

        self.channels[channel.channel_num].configure_vertical(
            range=channel.vertical_range, coupling=niscope.VerticalCoupling.DC
        )

        self.configure_trigger_digital(
            channel.trigger_channel, slope=scope_enums.TriggerSlope.POSITIVE
        )

        self.configure_horizontal_timing(
            min_sample_rate=int(channel.sample_rate),
            min_num_pts=channel.number_of_points,
            ref_position=channel.ref_position,
            num_records=channel.num_records,
            enforce_realtime=channel.enforce_realtime,
         )

        

    def __setattr__(self, name, value): 
        object.__setattr__(self, name, value)
        
    def run(self):

        channels_ = [self.cantilever_response_channel.channel_num]

        if self.AWG_channel_num is not None:
            channels_.append(self.excitation_channel.channel_num)
    
        wfm = self.channels[channels_].fetch(num_samples=int(self.BEwave.cantilever_excitation_time*self.cantilever_response_channel.sample_rate))        
        
        self.abort()
        
        return wfm
    

class PXI:

    def __init__(self, function_generator, oscilloscope) -> None:
        self.function_generator = function_generator
        self.oscilloscope = oscilloscope

    def run(self):
        self.oscilloscope.initiate()
        self.function_generator.run()
        wfm = self.oscilloscope.run()
        return wfm
