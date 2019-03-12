# Defines class to represent a signal - used in chunkifying

from taiyaki import fast5utils


class Signal:
    """
    Represents a read, with constructor
    obtaining signal data from a fast5 file
    or from a numpy array for testing.

    The only fiddly bit is that .untrimmed_dacs contains
    all the (integer) current numbers available, while
    .dacs and .current are trimmed according to the trimming
    parameters.
    """

    def __init__(self, read=None, dacs=None):
        """Loads data from read in fast5 file.
        If read is None
        and dacs is a np array then initialse the untrimmed_dacs to this array.
        (this allows testing with non-fast5 data)

        param read : an ont_fast5_api read object
        param dacs : np int array (only used if first param is None)
        """
        if read is None:
            try:
                self.untrimmed_dacs = dacs.copy()
            except:
                raise Exception("Cannot initialise SignalWithMap object")
            self.offset = 0
            self.range = 1
            self.digitisation = 1
        else:
            self.channel_info = {k: v for k, v in fast5utils.get_channel_info(read).items()}
            # channel_info contains attributes of the channel such as calibration parameters and sample rate
            self.read_attributes = {k: v for k, v in fast5utils.get_read_attributes(read).items()}
            # read_attributes includes read id, start time, and active mux
            #print("Channel info:",[(k,v) for k,v in self.channel_info.items()])
            #print("Read attributes:",[(k,v) for k,v in self.read_attributes.items()])
            # the sample number (counted from when the device was switched on) when the signal starts
            self.start_sample = self.read_attributes['start_time']
            self.sample_rate = self.channel_info['sampling_rate']
            # a unique key corresponding to this read
            self.read_id = self.read_attributes['read_id'].decode("utf-8")
            # digitised current levels.
            # this function returns a copy, not a reference.
            self.untrimmed_dacs = read.get_raw_data()
            # parameters to convert between DACs and picoamps
            self.range = self.channel_info['range']
            self.offset = self.channel_info['offset']
            self.digitisation = self.channel_info['digitisation']

        # We want to allow trimming without mucking about with the original data
        # To start with, set trimming parameters to trim nothing
        self.signalstart = 0
        # end is defined exclusively so that self.dacs[signalstart:signalend_exc] is the bit we want.
        self.signalend_exc = len(self.untrimmed_dacs)

    def set_trim_absolute(self, trimstart, trimend):
        """trim trimstart samples from the start and trimend samples from the end, starting
        with the whole stored data set (not starting with the existing trimmed ends)
        """
        untrimmed_len = len(self.untrimmed_dacs)
        if trimstart < 0 or trimend < 0:
            raise Exception("Can't trim a negative amount off the end of a signal vector.")
        if trimstart + trimend >= untrimmed_len:  # Nothing left!
            trimstart = 0
            trimend = 0
        self.signalstart = trimstart
        self.signalend_exc = untrimmed_len - trimend

    def set_trim_relative(self, trimstart, trimend):
        """trim trimstart samples from the start and trimend samples from the end, starting
        with the existing trimmed ends
        """
        untrimmed_len = len(self.untrimmed_dacs)
        self.set_trim_absolute(self.signalstart + trimstart, (untrimmed_len - self.signalend_exc) + trimend)

    @property
    def dacs(self):
        """dac numbers, trimmed according to trimming parameters"""
        return self.untrimmed_dacs[self.signalstart:self.signalend_exc].copy()

    @property
    def untrimmed_current(self):
        """Signal measured in pA, untrimmed"""
        return (self.untrimmed_dacs + self.offset) * self.range / self.digitisation

    @property
    def current(self):
        """Signal measured in pA, trimmed according to trimming parameters"""
        return (self.dacs + self.offset) * self.range / self.digitisation

    @property
    def trimmed_length(self):
        """Trimmed length of the signal in samples"""
        return self.signalend_exc - self.signalstart
