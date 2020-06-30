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

    def __init__(self, read=None, dacs=None,
                 channel_info={'offset': 0, 'range': 1, 'digitisation': 1,
                               'sampling_rate': 4000},
                 read_id=None, read_params={'trim_start': 0, 'trim_end': 0,
                                            'shift': 0, 'scale': 1}):
        """Constructor for Signal class. Loads data from read in fast5 file.
        
        If read is None and dacs is a np array then initialise the
        untrimmed_dacs to this array.
        (this allows testing with non-fast5 data)

        Args:
        
        read (ont_fast5_api read object) : the read data
        dacs (np int array) : (only used if first param is None)
        channel_info (dict) :  containing keys: offset, range, digitisation,
                                    and sampling_rate.
        read_id (str): UUID read identifier
        read_params (dict): dictionary containing keys: trim_start, trim_end,
                           shift, and scale (as returned from
                      prepare_mapping_funcs.get_per_read_params_dict_from_tsv)
                           
        Returns:
            new Signal object.
        """
        if read is None:
            try:
                self.untrimmed_dacs = dacs.copy()
            except:
                raise Exception("Cannot initialise Signal object")
            self.channel_info = channel_info
            self.read_id = read_id
        else:
            self.channel_info = dict(fast5utils.get_channel_info(read).items())
            # UUID read identifier
            self.read_id = fast5utils.get_read_attributes(read)[
                'read_id'].decode("utf-8")
            # digitised current levels.
            # this function returns a copy, not a reference.
            self.untrimmed_dacs = read.get_raw_data()

        self.sample_rate = self.channel_info['sampling_rate']
        # parameters to convert between DACs and picoamps
        self.range = self.channel_info['range']
        self.offset = self.channel_info['offset']
        self.digitisation = self.channel_info['digitisation']

        # We want to allow trimming without mucking about with the original data
        # To start with, set trimming parameters to trim nothing
        self.signalstart = 0
        # end is defined exclusively so that
        # self.dacs[signalstart:signalend_exc] is the bit we want.
        self.signalend_exc = len(self.untrimmed_dacs)

        self.set_trim_absolute(read_params['trim_start'],
                               read_params['trim_end'])
        self.shift_from_pA = read_params['shift']
        self.scale_from_pA = read_params['scale']

    def set_trim_absolute(self, trimstart, trimend):
        """trim trimstart samples from the start and trimend samples from the
        end, starting with the whole stored data set (not starting with the
        existing trimmed ends).
        
        Args:
            trimstart (int) : number of samples to trim from the start
            trimend (int)   : number of samples to trim from the end
        """
        untrimmed_len = len(self.untrimmed_dacs)
        if trimstart < 0 or trimend < 0:
            raise Exception("Can't trim a negative amount off the end of a " +
                            "signal vector.")
        # Nothing left!
        if trimstart + trimend >= untrimmed_len:
            trimstart = 0
            trimend = 0
        self.signalstart = trimstart
        self.signalend_exc = untrimmed_len - trimend

    @property
    def dacs(self):
        """Returns:
            np int array : dac numbers, trimmed according to trimming params"""
        return self.untrimmed_dacs[self.signalstart:self.signalend_exc].copy()

    @property
    def untrimmed_current(self):
        """Returns:
            np float array : Signal measured in pA, untrimmed"""
        return ((self.untrimmed_dacs + self.offset) *
                self.range / self.digitisation)

    @property
    def current(self):
        """Returns:
            np float array: Signal measured in pA, trimmed according to
                            trimming parameters"""
        return (self.dacs + self.offset) * self.range / self.digitisation

    @property
    def standardized_current(self):
        """Returns:
            np float array : Signal measured in standardized units according
                              to read_params, trimmed according to
                              trimming parameters"""
        return (self.current - self.shift_from_pA) / self.scale_from_pA
