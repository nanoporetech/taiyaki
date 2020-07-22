# Defines class to represent a read - used in chunkifying
# class also includes data and methods to deal with mapping table
# Also defines iterator giving all f5 files in directory, optionally using a
# read list

from collections import namedtuple

import numpy as np


class TaiyakiSigMapError(Exception):
    """ Custom Taiyaki signal mapping error for more graceful error handling
    """
    pass


class SignalMapping:
    """Represents a mapping between a signal and a reference, with attributes
    including the signal and the reference.

    We use the trimming parameters from the signal object to set limits on
    outputs (including chunks).
    """
    # data types for required/optional datasets/attributes
    req_data_types = namedtuple('req_data_types', (
        'read_id', 'shift_frompA', 'scale_frompA',
        'range', 'offset', 'digitisation',
        'Dacs', 'Ref_to_signal', 'Reference'))(
            str, float, float,
            float, float, float,
            np.int16, np.int32, np.int16)
    opt_data_types = namedtuple('opt_data_types', (
        'mapping_score', 'mapping_method'))(float, str)
    np_scalar_types = {
        float: np.floating,
        int: np.integer,
        bool: np.bool_
    }
    pass_str = 'pass'

    @staticmethod
    def is_numpy(x):
        """Is this a numpy array?

        Args:
            x : object to be tested

        Returns:
            bool : is it a numpy array?
        """
        return hasattr(x, 'dtype')

    def _typecheck(self, name):
        """Check the type of an attribute is correct.

        Args:
            name (str): name of the attr

        Returns:
            str :  empty string if type matches, , or error string
                   describing the problem if not
        """
        is_req = name in self.req_data_types._fields
        is_opt = name in self.opt_data_types._fields
        if not is_req or is_opt:
            return 'Invalid attribute name "' + name + '".\n'
        target_type = (getattr(self.req_data_types, name) if is_req else
                       getattr(self.opt_data_types, name))
        value = getattr(self, name)
        if self.is_numpy(target_type):
            if type(value) != np.ndarray:
                return "Type of attribute " + name + " is not np.ndarray\n"
            if value.dtype != target_type:
                return ("Data type of items in numpy array " + name +
                        " is not " + target_type + "\n")
        elif target_type in self.np_scalar_types:
            if not np.issubdtype(type(value),
                                 self.np_scalar_types[target_type]):
                return ('Type of attribute "{}" is "{}" and should be ' +
                        '"{}".\n').format(name, type(value), target_type)
        else:
            if not isinstance(value, target_type):
                return ('Type of attribute "{}" is "{}" and should be ' +
                        '"{}".\n').format(name, type(value), target_type)
        return ""

    def check(self):
        """Perform some checks on the attributes of a class instance.

        Returns:
            str : self.pass_str if read info passes some integrity tests.
                   Return failure information as string otherwise."""
        # type checking
        return_string = ''.join(self._typecheck(k)
                                for k in self.req_data_types._fields)
        return_string += ''.join(self._typecheck(k)
                                 for k in self.opt_data_types._fields
                                 if getattr(self, k) is not None)

        # Some validity checks
        maplen = len(self.Ref_to_signal)
        if self.reflen + 1 != maplen:
            return_string += ("Length of Ref_to_signal ({}) should be 1 + " +
                              "length of Reference ({})\n").format(
                                  maplen, self.reflen)
        # -1 and len(Dacs) + 1 are used as end markers
        if np.min(self.Ref_to_signal) < -1 or \
           np.max(self.Ref_to_signal) > len(self.Dacs) + 1:
            return_string += ("Range of locations in mapping exceeds " +
                              "length of Dacs\n")
        if np.any(np.diff(self.Ref_to_signal) < 0):
            return_string += "Mapping does not increase monotonically\n"

        if len(return_string) == 0:
            return self.pass_str
        return return_string

    def __init__(
            self, Ref_to_signal, Reference, *, signalObj=None,
            signalstart=None, shift_frompA=None, scale_frompA=None, range=None,
            offset=None, digitisation=None, read_id=None, Dacs=None,
            mapping_score=None, mapping_method=None):
        """Construct SignalMapping object from data.
        Args:
            Ref_to_signal (np int array): Assignment of reference positions to
                                          signal points. See get_reftosignal.
            Reference (np int array): Integer encoded reference sequence.
            signalObj (taiyaki.signal.Signal object): signal data
            signalstart (int): start of mapping within dacs
            shift_frompA (float): shift from pA scaling to standardised scaling
            scale_frompA (float): scale from pA scaling to standardised scaling
            range (float): pA scaling range parameter
            offset (float): pA scaling offset parameter
            digitisation (float): pA scaling digitisation parameter
            read_id (str): read UUID
            Dacs (np int16 array): Raw data aquisition values
            mapping_score (float): Mapping score
            mapping_method (str): Mapping method

        Returns:
            SignalMapping object.

        Note:
            Must provide either signalObj or all of: signalstart, shift_frompA,
            scale_frompA, range, offset, digitisation, read_id and Dacs.
        """
        # required attrs
        self.Ref_to_signal = Ref_to_signal.astype(
            self.req_data_types.Ref_to_signal)
        self.Reference = Reference.astype(self.req_data_types.Reference)
        if signalObj is None:
            self.shift_frompA = self.req_data_types.shift_frompA(shift_frompA)
            self.scale_frompA = self.req_data_types.scale_frompA(scale_frompA)
            self.range = self.req_data_types.range(range)
            self.offset = self.req_data_types.offset(offset)
            self.digitisation = self.req_data_types.digitisation(digitisation)
            self.read_id = self.req_data_types.read_id(read_id)
            self.Dacs = Dacs.astype(self.req_data_types.Dacs)
        else:
            self.shift_frompA = self.req_data_types.shift_frompA(
                signalObj.shift_from_pA)
            self.scale_frompA = self.req_data_types.scale_frompA(
                signalObj.scale_from_pA)
            self.range = self.req_data_types.range(signalObj.range)
            self.offset = self.req_data_types.offset(signalObj.offset)
            self.digitisation = self.req_data_types.digitisation(
                signalObj.digitisation)
            self.read_id = self.req_data_types.read_id(signalObj.read_id)
            self.Dacs = signalObj.untrimmed_dacs.astype(
                self.req_data_types.Dacs)

        # optional attrs
        self.mapping_score = mapping_score
        self.mapping_method = mapping_method

        # set data types for alternative values
        for opt_name in self.opt_data_types._fields:
            if getattr(self, opt_name) is None:
                continue
            if self.is_numpy(getattr(self, opt_name)):
                setattr(self, opt_name, getattr(self, opt_name).astype(
                    getattr(self.opt_data_types, opt_name)))
            else:
                setattr(self, opt_name, getattr(self.opt_data_types, opt_name)(
                    getattr(self, opt_name)))

        self.siglen = self.Dacs.shape[0]
        self.reflen = self.Reference.shape[0]

        return

    @staticmethod
    def get_integer_reference(string_reference, alphabet):
        """Get integer-coded reference sequence.

        Args:
            string_reference (str) : sequence
            alphabet (str) : alphabet

        Returns:
            np int array : integer coded ref
        """
        return np.array([
            alphabet.index(i) for i in string_reference],
            dtype=SignalMapping.req_data_types.Reference)

    @staticmethod
    def get_reftosignal(signalpos_to_refpos, reflen, siglen):
        """Return integer vector reftosig,  mapping reference to signal.

        Args:
            signalpos_to_refpos (np int array) : vector giving a ref position
                                                for each location in the signal
            reflen (int) : length of the ref
            siglen (int) : length of the signal
        Returns:
            np int array : reftosig, giving locations in signal for each point
                            in the ref.

        Note:
              Length of reftosig returned is (1 + reflen),
              where reflen = len(reference)
              reftosig[n] is the location in the untrimmed
              dacs where the base at reference[n] starts.

            The last element, reftosig[reflen] is consistent with this scheme:
            it is (1 + (last location in untrimmed dacs))

            if the start of the reference is not mapped, then reftosig will
            begin with a sequence of (-1)s

            if the end of the reference is not mapped, then reftosig will end
            with         ... f, f, f, f]  where f = siglen + 1.
        """
        rts_dt = SignalMapping.req_data_types.Ref_to_signal
        valid_sig_to_ref_idxs = np.where(
            signalpos_to_refpos != -1)[0].astype(rts_dt)
        # if the full read is clipped return an array of negative ones
        if len(valid_sig_to_ref_idxs) == 0:
            return -1 * np.ones(reflen + 1, dtype=rts_dt)

        valid_sig_to_ref = signalpos_to_refpos[valid_sig_to_ref_idxs]
        move_pos = np.concatenate([[1, ], np.diff(valid_sig_to_ref)])
        ref_to_sig = np.repeat(valid_sig_to_ref_idxs, move_pos)
        # add the end of the last mapped position
        ref_to_sig = np.concatenate([
            ref_to_sig,
            np.array([valid_sig_to_ref_idxs[-1] + 1, ], dtype=rts_dt)])

        # Insert the right number of -1s to get to the beginning of the
        # mapped region
        if valid_sig_to_ref[0] > 0:
            ref_to_sig = np.concatenate([
                -1 * np.ones(valid_sig_to_ref[0], dtype=rts_dt),
                ref_to_sig])
        if reflen + 1 > len(ref_to_sig):
            ref_to_sig = np.append(ref_to_sig, (siglen + 1) * np.ones(
                reflen + 1 - len(ref_to_sig), dtype=rts_dt))

        return ref_to_sig

    @classmethod
    def from_remapping_path(
            _class, sigtoref_downsampled, reference, stride, sig):
        """
        Construct Mapping object based on downsampled mapping information
        (rather than just copying sigtoref).

        Args:
            sigtoref_downsampled (numpy int vector): sigtoref_downsampled[k] is
                                        the location in the reference of the
                                        base starting at
                                        untrimmed_dacs[k*stride-1+signalstart]
            reference (numpy int16 array) : reference sequence. See
                                `signal_mapping.Mapping.get_integer_reference`
            stride (int): model stride
            sig (taiyaki.signal.Signal object) : signal data

        Returns:
            SignalMapping object.

        Note:

        There is a bit of freedom in where to put the signal locations
        because of transition weights
        and downsampling. We use a picture like this, shown for the case
        stride = 2

        signal                [0]   [1]   [2]   [3]   [4]   [5]
        blocks                ---------   ---------   ---------
        trans weights            [0]         [1]         [2]
        sigtoref           [0]         [1]         [2]         [3]
        mapping to signal             /           /           /
        signal                [0]   [1]   [2]   [3]   [4]   [5]

        in other words sigtoref[n] maps to signal[stride*n-1]
        Note that the very first element of the sigtoref vector that comes
        from remapping is ignored with this choice
        """
        rts_dt = SignalMapping.req_data_types.Ref_to_signal
        # Create null dacstoref ranging of the full (not downsampled) range
        # of locations and fill in the locations determined by the mapping
        # -1 means nothing  associated with this location
        fullsigtoref = np.full(len(sig.untrimmed_dacs), -1, dtype=rts_dt)
        # Calculate locations in signal associated with each location in the
        # input sigtoref There is a bit of freedom here: see the docstring
        siglocs = (np.arange(len(sigtoref_downsampled), dtype=rts_dt) *
                   stride - 1 + sig.signalstart)
        # Keep only signal locations between 0 and len(untrimmed_dacs)
        f = np.logical_and(np.greater_equal(siglocs, 0),
                           np.less(siglocs, len(fullsigtoref)))
        # Put numbers in
        fullsigtoref[siglocs[f]] = sigtoref_downsampled[f]
        ref_to_sig = SignalMapping.get_reftosignal(
            fullsigtoref, reference.shape[0], sig.untrimmed_dacs.shape[0])

        return _class(ref_to_sig, reference, signalObj=sig)

    def get_read_dictionary(self, check=True):
        """Return a read dict for insertion via AbstractMappedSignalWriter

        Args:
            check (bool): if True, do checking on the SignalMapping object.

        Returns:
            dict : contains all the attributes of the SignalMapping object.

        Raises:
            taiyaki.signal_mapping.TaiyakiSigMapError : if check=True and read
                fails check method.

        Note:
            We return the dictionary, not the object itself. That's because
                this method is used inside worker processes which need to pass
                their results out through the pickling mechanism in imap_mp.
        """
        if check:
            check_str = self.check()
            if check_str != self.pass_str:
                raise TaiyakiSigMapError(check_str)

        # create dictionary with required values and valid optional values
        readDict = dict((k, getattr(self, k))
                        for k in self.req_data_types._fields)
        readDict.update(dict(
            (k, getattr(self, k)) for k in self.opt_data_types._fields
            if getattr(self, k) is not None))
        return readDict

    def get_mapped_reference_region(self):
        """Get a the part of the reference that's mapped.

        Returns:
            tuple : (start,end_exc) where
                read_dict['Reference'][start:end_exc] is the mapped region
                of the reference"""
        # Locations in the ref that are mapped
        # note siglen indicates a valid mapping end position
        # while siglen + 1 indicates unmapped reference positions
        valid_ref_locs = np.where(np.logical_and(
            np.greater_equal(self.Ref_to_signal, 0),
            np.less_equal(self.Ref_to_signal, self.siglen)))[0]
        if len(valid_ref_locs) == 0:
            return 0, 0
        return valid_ref_locs[0], valid_ref_locs[-1]

    def get_mapped_dacs_region(self):
        """Get the part of the dacs vector that's mapped.

        Returns:
            tuple :range where self.Dacs[range[0]:range[1]] is the
                     mapped region of the signal"""
        # Locations in the DACs that are mapped
        # note siglen indicates a valid mapping end position
        # while siglen + 1 indicates unmapped reference positions
        valid_sig_locs = self.Ref_to_signal[np.logical_and(
            np.greater_equal(self.Ref_to_signal, 0),
            np.less_equal(self.Ref_to_signal, self.siglen))]
        if len(valid_sig_locs) == 0:
            return 0, 0
        return valid_sig_locs[0], valid_sig_locs[-1]

    def get_reference_locations(self, signal_location_vector):
        """Return reference locations that go with given signal locations.

        Args:
           signal_location_vector (numpy int vector): locations in the signal.

        Returns:
            numpy int vector :reference locations.

        Note:
            Feeding in a tuple works too but the result is still a vector

            If signal locations outside of the mapped region are requested an
            IndexError is raised.

            If we have a numpy-style range in a tuple
            t = (signal_start_inclusive, signal_end_exclusive)
            then f(t) is
            reference_start_inclusive, reference_end_exclusive
        """
        if isinstance(signal_location_vector, tuple):
            signal_location_vector = np.array(signal_location_vector)

        mapped_dacs_start, mapped_dacs_end = self.get_mapped_dacs_region()
        if any(signal_location_vector < mapped_dacs_start):
            raise IndexError('Signal location before mapped region requested.')
        if any(signal_location_vector > mapped_dacs_end):
            raise IndexError('Signal location after mapped region requested.')

        # searchsorted to the right for start coordinate in order to avoid
        # including slip bases. If selected dac position is within the signal
        # for a base, include that previous base. Core reason for this is that
        # the taiyaki.ctc.c_crf_flipflop.crf_flipflop_forward_step function
        # allows only stays in the first base of the chunk sequence. Subtract
        # one in base space to include this preceding base.
        seq_start = np.searchsorted(
            self.Ref_to_signal, signal_location_vector[0], 'right') - 1
        # searchsorted to the left side for the end position to avoid slip
        # bases at the end of a chunk.
        seq_end = np.searchsorted(
            self.Ref_to_signal, signal_location_vector[1], 'left')
        return np.array([seq_start, seq_end])

    def get_dacs(self, region=None):
        """Get vector of DAC levels

        Args:
            region (tuple of two ints or None):
                                region = (start_inclusive, end_exclusive)

        Returns:
            np int array : current[start_inclusive:end_exclusive].
        """
        if region is None:
            return self.Dacs
        a, b = region
        return self.Dacs[a:b]

    def get_current(self, region=None, standardize=True):
        """Get current vector and optionally apply standardization factors.

        Args:
            region (tuple of two ints or None):
                                region = (start_inclusive, end_exclusive)
            standardize (bool) : if true, then apply standardisation
                                 factors as stored in the SignalMapping object.

        Returns:
            np float array : current[start_inclusive:end_exclusive],
                              normalised if standardize=True.
        """
        dacs = self.get_dacs(region)

        current = (dacs + self.offset) * self.range / self.digitisation
        if standardize:
            current = (current - self.shift_frompA) / self.scale_frompA
        return current

    def _get_chunk(self, dacs_region, ref_region, standardize=True):
        """Get a chunk, returning a Chunk object.

        Args:
            dacs_region (tuple of two ints):
                                region = (start_inclusive, end_exclusive)
            ref_region (tuple of two ints):
                                region = (start_inclusive, end_exclusive)
            standardize (bool) : if true, then apply standardisation
                                 factors as stored in the SignalMapping object.

        Returns:
            chunk object : data from the selected region

        Note:
            There is no checking in this function that the dacs_region and
        ref_region are associated with one another. That must be done before
        calling this function.
        """
        if ref_region[1] == ref_region[0]:
            return Chunk(self.read_id, reject_reason=Chunk.rej_str_empty_seq)
        elif dacs_region[1] == dacs_region[0]:
            return Chunk(self.read_id, reject_reason=Chunk.rej_str_empty_sig)

        current = self.get_current(dacs_region, standardize)
        reference = self.Reference[ref_region[0]:ref_region[1]]
        dwells = np.diff(self.Ref_to_signal[ref_region[0]:ref_region[1]])
        # If the ref_region has length 1, then the diff has length zero
        # and the line to get maxdwell fails. So we need to check length
        if len(dwells) > 0:
            maxdwell = np.max(dwells)
        else:
            maxdwell = 1
        return Chunk(
            self.read_id, current, reference, maxdwell, dacs_region[0])

    def get_chunk_with_sample_length(
            self, chunk_len, start_sample=None, standardize=True):
        """Get a chunk, with chunk_len samples, returning a Chunk object.

        Args:
            chunk_len (int) : number of samples in the chunk
            start_sample (int or None): if None, then choose the start point
                        randomly over the possible start points that lead to
                        a chunk of the right size.
                        If start_sample is specified as an int, then use start
                        start_sample samples into the mapped region.

        Returns:
            Chunk object : signal and ref data.

        Note:

        The chunk should have length chunk_len samples, with the number of
        bases determined by the mapping.
        """
        mapped_dacs_region = self.get_mapped_dacs_region()
        spare_length = (
            mapped_dacs_region[1] - mapped_dacs_region[0] - chunk_len)
        if spare_length <= 0 or (start_sample is not None and
                                 start_sample >= spare_length):
            return Chunk(self.read_id, reject_reason=Chunk.rej_str_short)

        if start_sample is None:
            dacstart = np.random.randint(spare_length) + mapped_dacs_region[0]
        else:
            dacstart = start_sample + mapped_dacs_region[0]

        dacs_region = dacstart, chunk_len + dacstart
        try:
            ref_region = self.get_reference_locations(dacs_region)
        except IndexError:
            # this should never happen, but we don't want to halt training if
            # this is an outlier bug
            return Chunk(self.read_id, reject_reason=Chunk.rej_str_null_map)
        return self._get_chunk(dacs_region, ref_region, standardize)

    def get_chunk_with_sequence_length(
            self, chunk_bases, start_base=None, standardize=True):
        """Get a Chunk object with chunk_bases bases in the ref.

        Args:
            chunk_bases (int) : number of bases in the chunk
            start_base (int or None): if None, then choose the start point
                        randomly over the possible start points that lead to
                        a chunk of the right size.
                        If start_base is specified as an int, then use start
                        start_base bases into the mapped region.

        Returns:
            Chunk object : signal and ref data.

        Note:

        The chunk should have length chunk_bases bases, with the number of
        samples determined by the mapping.
        """
        mapped_reference_region = self.get_mapped_reference_region()
        spare_length = (mapped_reference_region[1] -
                        mapped_reference_region[0]) - chunk_bases
        # <= rather than < because we want to be able to look up the
        # end in the mapping
        if spare_length <= 0 or (start_base is not None and
                                 start_base >= spare_length):
            return Chunk(self.read_id, reject_reason=Chunk.rej_str_short)
        if start_base is None:
            refstart = (np.random.randint(spare_length) +
                        mapped_reference_region[0])
        else:
            refstart = start_base + mapped_reference_region[0]
        refend_exc = refstart + chunk_bases
        dacstart = self.Ref_to_signal[refstart]
        dacsend_exc = self.Ref_to_signal[refend_exc]

        return self._get_chunk((dacstart, dacsend_exc), (refstart, refend_exc),
                               standardize)


class Chunk(object):
    """
    Represents a chunk of a signal mapping including current, sequence,
    relevant metrics, and whether the chunk has been rejected.

    The apply_filters function checks for mean and maximum dwell metrics and
    sets the reject_reason attribute. The Chunk.valid_rej_strs define the set
    of potential rejection reasons. If a new rejection reason is added it
    should be added to this set.
    """
    # avoids overflow for 0-length chunk sequence computations
    _tiny = 0.00000001

    # Valid rejection reasons
    rej_str_pass = 'pass'
    rej_str_empty_seq = 'emptysequence'
    rej_str_empty_sig = 'emptysignal'
    rej_str_short = 'tooshort'
    rej_str_null_map = 'nullmapping'
    rej_str_mean_dwl = 'meandwell'
    rej_str_max_dwl = 'maxdwell'
    valid_rej_strs = set((
        rej_str_pass,
        rej_str_empty_seq, rej_str_empty_sig, rej_str_short, rej_str_null_map,
        rej_str_mean_dwl, rej_str_max_dwl))

    def __init__(
            self, read_id, current=None, sequence=None, max_dwell=None,
            start_sample=None, reject_reason=None):
        """
        Create a chunk object from signal data.

        Args:
            read_id (str): Read UUID
            current (np float array): Signal current array
            sequence (np int array): Integer encoded chunk sequence
            max_dwell (int): Maximum dwell found in this chunk
            start_sample (int): Index of the start of this chunk from the full
                                 read SignalMapping current.
            reject_reason (str): Reason for rejecting this chunk. Default of
                                 None indicates that the read is not rejected.

        Returns:
            Chunk object.
        """
        self.current = current
        self.sequence = sequence
        self.max_dwell = max_dwell
        self.start_sample = start_sample
        self.read_id = read_id
        self.reject_reason = (self.rej_str_pass if reject_reason is None else
                              reject_reason)
        assert self.reject_reason in self.valid_rej_strs

    @property
    def accepted(self):
        """Is chunk acceptable?
        Returns:
            bool : True if chunk not rejected"""
        return self.reject_reason == self.rej_str_pass

    @property
    def mean_dwell(self):
        """Calculate mean dwell from chunk.
        Returns:
            float : mean dwell
        """
        return len(self.current) / (len(self.sequence) + self._tiny)

    @property
    def seq_len(self):
        """Length of ref sequence in chunk.
        Returns:
            int : length of sequence (0 if not present)"""
        return len(self.sequence) if self.sequence is not None else 0

    @property
    def sig_len(self):
        """Length of signal in chunk.
        Returns:
            int : length of signal (0 if not present)"""
        return len(self.current) if self.current is not None else 0

    def apply_filters(self, filter_params):
        """Apply filtering conditions, and set rejected attribute.

        Args:
            filter_params (namedtuple as in
                           taiyaki.chunk_selection.FILTER_PARAMETERS)

        Note:
            If filter_params.median_mean_dwell or filter_params.mad_dwell are
            None or if chunk is already filtered, then don't filter.
        """
        if not self.accepted or \
           filter_params.median_meandwell is None or \
           filter_params.mad_meandwell is None:
            #  Short-circuit no filtering
            return

        mean_dwell_dev_from_median = abs(
            self.mean_dwell - filter_params.median_meandwell)
        if mean_dwell_dev_from_median > \
           filter_params.filter_mean_dwell * filter_params.mad_meandwell:
            #  Failed mean dwell filter
            self.reject_reason = self.rej_str_mean_dwl
            return

        if self.max_dwell > \
           filter_params.filter_max_dwell * filter_params.median_meandwell:
            #  Failed maximum dwell filter
            self.reject_reason = self.rej_str_max_dwl
