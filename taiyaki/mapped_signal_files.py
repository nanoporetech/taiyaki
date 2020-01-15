# Defines an abstract class used to read and write per-read "chunk" files
# and a derived class using HDF5 in the simplest way possible
# The base class provides a prototype for other file formats.
# If the class interface is fixed, we can swap to other classes
# (for example, Per_read_Fast5 or Per_read_SQLite)
#

from abc import ABC, abstractmethod
import h5py
import numpy as np
import posixpath

from taiyaki import alphabet

_version = 8

class Read(dict):
    """Class to represent the information about a read that is stored in
    a per-read file. Includes lots of checking methods, and methods
    to output chunk data.

    Much of the code in this class definition is the checking functions, which
    check that the data is consistent with the 'Chunkify version 7 discussion'
    https://wiki/display/RES/Chunkify+version+7+discussion

    range,offset and digitisation describe the mapping from Dacs to current in pA:

    current = (dacs + offset ) * range / digitisation

    scale_frompA and shift_frompA describe the mapping from current in pA to
    standardised numbers for training:

    standardised_current = ( current - shift ) / scale

    Ref_to_signal[n] is the location in Dacs corresponding to base n in Reference.

    """
    # The data to be stored, with types
    # In cases where int or float is specified, numpy types like np.int32 or np.float64 are allowed
    # for the scalar.
    # If the data type is a numpy one (e.g. np_int32) we interpret that as meaning an ndarray of that
    # dtype.
    # Also we use upper case for numpy arrays (or dataset in HDF5), lower case for scalar
    # in these dictionaries, although that is just an aid to reading and not checked in the code
    read_data = {
        'shift_frompA': 'float',
        'scale_frompA': 'float',
        'range': 'float',
        'offset': 'float',
        'digitisation': 'float',
        'Dacs': 'np_int16',
        'Ref_to_signal': 'np_int32',
        'Reference': 'np_int16'}

    optional_read_data = {
        'mapping_score': 'float',
        'mapping_method': 'str',
        'read_id': 'str'}

    def __init__(self, d):
        self.update(d)

    @staticmethod
    def _typecheck(name, x, target_type):
        """Returns empty string or error string depending on whether type matches"""
        if target_type == 'int':
            # Allow any integer type including numpy integer types
            # See
            # https://stackoverflow.com/questions/37726830/how-to-determine-if-a-number-is-any-type-of-int-core-or-numpy-signed-or-not
            if not np.issubdtype(type(x), np.integer):
                return "Type of attribute " + name + " is " + str(type(x)) + ": should be an integer type.\n"
        elif target_type == 'float':
            # Allow any float type including numpy float types
            if not np.issubdtype(type(x), np.floating):
                return "Type of attribute " + name + " is " + str(type(x)) + ": should be a float type.\n"
        elif target_type == 'bool':
            # Allow any boolean type including numpy bool type
            if not np.issubdtype(type(x), np.dtype(bool).type):
                return "Type of attribute " + name + " is " + str(type(x)) + ": should be a float type.\n"
        elif target_type == 'str':
            if not isinstance(x, str):
                return "Type of attribute " + name + " is " + str(type(x)) + ": should be a string.\n"
        elif target_type.startswith('np'):
            if type(x) != np.ndarray:
                return "Type of attribute " + name + " is not np.ndarray\n"
            target_dtype = target_type.split('_')[1]
            if str(x.dtype) != target_dtype:
                return "Data type of items in numpy array " + name + " is not " + target_dtype + "\n"
        else:
            if str(type(x)) != target_type:
                return "Type of attribute " + name + " is " + str(type(x)) + ": should be" + target_type + ".\n"
        return ""

    def check(self):
        """Return string "pass" if read info passes some integrity tests.
        Return failure information as string otherwise."""
        return_string = ""

        for k, target_type in self.read_data.items():
            try:
                x = self[k]
            except:
                return_string += "Failed to find element " + k + "\n"
                continue
            return_string += self._typecheck(k, x, target_type)

        try:
            maplen = len(self['Ref_to_signal'])
            reflen = len(self['Reference'])
            if reflen + 1 != maplen:
                return_string += "Length of Ref_to_signal (" + str(maplen) + \
                    ") should be 1+ length of Reference (" + str(reflen) + ")\n"
        except:
            return_string += "Not able to check len(Ref_to_signal)=len(Reference)\n"

        try:
            r = self['Ref_to_signal']
            s = self['Dacs']
            mapmin, mapmax = np.min(r), np.max(r)
            if mapmin < -1 or mapmax > len(s):  # -1 and len(s) are used as end markers
                return_string += "Range of locations in mapping exceeds length of Dacs\n"
        except:
            return_string += "Not able to check range of Ref_to_signal values is inside the signal vector (Dacs)\n"

        try:
            r = self['Ref_to_signal']
            if np.any(np.diff(r) < 0):
                return_string += "Mapping does not increase monotonically\n"
        except:
            return_string += "Not able to check that mapping increases monotonically\n"

        # Are there any items in the dictionary that should't be there?
        alldatakeys = set(self.read_data).union(self.optional_read_data)
        for k in self:
            if not (k in alldatakeys):
                return_string += "Data item " + k + " in read data shouldn't be there\n"

        if len(return_string) == 0:
            return "pass"
        return return_string

    def get_mapped_reference_region(self):
        """Return tuple (start,end_exc) so that
        read_dict['Reference'][start:end_exc] is the mapped region
        of the reference"""
        daclen = len(self['Dacs'])
        r = self['Ref_to_signal']
        # Locations in the ref that are mapped
        # note daclen indicates a valid mapping end position
        # while daclen + 1 indicates unmapped reference positions
        valid_ref_locs = np.where((r >= 0) & (r <= daclen))[0]
        if len(valid_ref_locs) == 0:
            return 0, 0
        return valid_ref_locs[0], valid_ref_locs[-1]

    def get_mapped_dacs_region(self):
        """Return tuple (start,end_exc) so that
        read_dict['Dacs'][start:end_exc] is the mapped region
        of the signal"""
        r = self['Ref_to_signal']
        daclen = len(self['Dacs'])
        # Locations in the DACs that are mapped
        # note daclen indicates a valid mapping end position
        # while daclen + 1 indicates unmapped reference positions
        valid_sig_locs = r[(r >= 0) & (r <= daclen)]
        if len(valid_sig_locs) == 0:
            return 0, 0
        return valid_sig_locs[0], valid_sig_locs[-1]

    def get_reference_locations(self, signal_location_vector):
        """Return reference locations that go with given signal locations.
        signal_location_vector should be a numpy integer vector of signal locations.
        The return value is a numpy integer vector of reference locations.
        (feeding in a tuple works too but the result is still a vector)

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
        result = np.searchsorted(self['Ref_to_signal'], signal_location_vector)
        if any(signal_location_vector < mapped_dacs_start):
            raise IndexError('Signal location before mapped region requested.')
        if any(signal_location_vector > mapped_dacs_end):
            raise IndexError('Signal location after mapped region requested.')

        return result


    def get_dacs(self, region=None):
        """Get vector of DAC levels
        If region is not None, then treat region as a tuple:
            region = (start_inclusive, end_exclusive)

        :returns: current[start_inclusive:end_exclusive].
        """
        if region is None:
            dacs = self['Dacs']
        else:
            a, b = region
            dacs = self['Dacs'][a:b]
        return dacs


    def get_current(self, region=None, standardize=True):
        """Get current vector and, optionally apply standardization factors.
        If region is not None, then treat region as a tuple:
            region = (start_inclusive, end_exclusive)

        :returns: current[start_inclusive:end_exclusive].
        """
        dacs = self.get_dacs(region)

        current = (dacs + self['offset']) * self['range'] / self['digitisation']
        if standardize:
            current = (current - self['shift_frompA']) / self['scale_frompA']
        return current

    def check_for_slip_at_refloc(self, refloc):
        """Return True if there is a slip at reference location refloc.
        This means that the signal location at refloc is the same as
        the signal location that goes with either the previous base
        or the next one."""
        r = self['Ref_to_signal']
        sigloc = r[refloc]
        # print("reftosig[",refloc-1,"-",refloc+1,"]=",r[refloc-1:refloc+2])
        if refloc < len(r) - 1:
            if r[refloc + 1] == sigloc:
                return True
        if refloc > 1:
            if r[refloc - 1] == sigloc:
                return True
        return False

    def _get_chunk(self, dacs_region, ref_region, standardize=True, verbose=False):
        """
        Get a chunk, returning a dictionary with entries:

           current, sequence, max_dwell, start_sample, read_id

        where current is standardised (i.e. scaled so roughly mean=0 std=1) and
        reference is a np array of ints.

        The function will return a dict containing at least keys 'read_id' and
        'rejected' (giving a reason for rejection) if either the reference region or the
        signal region is empty or if a boundary of the proposed chunk is in a slip.

        Note there is no checking in this function that the dacs_region and ref_region
        are associated with one another. That must be done before calling this function.

        If the optional data item read_id is not present in the read dictionary
        then the 'read_id' item will be missing in the dictionary returned.

        The mean dwell is len(reference_sequence) / len(standardised_current),
        so can be calculated from the data returned by this function.
        """
        if ref_region[1] == ref_region[0]:
            if verbose:
                print("Rejecting read because of zero-length sequence chunk")
            returndict = {'rejected': 'emptysequence'}
        elif dacs_region[1] == dacs_region[0]:
            if verbose:
                print("Rejecting read because of zero-length signal chunk")
            returndict = {'rejected': 'emptysignal'}
        else:
            current = self.get_current(dacs_region, standardize)
            reference = self['Reference'][ref_region[0]:ref_region[1]]
            dwells = np.diff(self['Ref_to_signal'][ref_region[0]:ref_region[1]])
            # If the ref_region has length 1, then the diff has length zero and the
            # line to get maxdwell fails. So we need to check length
            if len(dwells) > 0:
                maxdwell = np.max(dwells)
            else:
                maxdwell = 1
            returndict = {'current': current,
                          'sequence': reference,
                          'max_dwell': maxdwell,
                          'start_sample': dacs_region[0]}
            if self.check_for_slip_at_refloc(ref_region[0]) or self.check_for_slip_at_refloc(ref_region[1]):
                if verbose:
                    print("Rejecting read because of slip:", self.check_for_slip_at_refloc(
                        ref_region[0]), self.check_for_slip_at_refloc(ref_region[1]))
                returndict['rejected'] = 'slip'

        if 'read_id' in self:
            returndict['read_id'] = self['read_id']
        return returndict

    def get_chunk_with_sample_length(self, chunk_len, start_sample=None, standardize=True, verbose=False):
        """
        Get a chunk, with chunk_len samples, returning a dictionary as in the docstring for get_chunk()

        If start_sample is None, then choose the start point randomly over the possible start points
        that lead to a chunk of the right size.
        If start_sample is specified as an int, then use a start  point start_sample samples into
        the mapped region.

        The chunk should have length chunk_len samples, with the number of bases determined by the mapping.
        """
        mapped_dacs_region = self.get_mapped_dacs_region()
        spare_length = mapped_dacs_region[1] - mapped_dacs_region[0] - chunk_len
        if spare_length <= 0:
            if verbose:
                print("Rejecting read because spare_length=", spare_length,
                      ". mapped_dacs_region = ", mapped_dacs_region)
            return {'rejected':'tooshort','read_id':self.get('read_id')}

        if start_sample is None:
            dacstart = np.random.randint(spare_length) + mapped_dacs_region[0]
        else:
            if start_sample >= spare_length:
                if verbose:
                    print("Rejecting read because start_sample >= spare_length=", spare_length)
                return {'rejected':'tooshort','read_id':self.get('read_id')}
            dacstart = start_sample + mapped_dacs_region[0]

        dacs_region = dacstart, chunk_len + dacstart
        try:
            ref_region = self.get_reference_locations(dacs_region)
        except IndexError:
            # this should never happen, but we don't want to halt training if
            # this is an outlier bug
            return {'rejected':'nullmapping', 'read_id':self.get('read_id')}
        return self._get_chunk(dacs_region, ref_region, standardize, verbose)

    def get_chunk_with_sequence_length(self, chunk_bases, start_base=None, standardize=True):
        """Get a chunk containing a sequence of length chunk_bases,
        returning a dictionary as in the docstring for get_chunk()

        If start_base is None, then choose the start point randomly over the possible start points
        that lead to a chunk of the right size.
        If start_base is specified as an int, then use a start point start_base bases into
        the mapped region.

        The chunk should have length chunk_bases bases, with the number of samples determined by the mapping.
        """
        mapped_reference_region = self.get_mapped_reference_region()
        spare_length = (mapped_reference_region[1] - mapped_reference_region[0]) - chunk_bases
        if spare_length <= 0: #<= rather than < because we want to be able to look up the end in the mapping
            return {'rejected':'tooshort','read_id':self.get('read_id')}
        if start_base is None:
            refstart = np.random.randint(spare_length) + mapped_reference_region[0]
        else:
            if start_base >= spare_length:
                return {'rejected':'tooshort','read_id':self.get('read_id')}
            refstart = start_base + mapped_reference_region[0]
        refend_exc = refstart + chunk_bases
        dacstart = self['Ref_to_signal'][refstart]
        dacsend_exc = self['Ref_to_signal'][refend_exc]
        #print("get_chunk_with_sequence_length(): ref region",refstart,refend_exc)
        #print("                                  sig region",dacstart,dacsend_exc)

        return self._get_chunk((dacstart, dacsend_exc), (refstart, refend_exc), standardize=standardize)


class AbstractMappedSignalReader(ABC):
    """Abstract base class for files containing mapped reads.
    Methods specified as abstractnethod must be overridden
    in derived classes.
    Note that the methods to check reads and to check the metadata
    are not abstract and should not be overridden.

    The class has __enter__ and __exit__ so can be used as a context
    manager (i.e. with the 'with' statement).

    In all derived classes, the input and output from the file is done with
    the read_dict defined above.

    Derived classes should use the read class variables read_data and
    optional_read_data
    as much as possible so that changes made there will be propagated to the
    derived classes.
    """

    def __enter__(self):
        """Called when 'with' is used to create an object.
        Since we always return the instance, no need to override this."""
        return self

    def __exit__(self, *args):
        """No need to override this - just override the close() function.
        Called when 'with' finishes."""
        self.close()

    #########################################
    # Abstract methods in alphabetical order
    #########################################

    @abstractmethod
    def __init__(self, filename, mode="a"):
        """Open file in read-only mode (mode="r") or allowing
        writing of additional stuff and creating if empty
        (default mode "a") """
        pass

    @abstractmethod
    def close(self):
        """Close file"""
        pass

    @abstractmethod
    def get_read(self, read_id):
        """Return a read object containing all elements of the read."""
        pass

    @abstractmethod
    def get_read_ids(self):
        """Return list of read ids, or empty list if none present"""
        pass

    @property
    @abstractmethod
    def version(self):
        """Return integer version number"""
        pass

    @abstractmethod
    def get_alphabet_information(self):
        """Return taiyaki.alphabet.AlphabetInfo object"""
        pass

    # This function is not abstract because it can be left as-is.
    # But it may be overridden if there are speed gains to be had
    def get_multiple_reads(self, read_id_list, return_list=True, max_reads=None):
        """Get dictionary where keys are read ids from the list
        and values are the read objects. If read_id_list=="all" then get
        them all.
        If return_list, then return a list of read objects where the read_ids
        are incorporated in the dicts.
        If not, then a dict of dicts where the keys are the read_ids.
        If a read_id in the list is not present in the file, then just skip.
        Don't raise an exception."""
        read_ids_in_file = self.get_read_ids()
        if read_id_list == "all":
            read_ids_used = read_ids_in_file
        else:
            read_ids_used = set(read_id_list).intersection(read_ids_in_file)
        if max_reads is not None and max_reads < len(read_ids_used):
            read_ids_used = list(read_ids_used)[:max_reads]
        if return_list:
            # Make a new read object containing the read_id as well as other items, for each read id in the list
            return [Read({**self.get_read(read_id), 'read_id': read_id}) for read_id in read_ids_used]
        else:
            return {read_id: self.get_read(read_id) for read_id in read_ids_used}

    def check_read(self, read_id):
        """Check a read in the currently open file, returning "pass" or a report
        on errors."""
        try:
            read = self.get_read(read_id)
        except:
            return "Unable to get read " + read_id + " from file"
        return read.check()

    def check(self, limit_report_lines=100):
        """Check the whole file, returning report in the form of a string"""
        return_string = ""
        try:
            version_number = self.version
            return_string += Read._typecheck('version', version_number, 'int')
        except:
            return_string += "Can't get version number\n"

        read_ids = self.get_read_ids()
        if len(read_ids) == 0:
            return_string += "No reads in file\n"
        for read_id in read_ids:
            if return_string.count('\n') >= limit_report_lines:
                return_string += "----------Number of lines in error report limited to " + \
                    str(limit_report_lines) + "\n"
            else:
                read_check = self.check_read(read_id)
                if read_check != "pass":
                    return_string += "Read " + read_id + ":\n" + read_check
        if len(return_string) == 0:
            return "pass"
        else:
            return return_string


class AbstractMappedSignalWriter(ABC):
    """Abstract base class for writing files containing mapped reads.
    Methods specified as abstractmethod must be overridden
    in derived classes.
    Note that the methods to check reads and to check the metadata
    are not abstract and should not be overridden.

    The class has __enter__ and __exit__ so can be used as a context
    manager (i.e. with the 'with' statement).

    In all derived classes, the input and output from the file is done with
    the read_dict defined above.

    Derived classes should use the read class variables read_data and
    optional_read_data
    as much as possible so that changes made there will be propagated to the
    derived classes.
    """

    def __enter__(self):
        """Called when 'with' is used to create an object.
        Since we always return the instance, no need to override this."""
        return self

    def __exit__(self, *args):
        """No need to override this - just override the close() function.
        Called when 'with' finishes."""
        self.close()

    @abstractmethod
    def write_read(self, readdict):
        """Write a read to the appropriate place in the file, starting from a read dictionary"""
        pass

    @abstractmethod
    def _write_version(self):
        """Write version number of file format"""
        pass

    @abstractmethod
    def _write_alphabet_info(self, alphabet_info):
        """Write alphabet information to file"""
        pass


class HDF5Reader(AbstractMappedSignalReader):
    """A file storing mapped data in an HDF5 in the simplest
    possible way.
    NOT using a derivative of the fast5 format.
    This is an HDF5 file with structure below
    There can be as many read_ids as you like.
    version is an attr, and the read data are stored
    as Datasets or attributes as appropriate.

      file--|---Reads ----------|--<read_id_0>-- {
            \                   |                {(All the read data for read 0)
             version            |
             alphabet           |--<read_id_>--  {
             collapse_alphabet  |                {(All the read data for read 1)
             mod_long_names     |

    """

    def __init__(self, filename):
        self.hdf5 = h5py.File(filename, 'r')
        assert self.version == _version, 'Incorrect file version, got {} expected {}'.format(self.version, _version)

    def close(self):
        self.hdf5.close()

    def _get_read_path(self, read_id):
        """Returns string giving path within HDF5 file to data for a read"""
        return 'Reads/' + read_id

    def get_read(self, read_id):
        """Return a read object (see class definition above)."""
        h = self.hdf5[self._get_read_path(read_id)]
        d = {}
        for k, v in h.items():  # Iterate over datasets (the read group should have no subgroups)
            d[k] = v[()]
        for k, v in h.attrs.items():  # iterate over attributes
            d[k] = v
        return Read(d)

    def get_read_ids(self):
        """Return list of read ids, or empty list if none present"""
        try:
            return self.hdf5['read_ids'][()].tolist()
        except KeyError:
            pass
        try:
            return list(self.hdf5['Reads'].keys())
        except:
            return []

    @property
    def version(self):
        return self.hdf5.attrs['version']

    def get_alphabet_information(self):
        mod_long_names = self.hdf5.attrs['mod_long_names'].splitlines()
        return alphabet.AlphabetInfo(
            self.hdf5.attrs['alphabet'], self.hdf5.attrs['collapse_alphabet'],
            mod_long_names)


class HDF5Writer(AbstractMappedSignalWriter):
    """A file storing mapped data in an HDF5 in the simplest
    possible way.
    NOT using a derivative of the fast5 format.
    This is an HDF5 file with structure below
    There can be as many read_ids as you like.
    version is an attr, and the read data are stored
    as Datasets or attributes as appropriate.

      file--|---Reads ----------|--<read_id_0>-- {
            \                   |                {(All the read data for read 0)
             version            |
             alphabet           |--<read_id_>--  {
             collapse_alphabet  |                {(All the read data for read 1)
             mod_long_names     |

    """
    def __init__(self, filename, alphabet_info):
        # mode 'w' to preserve behaviour, 'x' would be more appropraite
        self.hdf5 = h5py.File(filename, 'w')
        self._write_version()
        self._write_alphabet_info(alphabet_info)
        # collect read_ids for storage since listing HDF5 keys can be very slow
        # for large numbers of groups. These will be dumped to a dataset when
        # the file is closed.
        self.read_ids = []

    def close(self):
        if len(self.read_ids) > 0:
            # special variable length string h5py data type
            dt = h5py.special_dtype(vlen=str)
            read_ids = np.array(self.read_ids, dtype=dt)
            # store read ids dataset in root so it isn't confused for a read
            read_ids_ds = self.hdf5.create_dataset(
                'read_ids', read_ids.shape, dtype=dt, compression="gzip")
            read_ids_ds[...] = read_ids

        self.hdf5.close()

    def write_read(self, readdict):
        """Write a read to the appropriate place in the file, starting from a read object"""
        read = Read(readdict)
        read_id = readdict['read_id']
        self.read_ids.append(read_id)
        g = self.hdf5.create_group(posixpath.join('Reads', read_id))
        for k, v in read.items():
            if isinstance(v, np.ndarray):
                g.create_dataset(k, data=v, compression='gzip', shuffle=True)
            else:
                g.attrs[k] = v

    def _write_version(self):
        self.hdf5.attrs['version'] = _version

    def _write_alphabet_info(self, alphabet_info):
        self.hdf5.attrs['alphabet'] = alphabet_info.alphabet
        self.hdf5.attrs['collapse_alphabet'] = alphabet_info.collapse_alphabet
        self.hdf5.attrs['mod_long_names'] = '\n'.join(alphabet_info.mod_long_names)
