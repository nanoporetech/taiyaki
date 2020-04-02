# Defines an abstract class used to read and write per-read "chunk" files
# and a derived class using HDF5 in the simplest way possible
# The base class provides a prototype for other file formats.
# If the class interface is fixed, we can swap to other classes
# (for example, Per_read_Fast5 or Per_read_SQLite)

from abc import ABC, abstractmethod
import h5py
import numpy as np
import posixpath

from taiyaki import alphabet, signal_mapping

_version = 8
READS_ROOT_TEXT = 'Reads'


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

    def get_multiple_reads(self, read_id_list, max_reads=None):
        """Get list of signal_mapping.SignalMapping objects from file.
        If read_id_list=="all" then get them all.
        If a read_id in the list is not present in the file, then just skip.
        Don't raise an exception."""
        read_ids_in_file = self.get_read_ids()
        if read_id_list == "all":
            read_ids_used = read_ids_in_file
        else:
            read_ids_used = set(read_id_list).intersection(read_ids_in_file)
        if max_reads is not None and max_reads < len(read_ids_used):
            read_ids_used = list(read_ids_used)[:max_reads]
        return [self.get_read(read_id) for read_id in read_ids_used]

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
        except:
            return_string += "Can't get version number\n"
        if not np.issubdtype(type(version_number), np.integer):
            return_string += (
                'Type of attribute "version" is "{}" and should be ' +
                '"{}".\n').format(type(version_number), int)

        read_ids = self.get_read_ids()
        if len(read_ids) == 0:
            return_string += "No reads in file\n"
        for read_id in read_ids:
            if return_string.count('\n') >= limit_report_lines:
                return_string += (
                    "----------Number of lines in error report limited to " +
                    str(limit_report_lines) + "\n")
                break
            else:
                read_check = self.check_read(read_id)
                if read_check != "pass":
                    return_string += "Read " + read_id + ":\n" + read_check
        if len(return_string) == 0:
            return "pass"
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
        """Write a read to the appropriate place in the file, starting from
        a read dictionary
        """
        pass

    @abstractmethod
    def _write_version(self):
        """Write version number of file format"""
        pass

    @abstractmethod
    def _write_alphabet_info(self, alphabet_info):
        """Write alphabet information to file"""
        pass

def _get_hdf5_read_path(read_id):
    """Returns string giving path within HDF5 file to data for a read"""
    return posixpath.join(READS_ROOT_TEXT, read_id)

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
        assert self.version == _version, (
            'Incorrect file version, got {} expected {}').format(
                self.version, _version)

    def close(self):
        self.hdf5.close()

    def get_read(self, read_id):
        """Return a read object (see class definition above)."""
        h = self.hdf5[_get_hdf5_read_path(read_id)]
        d = {}
        # Iterate over datasets (the read group should have no subgroups)
        for k, v in h.items():
            d[k] = v[()]
        # iterate over attributes
        for k, v in h.attrs.items():
            d[k] = v
        return signal_mapping.SignalMapping(**d)

    def get_read_ids(self):
        """Return list of read ids, or empty list if none present"""
        try:
            return self.hdf5['read_ids'][()].tolist()
        except KeyError:
            pass
        try:
            return list(self.hdf5[READS_ROOT_TEXT].keys())
        except:
            return []

    def get_alphabet_information(self):
        mod_long_names = self.hdf5.attrs['mod_long_names'].splitlines()
        return alphabet.AlphabetInfo(
            self.hdf5.attrs['alphabet'], self.hdf5.attrs['collapse_alphabet'],
            mod_long_names)

    @property
    def version(self):
        return self.hdf5.attrs['version']

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
        """Write a read to the appropriate place in the file, starting from
        a read object
        """
        read_id = readdict['read_id']
        self.read_ids.append(read_id)
        g = self.hdf5.create_group(_get_hdf5_read_path(read_id))
        for k, v in readdict.items():
            if isinstance(v, np.ndarray):
                g.create_dataset(k, data=v, compression='gzip', shuffle=True)
            else:
                g.attrs[k] = v

    def _write_alphabet_info(self, alphabet_info):
        self.hdf5.attrs['alphabet'] = alphabet_info.alphabet
        self.hdf5.attrs['collapse_alphabet'] = alphabet_info.collapse_alphabet
        self.hdf5.attrs['mod_long_names'] = '\n'.join(
            alphabet_info.mod_long_names)

    def _write_version(self):
        self.hdf5.attrs['version'] = _version
