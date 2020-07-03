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
        """ Called when 'with' is used to create an object.
        Since we always return the instance, no need to override this.
        """
        return self

    def __exit__(self, *args):
        """ No need to override this - just override the close() function.
        Called when 'with' finishes.
        """
        self.close()

    #########################################
    # Abstract methods in alphabetical order
    #########################################

    @abstractmethod
    def __init__(self, filename, mode="a"):
        """ Open file

        Note:
            File open `mode` is assumed to be consistent with Python `File`
        objects, but implementation is defined by the derived class.

        Args:
            filename (str): name of file to open.
            mode (str, optional): mode to open file.  Default "a" is to append,
                creating file if necessary.
        """
        pass

    @abstractmethod
    def close(self):
        """ Close file
        """
        pass

    @abstractmethod
    def get_read(self, read_id):
        """ Return a read object containing all elements of the read.

        Args:
            read_id (str): ID of read to get from object.

        Returns:
            :class:`signal_mapping.SignalMapping`: information about read
                mapping.
        """
        pass

    @abstractmethod
    def get_read_ids(self):
        """ Return read IDs present in file

        Returns:
            list of str: List of read ids present in file, empty list if no
                reads are present.
        """
        pass

    @property
    @abstractmethod
    def version(self):
        """ Return integer version number

        Returns:
            int: version number of file
        """
        pass

    @abstractmethod
    def get_alphabet_information(self):
        """  Get information about alphabet of mapping

        Modified base information is returned if present.

        Returns:
            :class:`alphabet.AlphabetInfo`
        """
        pass

    def get_multiple_reads(self, read_id_list, max_reads=None):
        """ Get signal_mapping.SignalMapping objects from file.

        Args:
            read_id_list (list of str):  read IDs for which mapping information
                should be returned.  If `read_id_list` == "all" then all reads
                are returned.

        Returns:
            list of :class:`signal_mapping.SignalMapping`: mapping information
                for reads requested in `read_id_list,`, if present in file.
                Missing reads are skipped.
        """
        read_ids_in_file = self.get_read_ids()
        if read_id_list == "all":
            read_ids_used = read_ids_in_file
        else:
            read_ids_used = set(read_id_list).intersection(read_ids_in_file)

        if max_reads is not None and max_reads < len(read_ids_used):
            read_ids_used = list(read_ids_used)[:max_reads]

        return [self.get_read(read_id) for read_id in read_ids_used]

    def check_read(self, read_id):
        """Check a read in the currently open file.

        Args:
            read_id (str): ID of read to check

        Returns:
            str: "pass" if correct, or a report on errors.
        """
        try:
            read = self.get_read(read_id)
        except:
            return "Unable to get read " + read_id + " from file"
        return read.check()

    def check(self, limit_report_lines=100):
        """Check the whole file

        Args:
            limit_report_lines (int, optional): maximum number of lines in error
                report.

        Returns:
            str: "pass" or report of any errors
        """
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
        """ Called when 'with' is used to create an object.
        Since we always return the instance, no need to override this.
        """
        return self

    def __exit__(self, *args):
        """ No need to override this - just override the close() function.
        Called when 'with' finishes.
        """
        self.close()

    @abstractmethod
    def write_read(self, readdict):
        """ Write a read to the appropriate place in the file, starting from
        a read dictionary

        Args:
            readdict (dict):
        """
        pass

    @abstractmethod
    def _write_version(self):
        """Write version number of file format
        """
        pass

    @abstractmethod
    def _write_alphabet_info(self, alphabet_info):
        """ Write alphabet information to file

        Args:
            alphabet_info (:class:`alphabet.AlphabetInfo`):  Alphabet to write

        """
        pass


def _get_hdf5_read_path(read_id):
    """ Returns string giving path within HDF5 file to data for a read

    Args:
        read_id (str): ID of read for which path is required.

    Returns:
        str: path of read in POSIX format.
    """
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

    Attributes:
        hdf5 (:class:`h5py.File`): File handle of HDF5 file
        version (int): Version of file
    """
    def __init__(self, filename):
        """ Open file and initialise

        Args:
            filename (str): name of file to open.
        """
        self.hdf5 = h5py.File(filename, 'r')
        assert self.version == _version, (
            'Incorrect file version, got {} expected {}').format(
                self.version, _version)

    def close(self):
        """ Close file handle.
        """
        self.hdf5.close()

    def get_read(self, read_id):
        """ Return a read object containing all elements of the read.

        Args:
            read_id (str): ID of read to get from object.

        Returns:
            :class:`signal_mapping.SignalMapping`: information about read
                mapping.
        """
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
        """ Return read IDs present in file

        Returns:
            list of str: List of read ids present in file, empty list if no
                reads are present.
        """
        try:
            return self.hdf5['read_ids'][()].tolist()
        except KeyError:
            pass
        try:
            return list(self.hdf5[READS_ROOT_TEXT].keys())
        except:
            return []

    def get_alphabet_information(self):
        """  Get information about alphabet of mapping

        Modified base information is returned if present.

        Returns:
            :class:`alphabet.AlphabetInfo`
        """
        mod_long_names = self.hdf5.attrs['mod_long_names'].splitlines()
        return alphabet.AlphabetInfo(
            self.hdf5.attrs['alphabet'], self.hdf5.attrs['collapse_alphabet'],
            mod_long_names)

    @property
    def version(self):
        """ Return integer version number

        Returns:
            int: version number of file
        """
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

    Attributes:
       hdf5 (:class:`h5py.File`):  File handle of HDF5 file to write to
       read_ids (list): Read ID written to file.
    """
    def __init__(self, filename, alphabet_info):
        """ Open file and initialise

        Args:
            filename (str): name of file to open.
            alphabet_info (:class:`alphabet.AlphabetInfo`):  Alphabet to write
        """
        # mode 'w' to preserve behaviour, 'x' would be more appropraite
        self.hdf5 = h5py.File(filename, 'w')
        self._write_version()
        self._write_alphabet_info(alphabet_info)
        # collect read_ids for storage since listing HDF5 keys can be very slow
        # for large numbers of groups. These will be dumped to a dataset when
        # the file is closed.
        self.read_ids = []

    def close(self):
        """ Finalise HDF5 file and close
        """
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

        Args:
            readdict (dict): information about read
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
        """  Write information about alphabet to file

        Args:
            alphabet_info (:class:`alphabet.AlphabetInfo`):  Alphabet to write
        """
        self.hdf5.attrs['alphabet'] = alphabet_info.alphabet
        self.hdf5.attrs['collapse_alphabet'] = alphabet_info.collapse_alphabet
        self.hdf5.attrs['mod_long_names'] = '\n'.join(
            alphabet_info.mod_long_names)

    def _write_version(self):
        """  Write file version
        """
        self.hdf5.attrs['version'] = _version
