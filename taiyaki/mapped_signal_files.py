# Defines an abstract class used to read and write per-read "chunk" files
# and a derived class using HDF5 in the simplest way possible
# The base class provides a prototype for other file formats.
# If the class interface is fixed, we can swap to other classes
# (for example, Per_read_Fast5 or Per_read_SQLite)

from abc import ABC, abstractmethod
from collections import defaultdict
import h5py
import numpy as np
import posixpath
import sys

from taiyaki.alphabet import AlphabetInfo
from taiyaki.signal_mapping import SignalMapping


_version = 8
READS_ROOT_TEXT = 'Reads'
BATCH_ROOT_TEXT = 'Batches'
BATCH_TMPLT = 'Batch_{}'
BATCH_LENGTH_SUFFIX = '_lengths'
VAR_LEN_STR_DT = h5py.special_dtype(vlen=str)


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
    pass_str = 'pass'

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
    def __iter__(self):
        """ Iterate over :class:`signal_mapping.SignalMapping` extracted from
        file.
        """
        pass

    @abstractmethod
    def __next__(self):
        """ Return next :class:`signal_mapping.SignalMapping` extracted from
        file.
        """
        pass

    @abstractmethod
    def _some_reads(self, read_ids):
        """ Generator returning read object containing all elements of the read.

        Args:
            read_ids (list of str): Read IDs for which mapping information
                should be returned.

        Yields:
            :class:`signal_mapping.SignalMapping`: information about read
                mapping.
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

    @abstractmethod
    def get_alphabet_information(self):
        """ Get information about alphabet of mapping

        Modified base information is returned if present.

        Returns:
            :class:`alphabet.AlphabetInfo`
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

    def reads(self, read_ids=None):
        """ Generator returning read object containing all elements of the read.

        Args:
            read_ids (list of str): Read IDs specifying which reads to yield.
                None will yield all reads.

        Yields:
            :class:`signal_mapping.SignalMapping`: information about read
                signal mapping.
        """
        if read_ids is None:
            yield from self
        else:
            yield from self._some_reads(read_ids)

    def check(self, limit_report_lines=100):
        """Check the whole file

        Args:
            limit_report_lines (int): maximum number of lines in error report
              (default 100)

        Returns:
            str: taiyaki.mapped_signal_file.AbstractMappedSignalReader.pass_str
                or report of any errors
        """
        return_string = ""
        try:
            version_number = self.version
        except Exception:
            return_string += "Can't get version number\n"

        if not np.issubdtype(type(version_number), np.integer):
            return_string += (
                'Type of attribute "version" is "{}" and should be ' +
                '"{}".\n').format(type(version_number), int)

        file_is_empty = True
        for read in self:
            file_is_empty = False
            if return_string.count('\n') >= limit_report_lines:
                return_string += (
                    "----------Number of lines in error report limited to " +
                    str(limit_report_lines) + "\n")
                break
            read_check = read.check()
            if read_check != SignalMapping.pass_str:
                return_string += "Read " + read.read_id + ":\n" + read_check

        if file_is_empty:
            return_string += "No reads in file\n"
        if len(return_string) == 0:
            return self.pass_str

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
            readdict (dict): information about read
        """
        pass

    @abstractmethod
    def _write_alphabet_info(self, alphabet_info):
        """ Write alphabet information to file

        Args:
            alphabet_info (:class:`alphabet.AlphabetInfo`):  Alphabet to write
        """
        pass

    @abstractmethod
    def _write_version(self):
        """ Write version number of file format
        """
        pass


class PerReadHDF5Reader(AbstractMappedSignalReader):
    """A file storing mapped data in an HDF5 in the simplest
    possible way.

    NOT using a derivative of the fast5 format.
    This is an HDF5 file with structure below
    There can be as many read_ids as you like.
    version is an attr, and the read data are stored
    as Datasets or attributes as appropriate.

    Global level attributes are version, alphabet, collapse_alphabet, and
    mod_long_names. A single global "Reads" group contains a group for each
    signal-mapped read

    Attributes:
        hdf5 (:class:`h5py.File`): File handle of HDF5 file
        version (int): Version of file
    """

    def __init__(self, filename, load_in_mem=False):
        """ Open file and initialise

        Args:
            filename (str): name of file to open.
        """
        self.hdf5 = h5py.File(filename, 'r', libver='v108',
                              driver='core' if load_in_mem else None)
        assert self.version == _version, (
            'Incorrect file version, got {} expected {}').format(
                self.version, _version)

    def __iter__(self):
        self.reads_iter = iter(self.hdf5[READS_ROOT_TEXT].values())
        return self

    def __next__(self):
        return self._get_read(next(self.reads_iter))

    def _some_reads(self, read_ids):
        read_ids = self.get_read_ids() if read_ids is None else \
            set(read_ids).intersection(self.get_read_ids())
        for read_id in read_ids:
            yield self.get_read(read_id)

    def get_alphabet_information(self):
        mod_long_names = self.hdf5.attrs['mod_long_names'].splitlines()
        return AlphabetInfo(
            self.hdf5.attrs['alphabet'], self.hdf5.attrs['collapse_alphabet'],
            mod_long_names)

    @property
    def version(self):
        return self.hdf5.attrs['version']

    def close(self):
        self.hdf5.close()

    def get_read(self, read_id):
        return self._get_read(self.hdf5[
            PerReadHDF5Writer._get_hdf5_read_path(read_id)])

    def _get_read(self, h5obj):
        """ Return a read object containing all elements of the read.

        Args:
            read_id (:class:`h5py.Group`): HDF5 group for read

        Returns:
            :class:`signal_mapping.SignalMapping`: information about read
                mapping.
        """
        d = {}
        # Iterate over datasets (the read group should have no subgroups)
        for k, v in h5obj.items():
            d[k] = v[()]
        # iterate over attributes
        for k, v in h5obj.attrs.items():
            d[k] = v
        return SignalMapping(**d)

    def get_read_ids(self):
        try:
            return self.hdf5['read_ids'][()].tolist()
        except KeyError:
            pass
        try:
            return list(self.hdf5[READS_ROOT_TEXT].keys())
        except Exception:
            return []


class PerReadHDF5Writer(AbstractMappedSignalWriter):
    """A file storing mapped data in an HDF5 in the simplest
    possible way.
    NOT using a derivative of the fast5 format.
    This is an HDF5 file with structure below
    There can be as many read_ids as you like.
    version is an attr, and the read data are stored
    as Datasets or attributes as appropriate.

    Global level attributes are version, alphabet, collapse_alphabet, and
    mod_long_names. A single global "Reads" group contains a group for each
    signal-mapped read

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
        self.hdf5 = h5py.File(filename, 'w', libver='v108', track_order=True)
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
            read_ids = np.array(self.read_ids, dtype=VAR_LEN_STR_DT)
            # store read ids dataset in root so it isn't confused for a read
            read_ids_ds = self.hdf5.create_dataset(
                'read_ids', read_ids.shape, dtype=VAR_LEN_STR_DT,
                compression="gzip")
            read_ids_ds[...] = read_ids

        self.hdf5.close()

    def write_read(self, readdict):
        read_id = readdict['read_id']
        self.read_ids.append(read_id)
        g = self.hdf5.create_group(
            PerReadHDF5Writer._get_hdf5_read_path(read_id))
        for k, v in readdict.items():
            if isinstance(v, np.ndarray):
                g.create_dataset(k, data=v, compression='gzip', shuffle=True)
            else:
                g.attrs[k] = v

    @staticmethod
    def _get_hdf5_read_path(read_id):
        """ Returns string giving path within HDF5 file to data for a read

        Args:
            read_id (str): ID of read for which path is required.

        Returns:
            str: path of read in POSIX format.
        """
        return posixpath.join(READS_ROOT_TEXT, read_id)

    def _write_alphabet_info(self, alphabet_info):
        self.hdf5.attrs['alphabet'] = alphabet_info.alphabet
        self.hdf5.attrs['collapse_alphabet'] = alphabet_info.collapse_alphabet
        self.hdf5.attrs['mod_long_names'] = '\n'.join(
            alphabet_info.mod_long_names)

    def _write_version(self):
        self.hdf5.attrs['version'] = _version


class BatchHDF5Reader(AbstractMappedSignalReader):
    """ A file storing mapped data in an HDF5 using batched arrays to improve
    access efficiency.

    NOT using a derivative of the fast5 format.
    This is an HDF5 file with structure below.
    There can be as many read_ids as you like.
    version is an attr, and the read data batches are stored
    as Datasets or attributes as appropriate.

    Global level attributes are version, alphabet, collapse_alphabet, and
    mod_long_names. A single global "Batches" group contains a group for each
    batch of mapped reads.

    Attributes:
        hdf5 (:class:`h5py.File`): File handle of HDF5 file
        version (int): Version of file
        batch_names (list): All batch names in this file
        read_id_to_batch_str (dict): mapping from read ID string to batch name
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
        self._load_read_ids()
        self.batch_names = list(self.hdf5[BATCH_ROOT_TEXT].keys())

    def __iter__(self):
        self.batch_iter = iter(self.batch_names)
        self.curr_batch = iter(self._load_reads_batch(
            next(self.batch_iter)).values())
        return self

    def __next__(self):
        try:
            return next(self.curr_batch)
        except StopIteration:
            # if curr_batch is exhausted, get next batch. If batch_iter is
            # exhausted complete iterator
            self.curr_batch = self._load_reads_batch(
                next(self.batch_iter)).values()
            return next(self.curr_batch)

    def _some_reads(self, read_ids):
        read_ids = set(read_ids).intersection(self.get_read_ids())
        # group read ids by batch for more efficient access
        batches_read_ids = defaultdict(list)
        for read_id in read_ids:
            batches_read_ids[self.read_id_to_batch_str[read_id]].append(
                read_id)

        # iterate over batches and select reads from each batch
        for batch_name, batch_read_ids in batches_read_ids.items():
            reads_batch = self._load_reads_batch(batch_name)
            for read_id in batch_read_ids:
                yield reads_batch[read_id]

    def close(self):
        self.hdf5.close()

    def _load_read_ids(self):
        """ Populate the `read_id_to_batch_str` dictionary with a mapping from
        read id to batch name string.
        """
        self.read_id_to_batch_str = {}
        for batch_name, reads_batch in self.hdf5[BATCH_ROOT_TEXT].items():
            for read_id in reads_batch['read_id'][()]:
                self.read_id_to_batch_str[read_id] = batch_name

    def _load_reads_batch(self, batch_name):
        """ Load and parse a batch of reads from file.

        Returns:
            Dictionary with read id keys and
                :class:`signal_mapping.SignalMapping` values.
        """
        if batch_name not in self.batch_names:
            raise RuntimeError('Invalid batch name requested: {}'.format(
                batch_name))

        # extract reads batch group from file
        reads_batch = self.hdf5['{}/{}'.format(BATCH_ROOT_TEXT, batch_name)]
        batch_keys = list(reads_batch.keys())
        # determine which keys hold information and which hold split points
        non_len_keys = [bk for bk in batch_keys
                        if not bk.endswith(BATCH_LENGTH_SUFFIX)]
        # load read information and split into read values
        batch_ds = []
        for k in non_len_keys:
            # read data into memory
            val = reads_batch[k][()]
            # determine if data type requires splitting
            k_type = (getattr(SignalMapping.req_data_types, k)
                      if k in SignalMapping.req_data_types._fields else
                      getattr(SignalMapping.opt_data_types, k))
            if k_type.__module__ == 'numpy':
                val = np.split(
                    val, np.cumsum(reads_batch[k + BATCH_LENGTH_SUFFIX][:-1]))
            batch_ds.append(val)

        # parse raw split datasets into signal mapping objects
        parsed_reads_batch = {}
        for read_ds in zip(*batch_ds):
            readdict = dict(zip(non_len_keys, read_ds))
            parsed_reads_batch[
                readdict['read_id']] = SignalMapping(**readdict)

        return parsed_reads_batch

    def get_read(self, read_id):
        return self._load_reads_batch(
            self.read_id_to_batch_str[read_id])[read_id]

    def get_read_ids(self):
        return list(self.read_id_to_batch_str.keys())

    def get_alphabet_information(self):
        mod_long_names = self.hdf5.attrs['mod_long_names'].splitlines()
        return AlphabetInfo(
            self.hdf5.attrs['alphabet'], self.hdf5.attrs['collapse_alphabet'],
            mod_long_names)

    @property
    def version(self):
        return self.hdf5.attrs['version']


class BatchHDF5Writer(AbstractMappedSignalWriter):
    """A file storing mapped data in an HDF5 using batched arrays to improve
    access efficiency.
    NOT using a derivative of the fast5 format.
    This is an HDF5 file with structure below.
    There can be as many read_ids as you like.
    version is an attr, and the read data are stored
    as Datasets or attributes as appropriate.

    Global level attributes are version, alphabet, collapse_alphabet, and
    mod_long_names. A single global "Batches" group contains a group for each
    batch of mapped reads.

    Attributes:
       hdf5 (:class:`h5py.File`):  File handle of HDF5 file to write to
       read_ids (list): Read ID written to file.
       batch_size (int): Size of the batches to write.
    """

    def __init__(self, filename, alphabet_info, batch_size=25000):
        # mode 'w' to preserve behaviour, 'x' would be more appropraite
        self.hdf5 = h5py.File(filename, 'w')
        self._write_version()
        self._write_alphabet_info(alphabet_info)
        # collect read_ids for storage since listing HDF5 keys can be very slow
        # for large numbers of groups. These will be dumped to a dataset when
        # the file is closed.
        self.read_ids = []
        self.batch_size = batch_size
        self._curr_batch = []
        self._curr_batch_idx = 0

    def write_curr_batch(self):
        g = self.hdf5.create_group(BATCH_ROOT_TEXT + '/' +
                                   BATCH_TMPLT.format(self._curr_batch_idx))
        batch_keys_set = set(tuple(sorted(rd.keys()))
                             for rd in self._curr_batch)
        if len(batch_keys_set) > 1:
            sys.stderr.write(
                '\n* WARNING: Mapped signal file batch contains reads with ' +
                'different keys.\n')
            batch_keys = sorted(set(k for ks in batch_keys_set for k in ks))
        else:
            batch_keys = batch_keys_set.pop()
        for k in batch_keys:
            k_type = (getattr(SignalMapping.req_data_types, k)
                      if k in SignalMapping.req_data_types._fields else
                      getattr(SignalMapping.opt_data_types, k))
            if k_type.__module__ == 'numpy':
                len_k = k + BATCH_LENGTH_SUFFIX
                batch_vals = []
                batch_len_val = np.zeros(len(self._curr_batch), dtype=np.int32)
                for ri, rd in enumerate(self._curr_batch):
                    try:
                        rv = rd[k]
                        batch_vals.append(rv)
                        batch_len_val[ri] = rv.shape[0]
                    except KeyError:
                        # skip missing keys (len 0 set at array init)
                        continue
                batch_vals = np.concatenate(batch_vals).astype(k_type)
                g.create_dataset(
                    k, data=batch_vals, compression='gzip', shuffle=True)
                g.create_dataset(
                    len_k, data=batch_len_val, compression='gzip',
                    shuffle=True)
            else:
                batch_vals = []
                for rd in self._curr_batch:
                    try:
                        batch_vals.append(rd[k])
                    except KeyError:
                        batch_vals.append('' if k_type is str else 0)
                        continue
                if k_type is str:
                    batch_vals = np.array(batch_vals, dtype=VAR_LEN_STR_DT)
                    k_ds = g.create_dataset(
                        k, batch_vals.shape, dtype=VAR_LEN_STR_DT,
                        compression="gzip")
                    k_ds[...] = batch_vals
                else:
                    batch_vals = np.array(batch_vals, dtype=k_type)
                    g.create_dataset(
                        k, data=batch_vals, compression='gzip', shuffle=True)

        self._curr_batch = []
        self._curr_batch_idx += 1

    def close(self):
        if len(self.read_ids) > 0:
            # special variable length string h5py data type
            read_ids = np.array(self.read_ids, dtype=VAR_LEN_STR_DT)
            # store read ids dataset in root so it isn't confused for a read
            read_ids_ds = self.hdf5.create_dataset(
                'read_ids', read_ids.shape, dtype=VAR_LEN_STR_DT,
                compression="gzip")
            read_ids_ds[...] = read_ids

        if len(self._curr_batch) > 0:
            self.write_curr_batch()

        self.hdf5.close()

    def write_read(self, readdict):
        read_id = readdict['read_id']
        self.read_ids.append(read_id)
        self._curr_batch.append(readdict)
        if len(self._curr_batch) >= self.batch_size:
            self.write_curr_batch()

    def _write_alphabet_info(self, alphabet_info):
        self.hdf5.attrs['alphabet'] = alphabet_info.alphabet
        self.hdf5.attrs['collapse_alphabet'] = alphabet_info.collapse_alphabet
        self.hdf5.attrs['mod_long_names'] = '\n'.join(
            alphabet_info.mod_long_names)

    def _write_version(self):
        self.hdf5.attrs['version'] = _version


def HDF5Reader(filename, load_in_mem=False):
    """ A file should contain mapped signal data written by the
    :class:`HDF5Writer`.
    This function opens either a per-read or batched HDF5 file for reading
    signal mappings.

    Args:
        filename (str): name of file to open.
        load_in_mem (bool): Load contents into memory using the h5py
            `driver='core'` option. Note that this is only applicable for
            per-read format files.

    Returns:
        Either :class:`PerReadHDF5Reader` or :class:`BatchHDF5Reader` object
    """
    with h5py.File(filename, 'r') as map_sig_hdf5:
        try:
            _ = map_sig_hdf5[BATCH_ROOT_TEXT]
            is_batch = True
        except KeyError:
            is_batch = False

    if is_batch:
        return BatchHDF5Reader(filename)
    return PerReadHDF5Reader(filename, load_in_mem)


def HDF5Writer(filename, alphabet_info, batch_format=True):
    """ A file storing mapped signal data in an HDF5 file.
    This function opens either a per-read or batched HDF5 file for storing
    signal mappings.

    Args:
        filename (str): name of file to open.
        alphabet_info (:class:`alphabet.AlphabetInfo`):  Alphabet to write
        batch_format (bool): Output batched mapped signal file format. This
            can significantly improve I/O performance and use less disk space.
            An entire batch must be loaded into memory in order access any read
            potentailly increasing RAM requirements.

    Returns:
        Either :class:`PerReadHDF5Writer` or :class:`BatchHDF5Writer` object
    """
    if batch_format:
        return BatchHDF5Writer(filename, alphabet_info)
    return PerReadHDF5Writer(filename, alphabet_info)


# global variables pointing to current default reader/writer
MappedSignalReader = HDF5Reader
MappedSignalWriter = HDF5Writer
