import tempfile
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from taiyaki.constants import DEFAULT_ALPHABET
from taiyaki import alphabet, mapped_signal_files, signal_mapping

if True:
    #  Protect in block to prevent autopep8 refactoring
    import matplotlib
    matplotlib.use('Agg')

# To run as a  single test, in taiyaki dir and in venv do
# pytest test/unit/test_mapped_signal_files.py

# lines which plot the ref_to_sig mapping and compare
# with result of searches to obtain sig_to_ref
# and with chunk limits are commented out with an 'if False'
# may be useful in debugging


def vectorprint(x):
    print('[' + (' '.join([str(i) for i in x])) + ']')


def construct_mapped_read_dict():
    """Constructs test data for a mapped read file

    Returns:
        dictionary containing the test data
    """
    Nsig = 20
    Nref = 16
    reftosigstart = np.concatenate((
        np.array([-1, -1], dtype=np.int32),  # Start marker
        np.arange(2, 5, dtype=np.int32),     # Steps, starting at 2
        # Stays (this is four fives, not five fours!)
        np.full(4, 5, dtype=np.int32),
        np.arange(7, 11, dtype=np.int32)     # Skip followed by steps
    ))
    # Note length of reftosig is 1+reflen
    reftosig = np.full(Nref + 1, Nsig, dtype=np.int32)
    reftosig[:len(reftosigstart)] = reftosigstart
    return {
        'shift_frompA': 0.0,
        'scale_frompA': 0.001,
        'range': 1.0,
        'offset': 0.0,
        'digitisation': float(1000),
        'Dacs': np.arange(Nsig, dtype=np.int16),
        'Ref_to_signal': reftosig,
        'Reference': np.arange(Nref, dtype=np.int16),
        'read_id': '11b284b3-397f-45e1-b065-9965c10857ac'
    }


class TestMappedReadFiles(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_directory = os.path.splitext(__file__)[0]
        self.testset_name = os.path.basename(self.test_directory)
        self.testset_work_dir = self.testset_name
        os.makedirs(self.testset_work_dir, exist_ok=True)
        self.plotfilepath = os.path.join(
            self.testset_work_dir, 'test_mapped_read_file.png')
        try:
            os.remove(self.testfilepath)
            print("Previous test file removed")
        except Exception:
            print("No previous test file to remove")

    def test_HDF5_mapped_read_file(self):
        """Test that we can save a mapped read file and open it again

        Also produces a plot for diagnostic purposes
        """

        print("Creating Read object from test data")
        read_dict = construct_mapped_read_dict()
        read_object = signal_mapping.SignalMapping(**read_dict)
        print("Checking contents")
        check_text = read_object.check()
        print("Check result on read object:")
        print(check_text)
        self.assertEqual(check_text, "pass")

        print("Writing to file")
        with tempfile.NamedTemporaryFile(
                delete=False, dir=self.testset_work_dir) as fh:
            testfilepath = fh.name
        alphabet_info = alphabet.AlphabetInfo(
            DEFAULT_ALPHABET, DEFAULT_ALPHABET)
        with mapped_signal_files.MappedSignalWriter(
                testfilepath, alphabet_info) as f:
            f.write_read(read_object.get_read_dictionary())

        print("Current dir = ", os.getcwd())
        print("File written to ", testfilepath)

        print("\nOpening file for reading")
        with mapped_signal_files.MappedSignalReader(testfilepath) as f:
            ids = f.get_read_ids()
            print("Read ids=", ids[0])
            print("Version number = ", f.version)
            self.assertEqual(ids[0], read_dict['read_id'])

            file_test_report = f.check()
            print("Test report:", file_test_report)
            self.assertEqual(file_test_report, "pass")

            read_list = list(f.reads())

        recovered_read = read_list[0]
        reflen = len(recovered_read.Reference)
        siglen = len(recovered_read.Dacs)

        # Get a chunk - note that chunkstart is relative to the start of
        # the mapped region, not relative to the start of the signal
        chunklen, chunkstart = 5, 3
        chunk = recovered_read.get_chunk_with_sample_length(
            chunklen, chunkstart)

        # Check that the extracted chunk is the right length
        self.assertEqual(chunk.sig_len, chunklen)

        # Check that the mapping data agrees with what we put in
        self.assertTrue(np.all(recovered_read.Ref_to_signal ==
                               read_dict['Ref_to_signal']))

        # Plot a picture showing ref_to_sig from the read object,
        # and the result of searches to find the inverse
        if False:
            plt.figure()
            plt.xlabel('Signal coord')
            plt.ylabel('Ref coord')
            ix = np.array([0, -1])
            plt.scatter(chunk.current[ix], chunk.sequence[ix],
                        s=50, label='chunk limits', marker='s', color='black')
            plt.scatter(recovered_read.Ref_to_signal, np.arange(reflen + 1),
                        label='reftosig (source data)',
                        color='none', edgecolor='blue', s=60)
            siglocs = np.arange(siglen, dtype=np.int32)
            sigtoref_fromsearch = recovered_read.get_reference_locations(
                siglocs)
            plt.scatter(siglocs, sigtoref_fromsearch,
                        label='from search', color='red', marker='x', s=50)
            plt.legend()
            plt.grid()
            plt.savefig(self.plotfilepath)
            print("Saved plot to", self.plotfilepath)

    def test_check_HDF5_mapped_read_file(self):
        """Check that constructing a read object which doesn't conform
        leads to errors.
        """
        print("Creating Read object from test data")
        valid_read_dict = construct_mapped_read_dict()
        valid_read_object = signal_mapping.SignalMapping(**valid_read_dict)
        print("Checking contents")
        check_text = valid_read_object.check()
        print("Check result on valid read object: should pass")
        print(check_text)
        self.assertEqual(check_text, signal_mapping.SignalMapping.pass_str)

        print("Creating flawed Read object from test data")
        invalid_read_dict = construct_mapped_read_dict()
        # set reference to incorrect length
        invalid_read_dict['Reference'] = np.zeros(
            len(invalid_read_dict['Reference']) - 1, dtype=np.int32)
        invalid_read_object = signal_mapping.SignalMapping(**invalid_read_dict)
        print("Checking contents")
        check_text = invalid_read_object.check()
        print("Check result on invalid read object: should fail")
        print(check_text)
        self.assertNotEqual(check_text, signal_mapping.SignalMapping.pass_str)

        print("Writing invalid read to file")
        alphabet_info = alphabet.AlphabetInfo(
            DEFAULT_ALPHABET, DEFAULT_ALPHABET)
        with tempfile.NamedTemporaryFile(
                delete=True, dir=self.testset_work_dir) as fh:
            testfilepath = fh.name
        with mapped_signal_files.MappedSignalWriter(
                testfilepath, alphabet_info) as f:
            try:
                f.write_read(invalid_read_object.get_read_dictionary())
            except signal_mapping.TaiyakiSigMapError:
                pass
            else:
                self.assertTrue(False, 'Invalid read passed checks.')

        print("Writing valid read to file")
        with tempfile.NamedTemporaryFile(
                delete=False, dir=self.testset_work_dir) as fh:
            testfilepath = fh.name
        with mapped_signal_files.MappedSignalWriter(
                testfilepath, alphabet_info) as f:
            try:
                f.write_read(valid_read_object.get_read_dictionary())
            except signal_mapping.TaiyakiSigMapError:
                self.assertTrue(False, 'Valid read failed checks.')

        print("Current dir = ", os.getcwd())
        print("File written to ", testfilepath)

        print("\nOpening valid file for reading")
        with mapped_signal_files.MappedSignalReader(testfilepath) as f:
            ids = f.get_read_ids()
            print("Read ids=", ids[0])
            print("Version number = ", f.version)
            self.assertEqual(ids[0], valid_read_dict['read_id'])

            file_test_report = f.check()
            print("Test report (should pass):", file_test_report)
            self.assertEqual(
                file_test_report, signal_mapping.SignalMapping.pass_str)
