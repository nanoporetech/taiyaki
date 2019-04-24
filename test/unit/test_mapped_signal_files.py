import numpy as np
import os
import unittest
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# To run as a  single test, in taiyaki dir and in venv do
# pytest test/unit/test_mapped_signal_files.py

# lines which plot the ref_to_sig mapping and compare
# with result of searches to obtain sig_to_ref
# and with chunk limits are commented out with an 'if False'
# may be useful in debugging

from taiyaki import mapped_signal_files


def vectorprint(x):
    print('[' + (' '.join([str(i) for i in x])) + ']')


def construct_mapped_read():
    """Test data for a mapped read file.
    Returns a dictionary containing the data"""
    Nsig = 20
    Nref = 16
    reftosigstart = np.concatenate((
        np.array([-1, -1], dtype=np.int32),  # Start marker
        np.arange(2, 5, dtype=np.int32),     # Steps, starting at 2
        np.full(4, 5, dtype=np.int32),       # Stays (this is four fives, not five fours!)
        np.arange(7, 11, dtype=np.int32)     # Skip followed by steps
    ))
    reftosig = np.full(Nref + 1, Nsig, dtype=np.int32)  # Note length of reftosig is 1+reflen
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
        self.testfilepath = os.path.join(self.testset_work_dir, 'test_mapped_read_file.hdf5')
        self.plotfilepath = os.path.join(self.testset_work_dir, 'test_mapped_read_file.png')
        try:
            os.remove(self.testfilepath)
            print("Previous test file removed")
        except:
            print("No previous test file to remove")

    def test_HDF5_mapped_read_file(self):
        """Test that we can save a mapped read file, open it again and
        use some methods to get data from it. Plot a picture for diagnostics.
        """

        print("Creating Read object from test data")
        read_dict = construct_mapped_read()
        read_object = mapped_signal_files.Read(read_dict)
        print("Checking contents")
        check_text = read_object.check()
        print("Check result on read object:")
        print(check_text)
        self.assertEqual(check_text, "pass")

        print("Writing to file")
        with mapped_signal_files.HDF5(self.testfilepath, "w") as f:
            f.write_read(read_object['read_id'], read_object)
            f.write_version_number()

        print("Current dir = ", os.getcwd())
        print("File written to ", self.testfilepath)

        print("\nOpening file for reading")
        with mapped_signal_files.HDF5(self.testfilepath, "r") as f:
            ids = f.get_read_ids()
            print("Read ids=", ids[0])
            print("Version number = ", f.get_version_number())
            self.assertEqual(ids[0], read_dict['read_id'])

            file_test_report = f.check()
            print("Test report:", file_test_report)
            self.assertEqual(file_test_report, "pass")

            read_list = f.get_multiple_reads("all")

        recovered_read = read_list[0]
        reflen = len(recovered_read['Reference'])
        siglen = len(recovered_read['Dacs'])

        # Get a chunk - note that chunkstart is relative to the start of the mapped
        # region, not relative to the start of the signal
        chunklen, chunkstart = 5, 3
        chunkdict = recovered_read.get_chunk_with_sample_length(chunklen, chunkstart)

        # Check that the extracted chunk is the right length
        self.assertEqual(len(chunkdict['current']), chunklen)

        # Check that the mapping data agrees with what we put in
        self.assertTrue(np.all(recovered_read['Ref_to_signal']==read_dict['Ref_to_signal']))

        # Plot a picture showing ref_to_sig from the read object,    def setup():
        # and the result of searches to find the inverse
        if False:
            plt.figure()
            plt.xlabel('Signal coord')
            plt.ylabel('Ref coord')
            ix = np.array([0, -1])
            plt.scatter(chunkdict['current'][ix], chunkdict['sequence'][ix],
                        s=50, label='chunk limits', marker='s', color='black')
            plt.scatter(recovered_read['Ref_to_signal'], np.arange(reflen + 1), label='reftosig (source data)',
                        color='none', edgecolor='blue', s=60)
            siglocs = np.arange(siglen, dtype=np.int32)
            sigtoref_fromsearch = recovered_read.get_reference_locations(siglocs)
            plt.scatter(siglocs, sigtoref_fromsearch, label='from search', color='red', marker='x', s=50)
            plt.legend()
            plt.grid()
            plt.savefig(self.plotfilepath)
            print("Saved plot to", self.plotfilepath)

        #raise Exception("Fail so we can read output")
        return

    def test_check_HDF5_mapped_read_file(self):
        """Check that constructing a read object which doesn't conform
        leads to errors.
        """
        print("Creating flawed Read object from test data")
        read_dict = construct_mapped_read()
        read_dict['Reference'] = "I'm not a numpy array!"  # Wrong type!
        read_object = mapped_signal_files.Read(read_dict)
        print("Checking contents")
        check_text = read_object.check()
        print("Check result on read object: should fail")
        print(check_text)
        self.assertNotEqual(check_text, "pass")

        print("Writing to file")
        with mapped_signal_files.HDF5(self.testfilepath, "w") as f:
            f.write_read(read_object['read_id'], read_object)
            f.write_version_number()

        print("Current dir = ", os.getcwd())
        print("File written to ", self.testfilepath)

        print("\nOpening file for reading")
        with mapped_signal_files.HDF5(self.testfilepath, "r") as f:
            ids = f.get_read_ids()
            print("Read ids=", ids[0])
            print("Version number = ", f.get_version_number())
            self.assertEqual(ids[0], read_dict['read_id'])

            file_test_report = f.check()
            print("Test report (should fail):", file_test_report)
            self.assertNotEqual(file_test_report, "pass")

        #raise Exception("Fail so we can read output")
        return
