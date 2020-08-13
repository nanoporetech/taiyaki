import os
import unittest

# To run as a  single test, in directory build/acctest and in venv do
# pytest ../../test/acceptance/test_merge_mappedsignalfiles.py

import util


from taiyaki import mapped_signal_files


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        test_directory = os.path.splitext(__file__)[0]
        self.testset_work_dir = os.path.basename(test_directory)
        self.merge_script = os.path.join(
            util.MISC_DIR, "merge_mappedsignalfiles.py")
        self.plot_script = os.path.join(
            util.MISC_DIR, "plot_mapped_signals.py")
        self.mapped_signal_file0 = os.path.join(
            util.DATA_DIR, "mapped_signal_file/mapped_reads_0.hdf5")
        self.mapped_signal_file1 = os.path.join(
            util.DATA_DIR, "mapped_signal_file/mapped_reads_1.hdf5")

    def work_dir(self, test_name):
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        cmd = [self.merge_script]
        # cmd = ['python3',self.merge_script]
        util.run_cmd(self, cmd).expect_exit_code(2).expect_stderr(
            util.any_line_starts_with(u"usage"))

    def count_reads(self, mapped_signal_file, print_readlist=True):
        """Count the number of reads in a mapped signal file."""
        with mapped_signal_files.HDF5Reader(mapped_signal_file) as f:
            read_ids = f.get_read_ids()
            if print_readlist:
                print("Read list:")
                print('\n'.join(read_ids))
        return len(read_ids)

    def test_merge(self):
        test_work_dir = self.work_dir("test_merge")
        merged_mapped_signal_file = os.path.join(
            test_work_dir, 'merged_mappedsignalfile.hdf5')
        print("Looking for {} (absolute path = {})".format(
            self.mapped_signal_file0,
            os.path.abspath(self.mapped_signal_file0)))
        print("Result to be placed in {} (absolute path = {})".format(
            merged_mapped_signal_file,
            os.path.abspath(merged_mapped_signal_file)))
        self.assertTrue(os.path.exists(self.mapped_signal_file0))
        self.assertTrue(os.path.exists(self.mapped_signal_file1))

        # Merge two mapped signal files
        cmd = [self.merge_script,
               merged_mapped_signal_file,
               "--input", self.mapped_signal_file0, "None",
               "--input", self.mapped_signal_file1, "None"]
        util.run_cmd(self, cmd)

        # Output of print statements only becomes accessible if test fails
        print("Counting reads in mapped signal file 0")
        numreads_0 = self.count_reads(self.mapped_signal_file0)
        print("Counting reads in mapped signal file 1")
        numreads_1 = self.count_reads(self.mapped_signal_file1)
        numreads_in = numreads_0 + numreads_1
        print(("Total number of reads in files to be merged " +
               "= {} + {} = {}").format(
                   numreads_0, numreads_1, numreads_in))

        print("Counting reads in merged mapped signal file")
        numreads_out = self.count_reads(merged_mapped_signal_file)
        print("Total number of reads in merged file = {}".format(numreads_out))

        self.assertTrue(numreads_in == numreads_out)
        self.assertTrue(numreads_in > 2)

        return
