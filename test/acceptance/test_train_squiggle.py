import os
import shutil
import unittest

import util


class AcceptanceTest(unittest.TestCase):
    """Acceptance test for squiggle training"""
    @classmethod
    def setUpClass(self):
        """Set up directories and files"""
        test_directory = os.path.splitext(__file__)[0]
        testset_name = os.path.basename(test_directory)

        self.testset_work_dir = testset_name

        self.script = os.path.join(util.BIN_DIR, "train_squiggle.py")

    def work_dir(self, test_name):
        """Set up directory for a single test.
        
        Args:
            test_name (str): name of test (will be taken from stem of the
            name of the current file - see setupClass())
            
        Returns:
            str : directory path for files"""
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        """Run a test of the script with no command line args to see if we
        get the usage message"""
        cmd = [self.script]
        util.run_cmd(self, cmd).expect_exit_code(2)

    def test_squiggle_training(self):
        """Run a test of the script with example data"""
        test_work_dir = self.work_dir(os.path.join("test_squiggle_training"))

        output_directory = os.path.join(test_work_dir, "training_output")
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        hdf5_file = os.path.join(
            util.DATA_DIR, "mapped_signal_file/mapped_remap_samref.hdf5")
        print("Trying to find ", hdf5_file)
        self.assertTrue(os.path.exists(hdf5_file))

        train_cmd = [self.script, "--batch_size", "7", "--size", "17",
                     "--niteration", "1", "--save_every", "1",
                     "--target_len", "150", "--outdir", output_directory,
                     # Seed random numbers so test is reproducible
                     "--seed", "1",
                     hdf5_file]

        util.run_cmd(self, train_cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_directory))
        self.assertTrue(os.path.exists(os.path.join(
            output_directory, "model_final.checkpoint")))
        self.assertTrue(os.path.exists(os.path.join(
            output_directory, "model_final.params")))
        self.assertTrue(os.path.exists(os.path.join(
            output_directory, "model.log")))
