import os
import shutil
import unittest

import util

class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        test_directory = os.path.splitext(__file__)[0]
        testset_name = os.path.basename(test_directory)

        self.testset_work_dir = testset_name

        self.script = os.path.join(util.BIN_DIR, "train_squiggle.py")

    def work_dir(self, test_name):
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        cmd = [self.script]
        util.run_cmd(self, cmd).expect_exit_code(2)

    def test_squiggle_training(self):
        test_work_dir = self.work_dir(os.path.join("test_squiggle_training"))

        output_directory = os.path.join(test_work_dir, "training_output")
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)

        
        hdf5_file = os.path.join(util.BIN_DIR, "../test/data/mapped_signal_file/mapped_remap_samref.hdf5")
        print("Trying to find ", hdf5_file)
        self.assertTrue(os.path.exists(hdf5_file))

        train_cmd = [self.script, "--batch_size", "50",
                     "--niteration", "1", "--save_every", "1",
                     "--seed","1", # Seed random numbers so test is reproducible
                     hdf5_file, output_directory]

        util.run_cmd(self, train_cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_directory))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_final.checkpoint")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model_final.params")))
        self.assertTrue(os.path.exists(os.path.join(output_directory, "model.log")))
