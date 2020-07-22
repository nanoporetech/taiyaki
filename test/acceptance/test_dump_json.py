import json
from parameterized import parameterized
import os
import unittest

import util


def is_valid_json(s):
    """ Tests whether a string can be parsed by json module

    Args:
        s (str): String containing JSON formatted data

    Returns:
        bool: True if data is loadable, False otherwise
    """
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


class AcceptanceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        test_directory = os.path.splitext(__file__)[0]
        self.testset_work_dir = os.path.basename(test_directory)
        self.script = os.path.join(util.BIN_DIR, "dump_json.py")
        self.json_to_cp = os.path.join(util.BIN_DIR, "json_to_checkpoint.py")
        self.model_file = os.path.join(
            util.MODELS_DIR, "mGru_flipflop_remapping_model_r9_DNA.checkpoint")

    def work_dir(self, test_name):
        """  Creates new directory in workspace if doesn't already exist

        Args:
            test_name (str): name of directory to create

        Returns:
            str: path of directory created
        """
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        """  Run without arguments, expect usage info
        """
        cmd = [self.script]
        util.run_cmd(self, cmd).expect_exit_code(
            2).expect_stderr(util.any_line_starts_with(u"usage"))

    @parameterized.expand([
        ["mGru_flipflop_remapping_model_r9_DNA.checkpoint"],
        ["mLstm_flipflop_model_r941_DNA.checkpoint"],
        ["mLstm_flipflop_model_r103_DNA.checkpoint"],
    ])
    def test_dump_to_stdout(self, model_name):
        """  Try dumping json to stdout

        Args:
            options (arrayof str): commandline options for command
            model_name (str): name of file from which to dump json
        """
        model_file = os.path.join(util.MODELS_DIR, model_name)
        self.assertTrue(os.path.exists(model_file))
        cmd = [self.script, model_file]
        print(cmd)
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout(
            lambda o: is_valid_json('\n'.join(o)))

    @unittest.expectedFailure
    def test_json_to_checkpoint(self):
        """  Try creating a checkpoint from a JSON file

        Notes:
            Unmaintained and expected to fail.

        """
        self.assertTrue(os.path.exists(self.model_file))
        test_work_dir = self.work_dir("test_json_to_checkpoint")

        # dump stored model to json
        json_file = os.path.join(test_work_dir, "output.json")
        # open and remove file in case it remains from previous test
        open(json_file, "w").close()
        os.remove(json_file)
        cmd = [self.script, self.model_file, "--output", json_file]
        util.run_cmd(self, cmd).expect_exit_code(0)
        self.assertTrue(os.path.exists(json_file))

        # convert json back to checkpoint
        re_model_file = os.path.join(test_work_dir, "re_model.checkpoint")
        cmd = [self.json_to_cp, json_file, "--output", re_model_file]
        open(re_model_file, "w").close()
        error_message = (
            "RuntimeError: File/path for 'output' exists, {}").format(
                re_model_file)
        util.run_cmd(self, cmd).expect_exit_code(1).expect_stderr(
            util.any_line_starts_with(error_message))
        os.remove(re_model_file)
        util.run_cmd(self, cmd).expect_exit_code(0)
        self.assertTrue(os.path.exists(re_model_file))

        # and finally convert checkpoint back to json again for comparison
        json_file2 = os.path.join(test_work_dir, "output2.json")
        # open and remove file in case it remains from previous test
        open(json_file2, "w").close()
        os.remove(json_file2)
        cmd = [self.script, re_model_file, "--output", json_file2]
        util.run_cmd(self, cmd).expect_exit_code(0)
        self.assertTrue(os.path.exists(json_file2))

        # note that checkpoints are not equal with a binary dump, but the
        # json files are due to the unused GRU bias parameters.
        json_dump = open(json_file, 'r').read()
        json_dump2 = open(json_file2, 'r').read()
        self.assertEqual(json_dump, json_dump2)
