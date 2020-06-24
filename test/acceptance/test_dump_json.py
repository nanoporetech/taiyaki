import json
from parameterized import parameterized
import os
import tempfile
import unittest

import util


def is_valid_json(s):
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
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        cmd = [self.script]
        util.run_cmd(self, cmd).expect_exit_code(
            2).expect_stderr(util.any_line_starts_with(u"usage"))

    @parameterized.expand([
        [["--params"], "mGru_flipflop_remapping_model_r9_DNA.checkpoint"],
        [["--no-params"], "mGru_flipflop_remapping_model_r9_DNA.checkpoint"],
        [["--params"], "mLstm_flipflop_model_r941_DNA.checkpoint"],
        [["--no-params"], "mLstm_flipflop_model_r941_DNA.checkpoint"],
        [["--params"], "mLstm_flipflop_model_r103_DNA.checkpoint"],
        [["--no-params"], "mLstm_flipflop_model_r103_DNA.checkpoint"],
    ])
    def test_dump_to_stdout(self, options, model_name):
        model_file = os.path.join(util.MODELS_DIR, model_name)
        self.assertTrue(os.path.exists(model_file))
        cmd = [self.script, model_file] + options
        print(cmd)
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout(
            lambda o: is_valid_json('\n'.join(o)))

    @parameterized.expand([
        [["--no-params"], "mGru_flipflop_remapping_model_r9_DNA.checkpoint"],
        [["--no-params"], "mLstm_flipflop_model_r941_DNA.checkpoint"],
        [["--no-params"], "mLstm_flipflop_model_r103_DNA.checkpoint"]
    ])
    def test_dump_to_a_file(self, options, model_name):
        model_file = os.path.join(util.MODELS_DIR, model_name)
        self.assertTrue(os.path.exists(self.model_file))

        test_work_dir = self.work_dir("test_dump_to_a_file")
        with tempfile.NamedTemporaryFile(dir=test_work_dir, suffix='.json',
                                         delete=False) as fh:
            output_file = fh.name

        cmd = [self.script, self.model_file, "--output", output_file] + options
        error_message = "RuntimeError: File/path for 'output' exists, {}".format(
            output_file)
        util.run_cmd(self, cmd).expect_exit_code(1).expect_stderr(
            util.any_line_starts_with(error_message))

        os.remove(output_file)

        util.run_cmd(self, cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_file))
        dump = open(output_file, 'r').read()

        self.assertTrue(is_valid_json(dump))

    @unittest.expectedFailure
    def test_json_to_checkpoint(self):
        subdir = "3"
        self.assertTrue(os.path.exists(self.model_file))
        test_work_dir = self.work_dir("test_json_to_checkpoint")

        # dump stored model to json
        json_file = os.path.join(test_work_dir, "output.json")
        # open and remove file in case it remains from previous test
        open(json_file, "w").close()
        os.remove(json_file)
        cmd = [self.script, self.model_file, "--output", json_file, "--params"]
        util.run_cmd(self, cmd).expect_exit_code(0)
        self.assertTrue(os.path.exists(json_file))

        # convert json back to checkpoint
        re_model_file = os.path.join(test_work_dir, "re_model.checkpoint")
        cmd = [self.json_to_cp, json_file, "--output", re_model_file]
        open(re_model_file, "w").close()
        error_message = "RuntimeError: File/path for 'output' exists, {}".format(
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
        cmd = [self.script, re_model_file, "--output", json_file2, "--params"]
        util.run_cmd(self, cmd).expect_exit_code(0)
        self.assertTrue(os.path.exists(json_file2))

        # note that checkpoints are not equal with a binary dump, but the
        # json files are due to the unused GRU bias parameters.
        json_dump = open(json_file, 'r').read()
        json_dump2 = open(json_file2, 'r').read()
        self.assertEqual(json_dump, json_dump2)
