import json
from parameterized import parameterized
import os
import sys
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
        self.model_file = os.path.join(util.MODELS_DIR, "mGru_flipflop_remapping_model_r9_DNA.checkpoint")

    def work_dir(self, test_name):
        directory = os.path.join(self.testset_work_dir, test_name)
        util.maybe_create_dir(directory)
        return directory

    def test_usage(self):
        cmd = [self.script]
        util.run_cmd(self, cmd).expect_exit_code(2).expect_stderr(util.any_line_starts_with(u"usage"))

    @parameterized.expand([
        [["--params"]],
        [["--no-params"]],
    ])
    def test_dump_to_stdout(self, options):
        self.assertTrue(os.path.exists(self.model_file))
        cmd = [self.script, self.model_file] + options
        util.run_cmd(self, cmd).expect_exit_code(0).expect_stdout(lambda o: is_valid_json('\n'.join(o)))

    @parameterized.expand([
        [["--no-params"], "2"],
    ])
    def test_dump_to_a_file(self, options, subdir):
        self.assertTrue(os.path.exists(self.model_file))
        test_work_dir = self.work_dir(os.path.join("test_dump_to_a_file", subdir))

        output_file = os.path.join(test_work_dir, "output.json")
        open(output_file, "w").close()

        cmd = [self.script, self.model_file, "--output", output_file] + options
        error_message = "RuntimeError: File/path for 'output' exists, {}".format(output_file)
        util.run_cmd(self, cmd).expect_exit_code(1).expect_stderr(util.any_line_starts_with(error_message))

        os.remove(output_file)

        util.run_cmd(self, cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(output_file))
        dump = open(output_file, 'r').read()

        self.assertTrue(is_valid_json(dump))

    def test_json_to_checkpoint(self):
        subdir="3"
        self.assertTrue(os.path.exists(self.model_file))
        test_work_dir = self.work_dir(os.path.join("test_json_to_checkpoint", subdir))

        output_file = os.path.join(test_work_dir, "output.json")
        cmd = [self.script, self.model_file, "--output", output_file, "--params"]
        util.run_cmd(self, cmd).expect_exit_code(0)
        self.assertTrue(os.path.exists(output_file))

        re_model_file = os.path.join(test_work_dir, "re_model.checkpoint")
        cmd = [self.json_to_cp, output_file, "--output", re_model_file]

        open(re_model_file, "w").close()
        error_message = "RuntimeError: File/path for 'output' exists, {}".format(re_model_file)
        util.run_cmd(self, cmd).expect_exit_code(1).expect_stderr(util.any_line_starts_with(error_message))
        os.remove(re_model_file)

        util.run_cmd(self, cmd).expect_exit_code(0)

        self.assertTrue(os.path.exists(re_model_file))
        model_dump = open(self.model_file, 'r').read()
        re_model_dump = open(re_model_file, 'r').read()

        self.assertEqual(model_dump, re_model_dump)
