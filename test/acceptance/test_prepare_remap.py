import os
import subprocess
import unittest

from taiyaki.mapped_signal_files import MappedSignalReader


class AcceptanceTest(unittest.TestCase):
    """Tests the prepare_mapped_reads.py script on a single fast5 file

    Uses cached per-read-params and references files
    """

    @classmethod
    def setUpClass(self):
        """ Make all paths absolute so that when we run Makefile in another
        dir it works OK """
        testset_directory_rel, _ = os.path.splitext(__file__)
        self.testset_name = os.path.basename(testset_directory_rel)
        self.taiyakidir = os.path.abspath(
            os.path.join(testset_directory_rel, '../../..'))
        self.testset_work_dir = os.path.join(
            self.taiyakidir, 'build/acctest/' + self.testset_name)
        os.makedirs(self.testset_work_dir, exist_ok=True)
        self.datadir = os.path.join(self.taiyakidir, 'test/data')
        self.read_dir = os.path.join(self.datadir, 'reads')
        self.per_read_refs = os.path.join(
            self.datadir, 'per_read_references.fasta')
        self.mod_per_read_refs = os.path.join(
            self.datadir, 'per_read_references.mod_bases.fasta')
        self.per_read_params = os.path.join(self.datadir, 'readparams.tsv')
        self.output_mapped_signal_file = os.path.join(
            self.testset_work_dir, 'mapped_signals.hdf5')
        self.remapping_model = os.path.join(
            self.taiyakidir,
            "models/mGru_flipflop_remapping_model_r9_DNA.checkpoint")
        self.script = os.path.join(
            self.taiyakidir, "bin/prepare_mapped_reads.py")

    def test_prepare_remap(self):
        """Tests the prepare_mapped_reads.py script"""
        print("Current directory is", os.getcwd())
        print("Taiyaki dir is", self.taiyakidir)
        print("Data dir is ", self.datadir)
        output_mapped_signal_file = (
            self.output_mapped_signal_file + '_test_prepare_remap')
        cmd = [self.script,
               self.read_dir,
               self.per_read_params,
               output_mapped_signal_file,
               self.remapping_model,
               self.per_read_refs,
               "--overwrite"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        print("Result of running make command in shell:")
        print("Stdout=", r.stdout.decode('utf-8'))
        print("Stderr=", r.stderr.decode('utf-8'))
        self.assertEqual(r.returncode, 0)

        # Open mapped read file and run checks to see if it complies with file
        # format.
        # Also get a chunk and check that speed is within reasonable bounds
        self.assertTrue(os.path.exists(output_mapped_signal_file))
        with MappedSignalReader(output_mapped_signal_file) as msr:
            testreport = msr.check()
            print("Test report from checking mapped read file:")
            print('"', testreport, '"')
            self.assertEqual(testreport, "pass")
            read0 = next(msr.reads())
            chunk = read0.get_chunk_with_sample_length(1000, start_sample=10)
            # Defined start_sample to make it reproducible - otherwise randomly
            # located chunk is returned.
            chunk_meandwell = chunk.sig_len / (chunk.seq_len + 0.0001)
            print("chunk mean dwell time in samples = ", chunk_meandwell)
            assert 7 < chunk_meandwell < 13, (
                "Chunk mean dwell time outside allowed range 7 to 13")

    def test_mod_prepare_remap(self):
        """Tests the prepare_mapped_reads.py script with modified bases"""
        print("Current directory is", os.getcwd())
        print("Taiyaki dir is", self.taiyakidir)
        print("Data dir is ", self.datadir)
        output_mapped_signal_file = (
            self.output_mapped_signal_file + 'test_mod_prepare_remap')
        cmd = [self.script,
               self.read_dir,
               self.per_read_params,
               output_mapped_signal_file,
               self.remapping_model,
               self.mod_per_read_refs,
               "--mod", "Z", "C", "5mC",
               "--mod", "Y", "A", "6mA",
               "--overwrite"]
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
        print("Result of running make command in shell:")
        print("Stdout=", r.stdout.decode('utf-8'))
        print("Stderr=", r.stderr.decode('utf-8'))
        self.assertEqual(r.returncode, 0)

        # Open mapped read file and run checks to see if it complies with file
        # format
        # Also get a chunk and check that speed is within reasonable bounds
        self.assertTrue(os.path.exists(output_mapped_signal_file))
        with MappedSignalReader(output_mapped_signal_file) as msr:
            testreport = msr.check()
            print("Test report from checking mapped read file:")
            print('"', testreport, '"')
            self.assertEqual(testreport, "pass")
            read0 = next(msr.reads())
            chunk = read0.get_chunk_with_sample_length(1000, start_sample=10)
            # Defined start_sample to make it reproducible - otherwise randomly
            # located chunk is returned.
            chunk_meandwell = chunk.sig_len / (chunk.seq_len + 0.0001)
            print("chunk mean dwell time in samples = ", chunk_meandwell)
            assert 7 < chunk_meandwell < 13, (
                "Chunk mean dwell time outside allowed range 7 to 13")
