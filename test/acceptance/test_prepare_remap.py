import os
import subprocess
import unittest


from taiyaki import mapped_signal_files

class AcceptanceTest(unittest.TestCase):
    """Test on a single fast5 file runs the first part of the workflow
    Makefile to make the per-read-params file and reference file
    and then do remapping"""

    @classmethod
    def setUpClass(self):
        """Make all paths absolute so that when we run Makefile in another dir it works OK"""
        testset_directory_rel, _ = os.path.splitext(__file__)
        self.testset_name = os.path.basename(testset_directory_rel)
        self.taiyakidir = os.path.abspath(os.path.join(testset_directory_rel,'../../..'))
        self.testset_work_dir = os.path.join(self.taiyakidir,'build/acctest/'+self.testset_name)
        os.makedirs(self.testset_work_dir, exist_ok=True)
        self.datadir = os.path.join(self.taiyakidir,'test/data')
        self.read_dir = os.path.join(self.datadir,'reads')
        self.per_read_refs = os.path.join(self.datadir,'per_read_references.fasta')
        self.per_read_params = os.path.join(self.datadir,'readparams.tsv')
        self.output_mapped_signal_file = os.path.join(self.testset_work_dir,'mapped_signals.hdf5')
        self.remapping_model = os.path.join(self.taiyakidir,"models/mGru_flipflop_remapping_model_r9_DNA.checkpoint")
        self.script = os.path.join(self.taiyakidir,"bin/prepare_mapped_reads.py")

    def test_prepare_remap(self):
        print("Current directory is",os.getcwd())
        print("Taiyaki dir is",self.taiyakidir)
        print("Data dir is ",self.datadir)
        cmd = [self.script,
               self.read_dir,
               self.per_read_params,
               self.output_mapped_signal_file,
               self.remapping_model,
               self.per_read_refs,
               "--device","cpu"]
        r=subprocess.run(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        print("Result of running make command in shell:")
        print("Stdout=",r.stdout.decode('utf-8'))
        print("Stderr=",r.stderr.decode('utf-8'))

        # Open mapped read file and run checks to see if it complies with file format
        # Also get a chunk and check that speed is within reasonable bounds
        with mapped_signal_files.HDF5(self.output_mapped_signal_file,"r") as f:
            testreport = f.check()
            print("Test report from checking mapped read file:")
            print(testreport)
            self.assertEqual(testreport,"pass")
            read0 = f.get_multiple_reads("all")[0]
            chunk = read0.get_chunk_with_sample_length(1000,start_sample=10)
            # Defined start_sample to make it reproducible - otherwise randomly
            # located chunk is returned.
            chunk_meandwell = len(chunk['current']) / (len(chunk['sequence']) + 0.0001)
            print("chunk mean dwell time in samples = ", chunk_meandwell)
            assert 7 < chunk_meandwell < 13, "Chunk mean dwell time outside allowed range 7 to 13"

        return
