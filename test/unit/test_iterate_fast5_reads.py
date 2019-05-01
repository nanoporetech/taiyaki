import os
import unittest

from . import DATA_DIR
from taiyaki.fast5utils import iterate_fast5_reads


class TestStrandList(unittest.TestCase):
    READ_DIR = os.path.join(DATA_DIR, "reads")
    MULTIREAD_DIR = os.path.join(DATA_DIR, "multireads")
    EXPECTED_READ_IDS = [
        '0f776a08-1101-41d4-8097-89136494a46e',
        '1f1a0f33-e2ac-431a-8f48-c3c687a7a7dc',
        'b7096acd-b528-474e-a863-51295d18d3de',
        'db6b45aa-5d21-45cf-a435-05fb8f12e839',
        'de1508c4-755b-489e-9ffb-51af35c9a7e6',
    ]
    STRAND_LIST_DIR = os.path.join(DATA_DIR, "strand_lists")
    SEQUENCING_SUMMARY = os.path.join(
        DATA_DIR, "basecaller_output/sequencing_summary.txt")

    def _check_found_read_ids(self, found_reads):
        found_read_ids = sorted([rid for _, rid in found_reads])
        self.assertEqual(found_read_ids, self.EXPECTED_READ_IDS)

    def test_no_strand_list_multiread(self):
        self._check_found_read_ids(iterate_fast5_reads(self.MULTIREAD_DIR))

    def test_no_strand_list_single_reads(self):
        self._check_found_read_ids(iterate_fast5_reads(self.READ_DIR))

    def test_sequencing_summary_multiread(self):
        self._check_found_read_ids(iterate_fast5_reads(self.MULTIREAD_DIR, strand_list=self.SEQUENCING_SUMMARY))

    def test_strand_list_single_reads(self):
        strand_list = os.path.join(self.STRAND_LIST_DIR, "strand_list_single.txt")
        self._check_found_read_ids(iterate_fast5_reads(self.READ_DIR, strand_list=strand_list))

    def test_strand_list_multiread(self):
        strand_list = os.path.join(self.STRAND_LIST_DIR, "strand_list.txt")
        self._check_found_read_ids(iterate_fast5_reads(self.MULTIREAD_DIR, strand_list=strand_list))

    def test_strand_list_no_filename_multiread(self):
        strand_list = os.path.join(self.STRAND_LIST_DIR, "strand_list_no_filename.txt")
        self._check_found_read_ids(iterate_fast5_reads(self.MULTIREAD_DIR, strand_list=strand_list))

    def test_strand_list_no_filename_single_reads(self):
        strand_list = os.path.join(self.STRAND_LIST_DIR, "strand_list_no_filename.txt")
        self._check_found_read_ids(iterate_fast5_reads(self.READ_DIR, strand_list=strand_list))

    def test_strand_list_no_read_id_multiread(self):
        strand_list = os.path.join(self.STRAND_LIST_DIR, "strand_list_no_read_id.txt")
        self._check_found_read_ids(iterate_fast5_reads(self.MULTIREAD_DIR, strand_list=strand_list))

    # TODO add recursive test (requires adding recursive data dir
