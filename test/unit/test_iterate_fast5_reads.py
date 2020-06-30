import os
import unittest

from . import DATA_DIR
from taiyaki.fast5utils import iterate_fast5_reads


class TestStrandList(unittest.TestCase):
    """Test strand list reading function"""
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
        """Test to see if read ids found.
        Args:
            found_reads (list of str) : reads found by file reader
        """
        found_read_ids = sorted([rid for _, rid in found_reads])
        self.assertEqual(found_read_ids, self.EXPECTED_READ_IDS)

    def test_no_strand_list_multiread(self):
        """See if read ids found in multiread file with no strand list"""
        self._check_found_read_ids(iterate_fast5_reads(self.MULTIREAD_DIR))

    def test_no_strand_list_single_reads(self):
        """See if read ids found in single-read file with no strand list"""
        self._check_found_read_ids(iterate_fast5_reads(self.READ_DIR))

    def test_sequencing_summary_multiread(self):
        """See if read ids found using sequencing-summary style strand list"""
        self._check_found_read_ids(iterate_fast5_reads(
            self.MULTIREAD_DIR, strand_list=self.SEQUENCING_SUMMARY))

    def test_strand_list_single_reads(self):
        """See if read ids found using strandlist with single read fast5s"""
        strand_list = os.path.join(
            self.STRAND_LIST_DIR, "strand_list_single.txt")
        self._check_found_read_ids(iterate_fast5_reads(
            self.READ_DIR, strand_list=strand_list))

    def test_strand_list_multiread(self):
        """See if read ids found using strand list with multi read fast5s"""
        strand_list = os.path.join(self.STRAND_LIST_DIR, "strand_list.txt")
        self._check_found_read_ids(iterate_fast5_reads(
            self.MULTIREAD_DIR, strand_list=strand_list))

    def test_strand_list_no_filename_multiread(self):
        """See if read ids found iterating through multiread files in directory
        with strand list"""
        strand_list = os.path.join(
            self.STRAND_LIST_DIR, "strand_list_no_filename.txt")
        self._check_found_read_ids(iterate_fast5_reads(
            self.MULTIREAD_DIR, strand_list=strand_list))

    def test_strand_list_no_filename_single_reads(self):
        """See if read ids found iterating through single-read files in
        directory with strand list"""
        strand_list = os.path.join(
            self.STRAND_LIST_DIR, "strand_list_no_filename.txt")
        self._check_found_read_ids(iterate_fast5_reads(
            self.READ_DIR, strand_list=strand_list))

    def test_strand_list_no_read_id_multiread(self):
        """See if reads ids found iterating through strand list containing
        filenames, not read ids"""
        strand_list = os.path.join(
            self.STRAND_LIST_DIR, "strand_list_no_read_id.txt")
        self._check_found_read_ids(iterate_fast5_reads(
            self.MULTIREAD_DIR, strand_list=strand_list))

    def test_strand_list_invalid(self):
        """Use strand list with no header line. Should throw an exception."""
        strand_list = os.path.join(
            self.STRAND_LIST_DIR, "invalid_strand_list_no_header.txt")
        with self.assertRaises(Exception):
            for fn, rid in iterate_fast5_reads(self.MULTIREAD_DIR, strand_list=strand_list):
                print("Filename=", fn, "read_id=", rid)

    # TODO add recursive test (requires adding recursive data dir
