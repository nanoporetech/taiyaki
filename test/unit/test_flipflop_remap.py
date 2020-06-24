import numpy as np
import unittest

from taiyaki import flipflop_remap, signal_mapping, signal


class TestFlipFlopMapping(unittest.TestCase):

    def test_flipflop_mapping(self):
        """Test that global flipflop remapping works as expected

        Sequence is AABA from an alphabet {A, B}

        Transition scores from 6 time points are used

        The best path is AaaBBAA where upper-case is a flip and
        lower-case is a flop

        All transition scores are set to 1 (on best path) and 0
        otherwise so the score for the best path should be exactly 6.
        """
        sequence = 'AABA'
        alphabet = 'AB'
        log_transitions = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Aa step
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # aa stay
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # aB step
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # BB stay
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # BA step
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # AA stay
        ], dtype='f4')
        score, path = flipflop_remap.flipflop_remap(
            log_transitions, sequence, alphabet=alphabet, localpen=-0.5)
        self.assertEqual(score, 6.0)
        self.assertEqual(path.tolist(), [0, 1, 1, 2, 2, 3, 3])

        # Check we get the same with the lower-level interface
        step_index = [8, 6, 1]
        step_score = log_transitions[:, step_index]
        stay_index = [0, 10, 5, 0]
        stay_score = log_transitions[:, stay_index]
        score2, path2 = flipflop_remap.map_to_crf_viterbi(
            log_transitions, step_index, stay_index, localpen=-0.5)
        self.assertEqual(score, score2)
        self.assertEqual(path.tolist(), path2.tolist())

    def test_flipflop_mapping_glocal(self):
        """Test the glocal flipflop remapping works as expected

        Sequence is BA from an alphabet {A, B}

        Transition scores from 5 time points are used

        The best path is --BA- where upper-case is a flip and
        lower-case is a flop, and - represents parts that should be
        clipped by the local mapping

        All transition scores are set to 1 (on best path) and 0
        otherwise. Scores in the local state are set to 0.5, so the
        best path should have a score of 3.5.
        """
        sequence = 'BA'
        alphabet = 'AB'
        log_transitions = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # clip
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # clip
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # BB stay
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # BA step
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # clip
        ], dtype='f4')
        score, path = flipflop_remap.flipflop_remap(
            log_transitions, sequence, alphabet=alphabet, localpen=-0.5)
        self.assertEqual(score, 3.5)
        self.assertEqual(path.tolist(), [-1, -1, 0, 0, 1, -1])

        # Check we get the same with the lower-level interface
        step_index = [1]
        step_score = log_transitions[:, step_index]
        stay_index = [5, 0]
        stay_score = log_transitions[:, stay_index]
        score2, path2 = flipflop_remap.map_to_crf_viterbi(
            log_transitions, step_index, stay_index, localpen=-0.5)
        self.assertEqual(score, score2)
        self.assertEqual(path.tolist(), path2.tolist())

    def test_mapping_reftosignal(self):
        """Test the conversion from remapped path to reftosignal output
        """
        sig = signal.Signal(dacs=np.zeros(12))
        # testing path with a single skip (over 3rd base; first "T")
        path = np.array([-1, 0, 0, 1, 1, 1, 3, 3, 3,
                         4, 4, 5, 6], dtype=np.int32)
        reference = 'ACTACGT'

        int_ref = signal_mapping.SignalMapping.get_integer_reference(
            reference, 'ACGT')
        sigtoref_res = signal_mapping.SignalMapping.from_remapping_path(
            path, int_ref, 1, sig).Ref_to_signal
        self.assertEqual(sigtoref_res.tolist(),
                         [0, 2, 5, 5, 8, 10, 11, 12])

        # now test with clipped bases
        sig = signal.Signal(dacs=np.zeros(15))
        # testing path with a single skip (over 4th base; first "T")
        path = np.array([-1, -1, 1, 1, 2, 2, 2, 4, 4, 4, 5,
                         5, 6, 7, -1, -1], dtype=np.int32)
        reference = 'AACTACGTTT'

        int_ref = signal_mapping.SignalMapping.get_integer_reference(
            reference, 'ACGT')
        sigtoref_res = signal_mapping.SignalMapping.from_remapping_path(
            path, int_ref, 1, sig).Ref_to_signal
        self.assertEqual(sigtoref_res.tolist(),
                         [-1, 1, 3, 6, 6, 9, 11, 12, 13, 16, 16])

        return

    def test_mapping_reftosignal_stride_2(self):
        """Test the conversion from remapped path to reftosignal output
        """
        sig = signal.Signal(dacs=np.zeros(24))
        # testing path with a single skip (over 3rd base; first "T")
        path = np.array([-1, 0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 5, 6],
                        dtype=np.int32)
        reference = 'ACTACGT'

        int_ref = signal_mapping.SignalMapping.get_integer_reference(
            reference, 'ACGT')
        sigtoref_res = signal_mapping.SignalMapping.from_remapping_path(
            path, int_ref, 2, sig).Ref_to_signal
        self.assertEqual(sigtoref_res.tolist(),
                         [1, 5, 11, 11, 17, 21, 23, 24])

        # now test with clipped bases
        sig = signal.Signal(dacs=np.zeros(30))
        # testing path with a single skip (over 4th base; first "T")
        path = np.array([-1, -1, 1, 1, 2, 2, 2, 4, 4, 4, 5,
                         5, 6, 7, -1, -1], dtype=np.int32)
        reference = 'AACTACGTTT'

        int_ref = signal_mapping.SignalMapping.get_integer_reference(
            reference, 'ACGT')
        sigtoref_res = signal_mapping.SignalMapping.from_remapping_path(
            path, int_ref, 2, sig).Ref_to_signal
        self.assertEqual(sigtoref_res.tolist(),
                         [-1, 3, 7, 13, 13, 19, 23, 25, 26, 31, 31])

        return
