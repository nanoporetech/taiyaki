import numpy as np
import unittest

from taiyaki import flipflop_remap


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
        score, path = flipflop_remap.flipflop_remap(log_transitions, sequence, alphabet=alphabet, localpen=-0.5)
        self.assertEqual(score, 6.0)
        self.assertEqual(path.tolist(), [0, 1, 1, 2, 2, 3, 3])

        # Check we get the same with the lower-level interface
        step_index = [8, 6, 1]
        step_score = log_transitions[:, step_index]
        stay_index = [0, 10, 5, 0]
        stay_score = log_transitions[:, stay_index]
        score2, path2 = flipflop_remap.map_to_crf_viterbi(log_transitions, step_index, stay_index, localpen=-0.5)
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
        score, path = flipflop_remap.flipflop_remap(log_transitions, sequence, alphabet=alphabet, localpen=-0.5)
        self.assertEqual(score, 3.5)
        self.assertEqual(path.tolist(), [-1, -1, 0, 0, 1, -1])

        # Check we get the same with the lower-level interface
        step_index = [1]
        step_score = log_transitions[:, step_index]
        stay_index = [5, 0]
        stay_score = log_transitions[:, stay_index]
        score2, path2 = flipflop_remap.map_to_crf_viterbi(log_transitions, step_index, stay_index, localpen=-0.5)
        self.assertEqual(score, score2)
        self.assertEqual(path.tolist(), path2.tolist())
