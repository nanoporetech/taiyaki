from collections import defaultdict
import numpy as np


class AlphabetInfo(object):
    """ Class to represent the information about a modeled alphabet and
    collapsed alphabet including attributes for the relation to a categorical
    modified (cat_mod) base model.

    `alphabet` is the single letter codes to reprsent the corresponding labels
        in training data. Thus ''.join(alphabet[li] for li in labels) would give
        the sequence corresponding to labels. All canonical bases should be
        present before any modified bases.
    `collapse_alphabet`  are the canonical bases corresponding to the each value
        in alphabet. collapse_alphabet must be the same length as alphabet and
        the values must be a subset of the values in alphabet.

    The cat_mod model outputs bases in an different order than provided by
    the `alphabet` and `collapse_alphabet` values. This ordering groups modified
    base labels with thier corresponding canonical bases.

    For example alphabet='ACGTZYXW', collapse_alphabet='ACGTCAAT' would produce
    cat_mod ordering of `AYXCZGTW`.

    - `nbase` contains the number of bases to be modeled (length of alphabet)
    - `collapse_labels` contains a mapping from alphabet labels to canonical
        labels. So `[collapse_alphabet[li] for li in labels]` will give
        canonical labels for a sequence of labels.
    - `ncan_base` is the number of canonical bases
    - `nmod_base` is the number of modified bases
    - `can_mods_offsets` is the offset for each canonical base in the
        cat_mod model output
    - `mod_labels` is the modified base label for each value in alphabet. This
        value is `0` for each canonical base and is incremented by one for each
        modified label conditioned on the canonical base. This is in alphabet
        order and NOT cat_mod order. Using the example alphabet above,
        mod_labels would be `[0, 0, 0, 0, 1, 1, 2, 1]`
    - `translation_table` contains a string.transtable to convert string
        sequence to canonical base values.
    """
    def compute_mod_inv_freq_weights(self, read_data, N):
        """ Compute modified base inverse frequency weights (compared to
        frequency of corresponding canonical base). Weights are intended to
        be used to scale the cat_mod model loss function to more
        equally weigh modified base observations from the training data.

        Values are the ratio of canoncial base frequency to modified base
        frequency within a sample of `read_data`.

        Output is in cat_mod model output ordering.

        :param read_data: list of ReadData objects, as return from
            mapped_signal_files.HDF5.get_multiple_reads
        :param N: Number of reads to sample for inverse frequency estimation
        """
        N = min(N, len(read_data))
        # sample N reads
        labels = np.concatenate([
            rd['Reference'] for rd in np.random.choice(
                read_data, N, replace=False)])
        lab_counts = np.bincount(labels)
        if lab_counts.shape[0] < self.nbase or np.any(lab_counts == 0):
            raise NotImplementedError
        mod_inv_weights = []
        for can_lab in range(self.ncan_base):
            mod_inv_weights.append(1.0)
            for mod_lab in np.where(self.collapse_labels == can_lab)[0][1:]:
                mod_inv_weights.append(
                    lab_counts[can_lab] / lab_counts[mod_lab])
        return np.array(mod_inv_weights, dtype=np.float32)

    def collapse_sequence(self, sequence_with_mods):
        """ Replace modified bases in a string sequence with corresponding
        canonical bases
        """
        return sequence_with_mods.translate(self.translation_table)

    def _parse_coll_alph(self):
        """ Parse alphabet and collapse alphabet and save attributes required
        to initialize a categorical modified base model.
        """
        self.nbase = len(self.alphabet)

        self.collapse_labels = np.array(
            [self.alphabet.find(cb) for cb in self.collapse_alphabet],
            dtype=np.int32)
        # should be 4
        self.ncan_base = len(set(self.collapse_labels))
        self.nmod_base = len(set(self.alphabet)) - self.ncan_base
        assert (set(self.alphabet[:self.ncan_base]) ==
                set(self.collapse_alphabet)), (
                    'All canonical bases in alphabet must appear before ' +
                    'any modified bases.')
        # record the canonical base group indices for modified base
        # categorical outputs from GlobalNormFlipFlopCatMod layer
        self.can_mods_offsets = np.cumsum([0] + [
            self.collapse_alphabet.count(can_b)
            for can_b in self.alphabet[:self.ncan_base]]).astype(np.int32)
        # create table of mod label offsets within each canonical label group
        mod_labels = [0, ] * self.ncan_base
        can_grouped_mods = defaultdict(int)
        for can_lab in self.collapse_labels[self.ncan_base:]:
            can_grouped_mods[can_lab] += 1
            mod_labels.append(can_grouped_mods[can_lab])
        self.mod_labels = np.array(mod_labels)
        self.max_can_grp_nmod = self.mod_labels.max()

        return

    def __init__(self, alphabet, collapse_alphabet):
        """ Parse alphabet and collapse_alphabet to extract information
        required for flip-flop modeling
        :param alphabet: Alphabet corresponding to read labels including
            modified bases
        :param collapse_alphabet: Alphabet with modified bases replaced with
            thier canonical counterpart
        """
        self.alphabet = alphabet
        self.collapse_alphabet = collapse_alphabet
        try:
            self.alphabet = self.alphabet.decode()
            self.collapse_alphabet = self.collapse_alphabet.decode()
        except:
            pass
        self.translation_table = alphabet.maketrans(alphabet, collapse_alphabet)

        self._parse_coll_alph()

        return


if __name__ == '__main__':
    NotImplementedError(
        'This is a taiyaki module and is not intended for direct use.')
