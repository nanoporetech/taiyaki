import numpy as np


class AlphabetInfo(object):
    """ Class containing data structures summairzing an alphabet, collapsed
    alphabet and modified base long names related to a mapped signal data set.

    `alphabet` is the single letter codes to reprsent the corresponding labels
        in training data. Thus ''.join(alphabet[li] for li in labels) would
        give the sequence corresponding to labels.
    `collapse_alphabet`  are the canonical bases corresponding to the each
        value in alphabet. collapse_alphabet must be the same length as
        alphabet and the values must be a subset of the values in alphabet.
    `mod_long_names` is a long names for each non-canonical (modified) base in
        alphabet. This value is not required if there are no modified bases in
        the alphabet.

    The cat_mod model outputs bases in a specific order. This ordering groups
    modified base labels with thier corresponding canonical bases. The
    `reorder` argument to this function will perform this re-ordering, but
    should not be set for alphabets related to a mapped signal dataset.

    For example alphabet='ACGTZYXW', collapse_alphabet='ACGTCAAT' would produce
    cat_mod ordering of `AYXCZGTW`.

    - `nbase` - total number of bases (length of alphabet)
    - `ncan_base` - number of canonical bases
    - `nmod_base` - number of modified bases
    - `mod_name_conv` - dictionary to map single letter codes with long names
    - `collapse_labels` - mapping from alphabet labels to canonical
        labels. So `[collapse_alphabet[li] for li in labels]` will give
        canonical labels for a sequence of labels.
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

        Args:
            read_data (list): :class:`signal_mapping.SignalMappings` objects
            N (int): Number of reads to sample for inverse frequency estimation
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

    def compute_log_odds_weights(self, read_data, N):
        """ Compute modified base inverse frequency weights (compared to
        frequency of corresponding canonical base). Weights are intended to
        be used to scale the cat_mod model loss function to more
        equally weigh modified base observations from the training data.

        Values are the ratio of canoncial base frequency to modified base
        frequency within a sample of `read_data`.

        Output is in cat_mod model output ordering.

        Args:
            read_data (list): :class:`signal_mapping.SignalMappings` objects
            N (int): Number of reads to sample for inverse frequency estimation
        """
        N = min(N, len(read_data))
        # sample N reads
        labels = np.concatenate([
            rd.Reference for rd in np.random.choice(
                read_data, N, replace=False)])
        lab_counts = np.bincount(labels)
        if lab_counts.shape[0] < self.nbase or np.any(lab_counts == 0):
            raise NotImplementedError
        log_odds_weights = []
        for can_base in self.can_bases:
            can_lab = self.alphabet.index(can_base)
            can_mods_sum = sum(
                lab_counts[mod_lab] for mod_lab in
                np.where(self.collapse_labels == can_lab)[0][1:])
            log_odds_weights.append(can_mods_sum / lab_counts[can_lab])
            for mod_lab in np.where(self.collapse_labels == can_lab)[0][1:]:
                log_odds_weights.append(
                    lab_counts[can_lab] / lab_counts[mod_lab])
        return np.array(log_odds_weights, dtype=np.float32)

    def contains_modified_bases(self):
        return len(self.mod_long_names) > 0

    def is_compatible_model(self, network):
        flipflop_layer = network.sublayers[-1]
        if hasattr(flipflop_layer, 'alphabet'):
            return all([
                self.alphabet == flipflop_layer.alphabet,
                self.collapse_alphabet == flipflop_layer.collapse_alphabet,
                self.mod_long_names == flipflop_layer.mod_long_names,
                self.mod_name_conv == flipflop_layer.mod_name_conv,
                self.can_bases == flipflop_layer.can_bases,
                self.mod_bases == flipflop_layer.mod_bases,
                self.ncan_base == flipflop_layer.ncan_base,
                self.nmod_base == flipflop_layer.nmod_base])
        return self.nbase == flipflop_layer.nbase

    def collapse_sequence(self, sequence_with_mods):
        """ Replace modified bases in a string sequence with corresponding
        canonical bases
        """
        return sequence_with_mods.translate(self.translation_table)

    def __str__(self):
        self_str = 'canonical alphabet {}'.format(''.join(self.can_bases))
        if self.nmod_base == 0:
            self_str += ' and no modified bases'
        else:
            mod_bases_str = ', '.join(
                '{}={} (alt to {})'.format(
                    mod_b, self.mod_name_conv[mod_b], can_b)
                for mod_b, can_b in zip(self.alphabet, self.collapse_alphabet)
                if mod_b in self.mod_bases_set)
            self_str += ' with modified base(s) {}'.format(mod_bases_str)
        return self_str

    def add_ordered_info(self):
        """ Save attributes that are dependent on alphabet order.
        """
        self.collapse_labels = np.array(
            [self.alphabet.find(cb) for cb in self.collapse_alphabet],
            dtype=np.int32)
        self.can_bases = ''.join(
            [b for b in self.alphabet if b in self.can_bases_set])
        self.mod_bases = ''.join(
            [b for b in self.alphabet if b in self.mod_bases_set])

        return

    def sort_alphabet(self):
        """ Re-order alphabet to canonical grouping. Each canonical base
        followed by all modified bases associated with that canonical base.
        """
        self.collapse_alphabet, self.alphabet = map(
            lambda x: ''.join(x), zip(*sorted(zip(
                self.collapse_alphabet, self.alphabet))))
        if self.mod_long_names is not None:
            self.mod_long_names = [self.mod_name_conv[b] for b in self.alphabet
                                   if b in self.mod_bases_set]
        self.is_sorted = True

        self.add_ordered_info()

        return

    def validate_alphabet(self):
        assert len(self.alphabet) == len(self.collapse_labels), (
            'Alphabet ({}) and collapse_labels ({}) must be ' +
            'the same length.').format(self.alphabet, self.collapse_labels)
        assert len(set(self.collapse_alphabet).difference(
            self.alphabet)) == 0, (
                'All bases in collapse alphabet must occur within alphabet.')
        if self.nmod_base > 0:
            assert self.mod_long_names is not None, (
                'Must speify mod_long_names if modified bases are presnt in ' +
                'alphabet.')
            assert self.nmod_base == len(self.mod_long_names), (
                'Must provide a long name for each modified base ' +
                'included in alphabet. Found {} modified bases and ' +
                'modified base long names: "{}"').format(
                    self.nmod_base, '", "'.join(self.mod_long_names))

        return

    def parse_alphabet_info(self):
        """ Parse alphabet information that is independent of alphabet order.
        """
        self.translation_table = self.alphabet.maketrans(
            self.alphabet, self.collapse_alphabet)

        self.nbase = len(self.alphabet)
        self.can_bases_set = set(self.collapse_alphabet)
        self.mod_bases_set = set(self.alphabet).difference(self.can_bases_set)
        mod_bases = [b for b in self.alphabet if b in self.mod_bases_set]
        self.mod_name_conv = (
            None if self.mod_long_names is None else
            dict(zip(mod_bases, self.mod_long_names)))
        # should be 4
        self.ncan_base = len(self.can_bases_set)
        self.nmod_base = self.nbase - self.ncan_base

        self.add_ordered_info()

        return

    def __init__(self, alphabet, collapse_alphabet, mod_long_names=[],
                 do_reorder=False):
        """ Parse alphabet and collapse_alphabet to extract information
        required for flip-flop modeling
        :param alphabet: Alphabet corresponding to read labels including
            modified bases
        :param collapse_alphabet: Alphabet with modified bases replaced with
            thier canonical counterpart
        :param mod_long_names: Long names for each modified base found in
            alphabet
        :param do_reorder: Re-order alphabet to canonical base grouping (
            canonical base followed by associated mods). Should not be used
            when reading in signal mapped data.
        """
        self.alphabet = alphabet
        self.collapse_alphabet = collapse_alphabet
        self.mod_long_names = mod_long_names
        try:
            self.alphabet = self.alphabet.decode()
            self.collapse_alphabet = self.collapse_alphabet.decode()
        except Exception:
            pass

        self.parse_alphabet_info()
        self.validate_alphabet()

        self.is_sorted = False
        if do_reorder:
            self.sort_alphabet()

        return

    def equals(self, alphabet_info2):
        """Does alphabet_info2 describe the same alphabet as self?"""
        if self.alphabet != alphabet_info2.alphabet:
            return False
        if self.collapse_alphabet != alphabet_info2.collapse_alphabet:
            return False
        if self.mod_long_names != alphabet_info2.mod_long_names:
            return False
        return True


if __name__ == '__main__':
    NotImplementedError(
        'This is a taiyaki module and is not intended for direct use.')
