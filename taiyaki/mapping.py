# Defines class to represent a read - used in chunkifying
# class also includes data and methods to deal with mapping table
# Also defines iterator giving all f5 files in directory, optionally using a read list

import numpy as np
import sys
from taiyaki import mapped_signal_files


class Mapping:
    """
    Represents a mapping between a signal and a reference, with attributes
    including the signal and the reference.

    We use the trimming parameters from the signal object to set limits on
    outputs (including chunks).
    """

    def __init__(self, signal, signalpos_to_refpos, reference, verbose=False):
        """
        param: signal                  : a Signal object
        param: signalpos_to_refpos     : a numpy integer array of the same length as signal.untrimmed_dacs
                                         where signalpos_to_refpos[n] is the location in the reference that
                                         goes with location n in signal.untrimmed_dacs. A (-1) in the
                                         vector indicates no association with that signal location.
        param: reference               : a bytes array or str containing the reference.
                                         (Note that this is converted into a str for use in the class)
        param: verbose                 : Print information about newly constructed mapping object to stdout
                                          (useful when writing new factory functions)
       """

        self.siglen = len(signal.untrimmed_dacs)
        self.reflen = len(reference)
        if self.siglen != len(signalpos_to_refpos):
            raise Exception('Mapping: mapping vector is different length from untrimmed signal')
        self.signal = signal
        self.signalpos_to_refpos = signalpos_to_refpos
        if isinstance(reference, str):
            self.reference = reference
            if verbose:
                print("Created reference from str")
        else:
            try:
                self.reference = reference.decode("utf-8")
                if verbose:
                    print("Created reference from bytes")
            except:
                if verbose:
                    print("REFERENCE NOT SET")
                raise Exception('Mapping: reference cannot be decoded as string or bytes array')
        if verbose:
            print("Signal constructor finished.")
            print("Signal (trimmed) length:", self.signal.trimmed_length)
            print("Mapping vector length:", self.siglen)
            print("reference length", self.reflen)

    @property
    def trimmed_length(self):
        """Trimmed length of the signal in samples. Convenience function,
        same as signal.trimmed_length"""
        return self.signal.trimmed_length

    def mapping_limits(self, mapping_margin=0):
        """Calculate start and (exclusive) endpoints for the signal
        so that only the mapped portion of the signal is included.

        After finding endpoints for the mapped region, trim off another
        mapping_margin samples from both ends.

        If resulting region is empty, then return start and end points
        so that nothing is left.

        Take no notice at all of the signal's trimming parameters
        (but see the function mapping_limits_with_signal_trim())

        param: mapping_margin  : extra number of samples to trim off both ends

        returns: (startsample, endsample_exc)
                 where self.signal.untrimmed_dacs[startsample:endsample_exc]
                 is the region that is included in the mapping.
        """
        firstmapped, lastmapped = -1, -1
        for untrimmed_dacloc, refloc in enumerate(self.signalpos_to_refpos):
            if refloc >= 0:
                if firstmapped < 0:
                    firstmapped = untrimmed_dacloc
                lastmapped = untrimmed_dacloc
        if firstmapped >= 0:  # If we have found any mapped locations
            startloc = firstmapped + mapping_margin
            endloc = lastmapped + 1 - mapping_margin
            if startloc <= endloc - 1:
                return startloc, endloc
        # Trim to leave nothing
        return 0, 0

    def mapping_limits_with_signal_trim(self, mapping_margin=0):
        """Calculate mapping limits as in the method mapping_limits()
        and then find start and end points for the intersection of
        the mapped region with the trimmed region of the signal.
        Note mapping_margin is applied to the mapped region before
        the signal trim is taken into account.
        """
        mapstart, mapend_exc = self.mapping_limits(mapping_margin)
        start = max(mapstart, self.signal.signalstart)
        end_exc = min(mapend_exc, self.signal.signalend_exc)

        if start < end_exc:
            return start, end_exc
        else:
            return 0, 0

    @classmethod
    def from_remapping_path(_class, signal, sigtoref_downsampled, reference, stride=1, signalstart=None):
        """
        Construct Mapping object based on downsampled mapping information
        (rather than just copying sigtoref).
        Inputs:
            sigtoref =  a numpy int vector where sigtoref_downsampled[k] is the
                        location in the reference of the base starting at
                        untrimmed_dacs[k*stride-1+signalstart]
            reference = a string containing the reference
        By default, we assume that signalstart is self.signalstart, the trim start
        stored within the signal object.

        There is a bit of freedom in where to put the signal locations
        because of transition weights
        and downsampling. We use a picture like this, shown for the case
        stride = 2


        signal                [0]   [1]   [2]   [3]   [4]   [5]
        blocks                ---------   ---------   ---------
        trans weights            [0]         [1]         [2]
        sigtoref           [0]         [1]         [2]         [3]
        mapping to signal             /           /           /
        signal                [0]   [1]   [2]   [3]   [4]   [5]

        in other words sigtoref[n] maps to signal[stride*n-1]
        Note that the very first element of the sigtoref vector that comes
        from remapping is ignored with this choice

        """

        if signalstart is None:
            signalstart_used = signal.signalstart
        else:
            signalstart_used = signalstart

        # Create null dacstoref ranging of the full (not downsampled) range of locations
        # and fill in the locations determined by the mapping
        # -1 means nothing  associated with this location
        fullsigtoref = np.full(len(signal.untrimmed_dacs), -1, dtype=np.int32)
        # Calculate locations in signal associated with each location in the input sigtoref
        # There is a bit of freedom here: see the docstring
        siglocs = np.arange(len(sigtoref_downsampled), dtype=np.int32) * stride - 1 + signalstart_used
        # We keep only signal locations that are between 0 and len(untrimmed_dacs)
        f = (siglocs >= 0) & (siglocs < len(fullsigtoref))
        # Put numbers in

        # print("Len(fullsigtoref)=",len(fullsigtoref))
        # print("Max(siglocs[f])=",np.max(siglocs[f]))

        fullsigtoref[siglocs[f]] = sigtoref_downsampled[f]

        return _class(signal, fullsigtoref, reference)

    def get_reftosignal(self):
        """Return integer vector reftosig,  mapping reference to signal.

        length of reftosig returned is (1+reflen), where reflen = len(self.reference)

        reftosig[n] is the location in the untrimmed dacs where the base at
        self.reference[n] starts.

        The last element, reftosig[reflen] is consistent with this scheme: it is
        (1 + (last location in untrimmed dacs))

        if the start of the reference is not mapped, then reftosig will begin
        with a sequence of (-1)s

        if the end of the reference is not mapped, then reftosig will end with
         ... f, f, f, f]  where f is the last mapped location in the signal.
        """
        valid_sig_to_ref_idxs = np.where(
            self.signalpos_to_refpos != -1)[0].astype(np.int32)
        # if the full read is clipped return an array of negative ones
        if len(valid_sig_to_ref_idxs) == 0:
            return -1 * np.ones(self.reflen + 1, dtype=np.int32)

        valid_sig_to_ref = self.signalpos_to_refpos[valid_sig_to_ref_idxs]
        move_pos = np.concatenate([[1,], np.diff(valid_sig_to_ref)])
        ref_to_sig = np.repeat(valid_sig_to_ref_idxs, move_pos)
        # add the end of the last mapped position
        ref_to_sig = np.concatenate([
            ref_to_sig,
            np.array([valid_sig_to_ref_idxs[-1] + 1,], dtype=np.int32)])

        # Insert the right number of -1s to get to the beginning of the
        # mapped region
        if valid_sig_to_ref[0] > 0:
            ref_to_sig = np.concatenate([
                -1 * np.ones(valid_sig_to_ref[0], dtype=np.int32),
                ref_to_sig])
        if self.reflen + 1 > len(ref_to_sig):
            ref_to_sig = np.append(ref_to_sig, (self.siglen + 1) * np.ones(
                self.reflen + 1 - len(ref_to_sig), dtype=np.int32))

        return ref_to_sig

    def add_integer_reference(self, alphabet):
        self.integer_reference = np.array([
            alphabet.index(i) for i in self.reference], dtype=np.int16)
        return

    def get_read_dictionary(self, shift, scale, read_id, check=True):
        """Return a read dictionary of the sort specified in mapped_signal_files.Read.
        Note that we return the dictionary, not the object itself.
        That's because this method is used inside worker processes which
        need to pass their results out through the pickling mechanism in imap_mp.
        Apply error checking if check = True, and raise an Exception if it fails"""
        readDict = {
            'shift_frompA': float(shift),
            'scale_frompA': float(scale),
            'range': float(self.signal.range),
            'offset': float(self.signal.offset),
            'digitisation': float(self.signal.digitisation),
            'Dacs': self.signal.untrimmed_dacs.astype(np.int16),
            'Ref_to_signal': self.get_reftosignal(),
            'Reference': self.integer_reference,
            'read_id': read_id
        }
        if check:
            readObject = mapped_signal_files.Read(readDict)
            checkstring = readObject.check()
            if checkstring != "pass":
                print("Channel info:")
                for k, v in self.signal.channel_info.items():
                    print("   ", k, v)
                print("Read attributes:")
                for k, v in self.signal.read_attributes.items():
                    print("   ", k, v)
                sys.stderr.write(
                    "Read object for {} to place in file doesn't pass tests:\n {}\n".format(read_id, checkstring))
                raise Exception("Read object failed error checking in mapping.get_read_dictionary()")
        return readDict

    def to_ssv(self, filename, appendRef=True):
        """Saves untrimmed dac, signal and mapping to a
        space-separated file. If appendRef then add
        the reference on the end, starting with a #"""
        with open(filename, "w") as f:
            f.write("dac signal dactoref\n")
            for dac, refpos in zip(self.signal.untrimmed_dacs, self.signalpos_to_refpos):
                sig = (dac + self.signal.offset) * self.signal.range / self.signal.digitisation
                f.write(str(dac) + " " + str(sig) + " " + str(refpos) + "\n")
            if appendRef:
                f.write("#" + self.reference)
