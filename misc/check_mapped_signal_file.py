#!/usr/bin/env python3
# Sanity checks and metrics report for mapped signal file

import sys
import argparse
from collections import Counter

import numpy as np

from taiyaki.mapped_signal_files import MappedSignalReader
from taiyaki.maths import MAD_SD_FACTOR, med_mad


SINGLE_LETTER_CODE = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'B': 'CGT', 'D': 'AGT', 'H': 'ACT',
    'K': 'GT', 'M': 'AC', 'N': 'ACGT', 'R': 'AG', 'S': 'CG', 'V': 'ACG',
    'W': 'AT', 'Y': 'CT'}


def get_parser():
    parser = argparse.ArgumentParser(
        description='Produce report .')
    parser.add_argument('input', help='Mapped signal file.')
    parser.add_argument(
        '--num_reads', type=int,
        help='Number of reads to process.')
    parser.add_argument(
        '--motif', nargs=2, action='append',
        help='Sequence motif to search for modbases. Motif represnted ' +
        'by canonical base motif and relative position within to search for ' +
        'modified bases. Example: `--motif CHG 0`')
    parser.add_argument(
        '--num_chunks', type=int, default=500,
        help='Number of chunks to select. Set to "None" to skip chunk ' +
        'metrics. Default: %(default)d')
    parser.add_argument(
        '--chunk_len', type=int, default=5000,
        help='Length of chunks to sample. Default: %(default)d')
    parser.add_argument(
        '--chunk_percentiles', type=int,
        default=[0, 50, 90, 95, 99, 99.9, 100], nargs='+',
        help='Percentiles to report for chunk metrics. Default: %(default)s')
    parser.add_argument(
        '--output', help='Output filename. Default: stdout')
    return parser


def main():
    args = get_parser().parse_args()
    out_fp = sys.stdout if args.output is None else open(args.output, 'w')
    sys.stderr.write('* Reading data from file\n')
    out_fp.write('*' * 10 + ' General Metrics ' + '*' * 10 + '\n')
    with MappedSignalReader(args.input) as msr:
        alphabet_info = msr.get_alphabet_information()
        out_fp.write('Alphabet: {}\n'.format(str(alphabet_info)))
        read_ids = msr.get_read_ids()
        out_fp.write('Total reads: {}\n'.format(len(read_ids)))
        if args.num_reads is not None:
            np.random.shuffle(read_ids)
            read_ids = read_ids[:args.num_reads]
        reads = list(msr.reads(read_ids))

    sys.stderr.write('* Computing sanity check metrics\n')
    out_fp.write('\n\n' + '*' * 10 + ' Sanity Checks ' + '*' * 10 + '\n')
    current_meds = np.array([np.median(read.get_current()) for read in reads])
    out_fp.write((
        'Median of medians of normalized signal: {:.6f} (should usually ' +
        'be close to 0)\n').format(np.median(current_meds)))
    current_mads = np.array([np.median(np.abs(read.get_current() - r_med))
                             for read, r_med in zip(reads, current_meds)])
    out_fp.write((
        'Median of MADs of normalized signal: {:.6f} (should usually ' +
        'be close to {:.6f})\n').format(
            np.median(current_mads), 1 / MAD_SD_FACTOR))
    all_seqs = np.concatenate([read.Reference for read in reads])
    seq_counts = np.bincount(all_seqs)
    total_bases = np.sum(seq_counts)
    out_fp.write(
        'Global sequence composition:\n{: >11}{: >11}{: >11}\n'.format(
            'base', 'count', 'percent') + '\n'.join(
                '{: >11}{:11.0f}{:11.4f}'.format(*base_metrics)
                for base_metrics in zip(
                    alphabet_info.alphabet, seq_counts,
                    100 * seq_counts / total_bases)) + '\n')

    if args.motif is not None:
        sys.stderr.write('* Computing motif metrics (this may take a ' +
                         'while to complete)\n')
        out_fp.write('\n\n' + '*' * 10 + ' Motif Metrics ' + '*' * 10 + '\n')

        def match_motif(seq, motif):
            for base, motif_pos_bases in zip(seq, motif):
                if base not in motif_pos_bases:
                    return False
            return True

        motifs = [
            ([np.concatenate([np.where([m_base == a_base for a_base in
                                        alphabet_info.collapse_alphabet])[0]
                              for m_base in SINGLE_LETTER_CODE[m_raw_base]])
              for m_raw_base in motif], int(rel_pos))
            for motif, rel_pos in args.motif]
        motif_mod_counts = [[] for _ in range(len(motifs))]
        for read in reads:
            for motif_i, (motif, rel_pos) in enumerate(motifs):
                for offset in range(read.Reference.shape[0] - len(motif)):
                    if match_motif(
                            read.Reference[offset:offset + len(motif)], motif):
                        motif_mod_counts[motif_i].append(
                            read.Reference[offset + rel_pos])

        for (raw_motif, rel_pos), m_mod_counts in zip(
                args.motif, motif_mod_counts):
            rel_pos = int(rel_pos)
            motif_mod_counts = np.bincount(m_mod_counts)
            total_mod_bases = np.sum(motif_mod_counts)
            out_fp.write(
                ('{} Motif Modified Base Counts:\n{: >11}{: >11}' +
                 '{: >11}\n').format(raw_motif, 'base', 'count', 'percent') +
                '\n'.join('{: >11}{:11.0f}{:11.4f}'.format(
                    raw_motif[:rel_pos] + base + raw_motif[rel_pos + 1:],
                    count, pct) for base, count, pct in zip(
                        alphabet_info.alphabet, motif_mod_counts,
                        100 * motif_mod_counts / total_mod_bases)
                    if count > 0) + '\n\n')

    sys.stderr.write('* Computing read metrics\n')
    out_fp.write('\n' + '*' * 10 + ' Read Metrics ' + '*' * 10 + '\n')
    read_lens = np.array([read.reflen for read in reads])
    out_fp.write(('Median read length: {}\n').format(np.median(read_lens)))
    sig_lens = np.array([read.siglen for read in reads])
    out_fp.write(('Median signal length: {}\n').format(
        np.median(sig_lens)))

    if args.num_chunks is None:
        return
    sys.stderr.write('* Computing chunk metrics\n')
    out_fp.write('\n\n' + '*' * 10 + ' Chunk Metrics ' + '*' * 10 + '\n')
    chunks, rej_res = [], []
    while len(chunks) < args.num_chunks:
        chunk = np.random.choice(reads, 1)[0].get_chunk_with_sample_length(
            args.chunk_len)
        if chunk.accepted:
            chunks.append(chunk)
        else:
            rej_res.append(chunk.reject_reason)
    if len(rej_res) > 0:
        out_fp.write(
            'Chunk rejection reasons:\n{: >16}{: >16}\n'.format(
                'Reject Reason', 'Num. Chunks') +
            '\n'.join('{: >16}{: >16}'.format(*x)
                      for x in Counter(rej_res).most_common()) + '\n\n')
    else:
        out_fp.write('All chunks passed filters\n\n')
    mean_dwells = np.array([chunk.mean_dwell for chunk in chunks])
    max_dwells = np.array([chunk.max_dwell for chunk in chunks])
    # report in MAD units for direct use in command line parameter usage
    median_meandwell, mad_meandwell = med_mad(mean_dwells)
    out_fp.write(
        ('Chunk dwell distribution (standard units for direct use with ' +
         '--filter_max_dwell and --filter_mean_dwell):\n' +
         '{: >15}{: >15}{: >15}{: >15}{: >15}\n').format(
             'percentile', 'mean_dwell', 'mean_std_units',
             'max_dwell', 'max_std_units') +
        '\n'.join('{:15.2f}{:15.2f}{:15.2f}{:15.2f}{:15.2f}'.format(
            *pctl_metrics) for pctl_metrics in zip(
                args.chunk_percentiles,
                np.percentile(mean_dwells, args.chunk_percentiles),
                np.percentile(mean_dwells - median_meandwell / mad_meandwell,
                              args.chunk_percentiles),
                np.percentile(max_dwells, args.chunk_percentiles),
                np.percentile(max_dwells / median_meandwell,
                              args.chunk_percentiles))) + '\n')

    if args.output is not None:
        out_fp.close()


if __name__ == '__main__':
    main()
