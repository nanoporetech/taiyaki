#!/usr/bin/env python3
import argparse
from collections import namedtuple
import os
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt
import pysam
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

from taiyaki.cmdargs import AutoBool, proportion


ACC_METRICS = namedtuple('ACC_METRICS', (
    'reference', 'query', 'strand', 'reference_start', 'reference_end',
    'match', 'mismatch', 'insertion', 'deletion', 'coverage', 'id', 'accuracy',
    'information'))

DEFAULT_QUANTILES = [5, 25, 50, 75, 95]

PLOT_DO_FILL = True

INVALID_SUMM = """*** Summary report for {} ***
No sequences mapped
"""
VALID_SUMM = """*** Summary report for {} ***
Number of mapped reads:  {}
Mean accuracy:  {:.5f}
Mode accuracy:  {:.5f}
Accuracy quantiles:
  {}
  {}
Proportion with accuracy >90%:  {:.5f}
Number with accuracy >90%:  {}
CIscore (Mbits): {:.5f}
"""


def get_parser():
    parser = argparse.ArgumentParser(
        description='Align reads to reference and output accuracy statistics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--coverage', metavar='proportion', default=0.6, type=proportion,
        help='Minimum coverage')
    parser.add_argument(
        '--data_name', default=None,
        help="Data name. If not set file name is used.")
    parser.add_argument(
        '--figure_format', default="png",
        help="Figure file format.")
    parser.add_argument(
        '--show_median', default=False, action=AutoBool,
        help='Show median in a histogram plot')
    parser.add_argument(
        '--output_text', default=True, action=AutoBool,
        help='Output per-read text report.')
    parser.add_argument(
        '--output_plot', default=True, action=AutoBool,
        help='Output accuracy distribution plot(s).')
    parser.add_argument(
        '--quantiles', type=int, default=DEFAULT_QUANTILES, nargs='+',
        help='Quantiles to report in summary. Default: %(default)s')

    parser.add_argument(
        'files', metavar='input', nargs='+',
        help="One or more alignment files in SAM/BAM/CRAM format.")

    return parser


def compute_mismatch(read, ref_fa):
    # TODO implement extraction of reference sequence and alignment parsing
    # when NM tag is not available.
    raise NotImplementedError(
        'Alignment mismatch counting currently requires NM flag.')


def samacc(align_fn, min_coverage=0.6):
    """Read alignments from sam/bam/cram file and return accuracy metrics

    :param sam: filename of input alignment file
    :min_coverage: alignments are filtered by coverage

    :returns: list of ACC_METRICS namedtuples containing accuracy metrics for
        each valid alignment.
    """
    res = {}
    with pysam.AlignmentFile(align_fn, 'r') as sf:
        for read in sf.fetch(until_eof=True):
            if read.flag != 0 and read.flag != 16:
                continue
            coverage = float(read.query_alignment_length) / read.query_length
            if coverage < min_coverage:
                continue

            bins = np.zeros(9, dtype='i4')
            for flag, count in read.cigar:
                bins[flag] += count

            alnlen = np.sum(bins[:3])
            try:
                mismatch = read.get_tag('NM')
            except KeyError:
                mismatch = compute_mismatch(read, None)
            correct = alnlen - mismatch
            readlen = bins[0] + bins[1]
            perr = min(0.75, float(mismatch) / readlen)
            pmatch = 1.0 - perr
            accuracy = float(correct) / alnlen

            entropy = pmatch * np.log2(pmatch)
            if mismatch > 0:
                entropy += perr * np.log2(perr / 3.0)

            if (read.query not in res
                or res[read.query].accuracy < accuracy):
                res[read.query] = ACC_METRICS(
                    reference=read.reference_name,
                    query=read.query_name,
                    strand='-' if read.is_reverse else '+',
                    reference_start=read.reference_start,
                    reference_end=read.reference_end,
                    match=bins[0],
                    mismatch=mismatch,
                    insertion=bins[1],
                    deletion=bins[2],
                    coverage=coverage,
                    id=float(correct) / float(bins[0]),
                    accuracy=accuracy,
                    information=bins[0] * (2.0 + entropy)))

    return list(res.values())


def acc_plot(acc, mode, median, title, fill=PLOT_DO_FILL):
    """Plot accuracy histogram

    :param acc_dat: numpy array of accuracy values
    :param title: plot title

    :returns: (figure handle, axes handle)
    """
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.hist(acc, bins=np.arange(0.65, 1.0, 0.01), fill=fill)
    ax.set_xlim(0.65, 1)
    _, ymax = ax.get_ylim()
    ax.plot([mode, mode], [0, ymax], 'r--')
    if median:
        ax.plot([median, median], [0, ymax], 'b--')
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    return f, ax


def summary(acc_dat, data_name, show_median, quants):
    """Create summary report and plots for accuracy statistics

    :param acc_dat: list of row dictionaries of read accuracy metrics

    :returns: (report string, figure handle, axes handle)
    """
    if len(acc_dat) == 0:
        res = INVALID_SUMM.format(data_name)
        return res, None, None

    acc = np.array([r.accuracy for r in acc_dat])
    ciscore = np.array([r.information for r in acc_dat])
    mean = acc.mean()

    if len(acc) > 1:
        try:
            da = gaussian_kde(acc)
            optimization_result = minimize_scalar(
                lambda x: -da(x), bounds=(0, 1), method='Bounded')
            if optimization_result.success:
                try:
                    mode = optimization_result.x[0]
                except IndexError:
                    mode = optimization_result.x
            else:
                sys.stderr.write("Mode computation failed")
                mode = 0
        except Exception:
            sys.stderr.write("Mode computation failed - da or opt")
            mode = 0
    else:
        mode = acc[0]

    qstring1 = ''.join(
        ('Q{:<11}'.format(q) for q in quants)).strip()
    quantiles = [v for v in np.percentile(acc, quants)]
    qstring2 = '    '.join(['{:.5f}'.format(v) for v in quantiles])

    if show_median:
        median = np.median(acc)
    else:
        median = None

    a90 = (acc > 0.9).mean()
    n_gt_90 = (acc > 0.9).sum()
    nmapped = len(set([r.query for r in acc_dat]))

    res = VALID_SUMM.format(
        data_name, nmapped, mean, mode, qstring1, qstring2, a90, n_gt_90,
        sum(ciscore) / 1e6)
    plot_title = "{} (n = {})".format(data_name, nmapped)
    f, ax = acc_plot(acc, mode, median, plot_title)

    return res, f, ax


def main(args):
    exit_code = 0
    for fn in args.files:
        try:
            prefix, suffix = os.path.splitext(fn)

            # compile accuracy metrics
            acc_dat = samacc(fn, min_coverage=args.coverage)
            if args.output_text and len(acc_dat) > 0:
                with open(prefix + '.samacc', 'w') as fs:
                    fs.write(' '.join(ACC_METRICS._fields) + '\n')
                    fs.write(
                        '\n'.join((' '.join(map(str, r_acc))
                                   for r_acc in acc_dat)) + '\n')

            # write summary file and plot
            data_name = fn if args.data_name is None else args.data_name
            report, f, ax = summary(
                acc_dat, data_name, args.show_median, args.quantiles)
            if args.output_plot and f is not None:
                f.savefig(prefix + '.' + args.figure_format)
            sys.stdout.write('\n' + report + '\n')
            with open(prefix + '.summary', 'w') as fs:
                fs.writelines(report)
        except Exception:
            sys.stderr.write(
                "{}: something went wrong, skipping\n\n".format(fn))
            sys.stderr.write(
                "Traceback:\n\n{}\n\n".format(traceback.format_exc()))
            exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main(get_parser().parse_args())
