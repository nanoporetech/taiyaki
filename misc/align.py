#!/usr/bin/env python3
import argparse
import csv
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pysam
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
import subprocess
import sys
import traceback
from taiyaki.cmdargs import AutoBool, proportion


parser = argparse.ArgumentParser(
    description='Align reads to reference and output accuracy statistics',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# TODO: add several named commonly used values for bwa_mem_args
parser.add_argument('--bwa_mem_args', metavar='args', default='-k14 -W20 -r10 -t 16 -A 1 -B 2 -O 2 -E 1',
                    help="Command line arguments to pass to bwa mem")
parser.add_argument('--coverage', metavar='proportion', default=0.6, type=proportion,
                    help='Minimum coverage')
parser.add_argument('--data_set_name', default=None,
                    help="Data set name. If not set file name is used.")
parser.add_argument('--figure_format', default="png",
                    help="Figure file format.")
parser.add_argument('--fill', default=True, action=AutoBool,
                    help='Fill basecall quality histogram with color')
parser.add_argument('--show_median', default=False, action=AutoBool,
                    help='Show median in a histogram plot')
parser.add_argument('--reference', default=None,
                    help="Reference sequence to align against")

parser.add_argument('files', metavar='input', nargs='+',
                    help="One or more files containing query sequences")


STRAND = {0: '+',
          16: '-'}

QUANTILES = [5, 25, 50, 75, 95]


def call_bwa_mem(fin, fout, genome, clargs=''):
    """Call bwa aligner using the subprocess module

    :param fin: input sequence filename
    :param fout: filename for the output sam file
    :param genome: path to reference to align against
    :param clargs: optional command line arguments to pass to bwa as a string

    :returns: stdout of bwa command

    :raises: subprocess.CalledProcessError
    """
    command_line = "bwa mem {} {} {} > {}".format(clargs, genome, fin, fout)
    try:
        output = subprocess.check_output(command_line,
                                         stderr=subprocess.STDOUT,
                                         shell=True,
                                         universal_newlines=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(
            "Error calling bwa, exit code {}\n".format(e.returncode))
        sys.stderr.write(e.output + "\n")
        raise
    return output


def samacc(sam, min_coverage=0.6):
    """Read alignments from sam file and return accuracy metrics

    :param sam: filename of input sam file
    :min_coverage: alignments are filtered by coverage

    :returns: list of row dictionaries with keys:
        reference: reference name
        query: query name
        reference_start: first base of reference match
        reference_end: last base of reference match
        strand: + or -
        match: number of matches
        mismatch: number of mismatches
        insertion: number of insertions
        deletion: number of deletions
        coverage: query alignment length / query length
        id: identity = sequence matches / alignment matches
        accuracy: sequence matches / alignment length
    """
    res = []
    with pysam.Samfile(sam, 'r') as sf:
        ref_name = sf.references
        for read in sf:
            if read.flag != 0 and read.flag != 16:
                continue

            coverage = float(read.query_alignment_length) / read.query_length
            if coverage < min_coverage:
                continue

            bins = np.zeros(9, dtype='i4')
            for flag, count in read.cigar:
                bins[flag] += count

            tags = dict(read.tags)
            alnlen = np.sum(bins[:3])
            mismatch = tags['NM']
            correct = alnlen - mismatch
            readlen = bins[0] + bins[1]
            perr = min(0.75, float(mismatch) / readlen)
            pmatch = 1.0 - perr

            entropy = pmatch * np.log2(pmatch)
            if mismatch > 0:
                entropy += perr * np.log2(perr / 3.0)

            row = OrderedDict([
                ('reference', ref_name[read.reference_id]),
                ('query', read.qname),
                ('strand', STRAND[read.flag]),
                ('reference_start', read.reference_start),
                ('reference_end', read.reference_end),
                ('match', bins[0]),
                ('mismatch', mismatch),
                ('insertion', bins[1]),
                ('deletion', bins[2]),
                ('coverage', coverage),
                ('id', float(correct) / float(bins[0])),
                ('accuracy', float(correct) / alnlen),
                ('information', bins[0] * (2.0 + entropy))
            ])
            res.append(row)
    return res


def acc_plot(acc, mode, median, fill, title):
    """Plot accuracy histogram

    :param acc_dat: list of row dictionaries of basecall accuracy data
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


def summary(acc_dat, data_set_name, fill, show_median):
    """Create summary report and plots for accuracy statistics

    :param acc_dat: list of row dictionaries of read accuracy metrics

    :returns: (report string, figure handle, axes handle)
    """
    if len(acc_dat) == 0:
        res = """*** Summary report for {} ***
No sequences mapped
""".format(data_set_name)
        return res, None, None

    acc = np.array([r['accuracy'] for r in acc_dat])
    ciscore = np.array([r['information'] for r in acc_dat])
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
        except:
            sys.stderr.write("Mode computation failed - da or opt")
            mode = 0
    else:
        mode = acc[0]

    qstring1 = ''.join(['{:<11}'.format('Q' + str(q))
                        for q in QUANTILES]).strip()
    quantiles = [v for v in np.percentile(acc, QUANTILES)]
    qstring2 = '    '.join(['{:.5f}'.format(v) for v in quantiles])

    if show_median:
        median = np.median(acc)
    else:
        median = None

    a90 = (acc > 0.9).mean()
    n_gt_90 = (acc > 0.9).sum()
    nmapped = len(set([r['query'] for r in acc_dat]))

    res = """*** Summary report for {} ***
Number of mapped reads:  {}
Mean accuracy:  {:.5f}
Mode accuracy:  {:.5f}
Accuracy quantiles:
  {}
  {}
Proportion with accuracy >90%:  {:.5f}
Number with accuracy >90%:  {}
CIscore (Mbits): {:.5f}
""".format(data_set_name, nmapped, mean, mode, qstring1, qstring2, a90, n_gt_90, sum(ciscore) / 1e6)
    plot_title = "{} (n = {})".format(data_set_name, nmapped)
    f, ax = acc_plot(acc, mode, median, fill, plot_title)
    return res, f, ax


def main():
    args = parser.parse_args()

    exit_code = 0
    for fn in args.files:
        try:
            prefix, suffix = os.path.splitext(fn)
            samfile = prefix + '.sam'
            samaccfile = prefix + '.samacc'
            summaryfile = prefix + '.summary'
            graphfile = prefix + '.' + args.figure_format

            # align sequences to reference
            if args.reference and not suffix == '.sam':
                sys.stdout.write("Aligning {}...\n".format(fn))
                bwa_output = call_bwa_mem(
                    fn, samfile, args.reference, args.bwa_mem_args)
                sys.stdout.write(bwa_output)

            # compile accuracy metrics
            acc_dat = samacc(samfile, min_coverage=args.coverage)
            if len(acc_dat) > 0:
                with open(samaccfile, 'w') as fs:
                    fields = list(acc_dat[0].keys())
                    writer = csv.DictWriter(
                        fs, fieldnames=fields, delimiter=' ')
                    writer.writeheader()
                    for row in acc_dat:
                        writer.writerow(row)

            # write summary file and plot
            data_set_name = fn if args.data_set_name is None else args.data_set_name
            report, f, ax = summary(
                acc_dat, data_set_name, args.fill, args.show_median)
            if f is not None:
                f.savefig(graphfile)
            sys.stdout.write('\n' + report + '\n')
            with open(summaryfile, 'w') as fs:
                fs.writelines(report)
        except:
            sys.stderr.write(
                "{}: something went wrong, skipping\n\n".format(fn))
            sys.stderr.write("Traceback:\n\n{}\n\n".format(
                traceback.format_exc()))
            exit_code = 1

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
