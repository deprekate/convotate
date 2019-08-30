import os
import re
import sys
import argparse
from argparse import RawTextHelpFormatter
from argparse import ArgumentTypeError as err
from modules.pathtype import PathType

def natural_sorted(unsorted_list):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_list, key=alphanum_key)

def is_valid_file(x):
    convotate_location = os.path.dirname(os.path.dirname(__file__))
    if x in ['data/labels.pkl', 'data/ontology.txt', 'data/merged_subsystems.txt', 'data/label_indexes.pkl', 'data/base_models/'] and os.getcwd() != convotate_location:
        x = os.path.join(convotate_location, x)
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def is_empty_file(openner):
    def checker(fpath):
        if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
            raise argparse.ArgumentTypeError("output file {0} is not empty".format(fpath))
        return openner(fpath)
    return checker

def is_empty_folder(x):
    def checker(fpath):
        if os.path.exists(fpath):
            if os.path.getsize(fpath) > 0:
                pass
        else:
            raise argparse.ArgumentTypeError("output file {0} is not empty".format(fpath))
        return openner(fpath)
    return checker

class set_default_folder(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        default_folder = getattr(namespace, 'outdir')
        try:
            default_folder = default_folder%values
        except TypeError:
            pass
        setattr(namespace, 'outdir', default_folder)

def get_args():
    usage = 'model.py [-opt1, [-opt2, ...]] infile'
    parser = argparse.ArgumentParser(description='MODEL: A program to classify genes', formatter_class=RawTextHelpFormatter, usage=usage)
    parser.add_argument('infile', type=is_valid_file, help='input file in fasta format', action=set_default_folder)
    parser.add_argument('-fl', '--label_file',      action="store", type=is_valid_file, default='data/labels.pkl', dest='label_file',
                                                    help='model label reference file')
    parser.add_argument('-fo', '--ontology_file',   action="store", type=is_valid_file, default='data/ontology.txt', dest='ontology_file',
                                                    help='')
    parser.add_argument('-fm', '--merged_file',     action="store", type=is_valid_file, default='data/merged_subsystems.txt', dest='merged_file',
                                                    help='')
    parser.add_argument('-fi', '--indexes_file',    action="store", type=is_valid_file, default='data/label_indexes.pkl', dest='indexes_file',
                                                    help='')
    #parser.add_argument('-fp', '--pattern_files',   action="store", type=is_valid_file, default='data/set_patterns/', dest='pattern_files',
    #                                                help='')
    parser.add_argument('-fb', '--basemodel_files', action="store", type=is_valid_file, default='data/base_models/', dest='basemodel_files',
                                                    help='')
    parser.add_argument('-b', '--batch_size',       action="store", type=int, default=10000, dest='batch_size',
                                                    help='number of sequences to run at a time (default 10000)')
    parser.add_argument('-m', '--max_length',       action="store", type=int, default=1950, dest='max_length',
                                                    help='maximum sequence length - sequences truncated after this point (default 1950)')
    parser.add_argument('-c', '--confidence',       action="store", type=float, default=0.99, dest='confidence_threshold',
                                                    help='confidence threshold cutoff, between 0 and 1 (default 0.99)')
    parser.add_argument('-d', '--delimiter',        action="store", type=str, default='\t', dest='delimiter',
                                                    help='output delimiter character [TAB]')
    parser.add_argument('-o', '--outdir',           action  = "store",
                                                    default = "%s_CONVOTATE",
                                                    type    = PathType(exists=None, type='dir', empty=True, dash_ok=False),
                                                    help    = "where to write the output []")
    args = parser.parse_args()
    return args

class FastaFile():
    def __init__(self, file_path, chunk_size):
        self.fp = open(file_path)
        self.chunk_size = chunk_size
        self.head = ''
        self.data = ''

    def clean_dict(self, dictionary):
        if '' in dictionary:
            del dictionary['']

    def get_chunk(self):
        sequences = dict()
        for fasta_line in self.fp:
            if(fasta_line.startswith(">")):
                sequences[self.head] = self.data
                self.head = fasta_line.split()[0]
                self.data = ''
            else:
                self.data += fasta_line.replace("\n", "").upper()
            if(len(sequences) == self.chunk_size):
                self.clean_dict(sequences)
                return sequences
        sequences[self.head] = self.data
        self.clean_dict(sequences)
        self.head = ''
        self.data = ''
        return sequences

def read_fasta(fasta_filepath):
    # standard fasta file reading where a sequence is composed
    # of a header line that starts with > and a number of lines
    # with characters corresponding to the amino-acids
    fasta_sequences = dict()
    seq_head = ''
    seq_data = ''
    with open(fasta_filepath, mode="r") as fasta_file:
        for fasta_line in fasta_file:
            if(fasta_line.startswith(">")):
                fasta_sequences[seq_head] = seq_data
                seq_head = fasta_line.split()[0]
                seq_data = ''
            else:
                seq_data += fasta_line.replace("\n", "").upper()
        fasta_sequences[seq_head] = seq_data
    # get rid of empty first sequence added
    if '' in fasta_sequences:
        del fasta_sequences['']
    yield fasta_sequences
