import sys
import pickle as pk

import modules.helper_functions as helpers
import modules.protein_function_hierarchy as ph

def main(argv):

    cl_args = helpers.get_args()

    # we should check the input file format a little before we spend time doing the rest
    #input_sequences = helpers.read_fasta(cl_args.infile);
    #f = open(cl_args.infile, mode="r")

    hi = ph.HierarchicalProteinClassification(**vars(cl_args))

    hi.predict_all(save_path='testpreds/')

    return 1

if __name__ == "__main__":
    main(sys.argv)
