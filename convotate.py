import sys
import pickle as pk

import modules.helper_functions as helpers
import modules.protein_function_hierarchy as ph

def main(argv):

    cl_args = helpers.get_args()

    input_sequences = helpers.read_fasta(cl_args.infile);

    hi = ph.HierarchicalProteinClassification(**vars(cl_args))

    #hi.predict_all(save_path='test_preds/')
if __name__ == "__main__":
    main(sys.argv)
