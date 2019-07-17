import pickle as pk

import modules.helper_functions as helpers
import modules.protein_function_hierarchy as ph


args = helpers.get_args()

exit()

hi = ph.Hierarchical_Protein_Classification(
                 sequence_file = 'small_testset.txt',
                 model_label_refs ='model_label_referencefile_190620.pkl',
                 ontology_file = 'Ontology_20190609_Final.txt', 
                 subsystem_merge_file = 'Subsystem_Name_toMerged_mapping.txt',
                 subsys_sets_idx_file = 'setlabel_subsystemlabel_xref_190620.pkl',
                 set_files_pattern = 'finalmodels/set*_subsystems_model_ker50.h5',
                 base_models_pattern = 'finalmodels/*bigdata.h5',             
                 chunksize = 1000, max_seq_len = 1950, confidence_threshold = 0.9,
)

#hi.predict_all(save_path='test_preds/')
