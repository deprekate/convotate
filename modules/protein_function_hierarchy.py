import os
import pickle as pk
from glob import glob
import string
import h5py 

import pandas as pd
import numpy as np

from modules.helper_functions import natural_sorted
from modules.helper_functions import FastaFile

from keras.optimizers import Adam, RMSprop

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape, Permute
from keras.layers import Conv1D,MaxPool1D #Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.layers import LeakyReLU, Dropout, PReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.backend import set_session

from keras import backend as K
from keras.models import load_model 
from keras.engine.topology import Layer
import keras.backend as K

# from tensorflow.examples.tutorials.mnist import input_data

# from spp.SpatialPyramidPooling import SpatialPyramidPooling

import tensorflow as tf
set_session(tf.InteractiveSession())


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class SpatialPyramidPooling1D(Layer):
    """Spatial pyramid pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        3D tensor with shape:
        `(samples, rows, channels)` if dim_ordering='tf'.
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i * i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i for i in pool_list])
        super(SpatialPyramidPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        input_shape = K.shape(x)
        if self.dim_ordering == 'th':
            num_rows = input_shape[2]
        elif self.dim_ordering == 'tf':
            num_rows = input_shape[1]
        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        outputs = []
        if self.dim_ordering == 'tf':
            for pool_num, num_pool_regions in enumerate(self.pool_list):
                for jy in range(num_pool_regions):
                        y1 = jy * row_length[pool_num]
                        y2 = jy * row_length[pool_num] + row_length[pool_num]
                        y1 = K.cast(K.round(y1), 'int32')
                        y2 = K.cast(K.round(y2), 'int32')
                        new_shape = [input_shape[0], y2 - y1,
                                     input_shape[-1]]
                        x_crop = x[:, y1:y2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1,))
                        outputs.append(pooled_val)
        else:
            raise TypeError()
        if self.dim_ordering == 'th':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'tf':
            outputs = K.concatenate(outputs)
        return outputs
    
amino_acids = {a:i for i,a in enumerate(string.ascii_uppercase+'_')} # '_':unknown stuff

def barcode1(seq, length=1500, amino_acids=amino_acids):
    protein_barcodes = np.zeros((length, len(amino_acids)), dtype = np.uint8)
    for i2,j in enumerate(seq[:length]):
        protein_barcodes[i2,amino_acids.get(j,26)] = 1 # if character not found, map to '_'
    return protein_barcodes



def make_set_model(model_weight_file_path,max_len = 1500, pool_list = [1,4,16,32] ):
    weight_file = h5py.File(model_weight_file_path, 'r')
    # The shape of the model should be saved separately in a json or pickle file
    # get the shapes
    for k in weight_file['model_weights']:
        # find how the conv and dense layer are named
        if 'conv1d' in k:
            conv_name = k
        elif 'dense' in k:
            dense_name = k
    weights_conv = weight_file['model_weights'][conv_name][conv_name]['kernel:0']
    kernel_size, num_amino_acids, num_conv_filters = weights_conv.shape
    weights_dense = weight_file['model_weights'][dense_name][dense_name]['kernel:0']
    _in_dim, out_dim = weights_dense.shape
    model_base = Sequential()
    model_base.add(Conv1D(num_conv_filters, kernel_size = kernel_size, activation='relu', input_shape = (max_len, num_amino_acids)))
    model_base.add(SpatialPyramidPooling1D(pool_list)) 
    model_base.add(Dropout(0.2))
    model_base.add(Dense(out_dim, activation='softmax'))
    ### Finally, load trained weights
    model_base.load_weights(model_weight_file_path)
    return model_base

class HierarchicalProteinClassification():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        #self._sequence_file = pd.read_csv(self.infile ,sep ='\t', iterator=True, chunksize=self.batch_size )
        self._sequence_file = FastaFile(self.infile, self.batch_size) 
        # read in all data files
        self._ontology_file = pd.read_csv(self.ontology_file, sep='\t')
        self._subsystem_map = {k:v for k,v in pd.read_csv(self.merged_file, sep= '\t').values}
        # model_label_refs contains map of classification indices to name of label classes for all levels of hierarchy and all sets 
        self.model_label_refs = pk.load(open(self.label_file,'rb'))
        self.ontology = self.get_ontology()
        self.get_subsystem_sets_map(self.indexes_file)
        self.load_trained_models(self.basemodel_files, self.pattern_files)
        self.classification_level = {i:k for i,k in enumerate(['Superclass', 'Class', 'Subclass', 'Subsystem']) }
        
    def get_ontology(self ):
        ont = self._ontology_file
        ont_tree = {ont['Subsystem Merged'][i]: {k: ont[k][i] for k in ['Superclass', 'Class', 'Subclass']} for i in range(len(ont))}
        # ont_tree is missing 'Subsystem Merged' mapping, so:
        ontology = {k: {'Subsystem Merged':v} for k,v in self._subsystem_map.items()}
        for k in ontology:
            v = ontology[k]
            v.update(ont_tree[v['Subsystem Merged']])
        return ontology
    
    def get_subsystem_sets_map(self,subsys_file):
        self.set_class_list = pk.load(open(subsys_file, 'rb'))
        self.subsys_to_sets_map = {} 
        for set_num, subsys_list in self.set_class_list.items():
            for s in subsys_list:
                self.subsys_to_sets_map[s] = self.subsys_to_sets_map.get(s,[]) + [set_num]

    def load_trained_models(self, base_models_pattern,set_files_pattern ):
        self.load_base_models(base_models_pattern)
        self.load_model_sets(set_files_pattern)
        
    def load_base_models(self, base_models_pattern):
        print('Loading base models...')
        self.base_models = {}
        for fpath in glob(os.path.join(base_models_pattern, '*.h5')):
            path, fname = os.path.split(fpath)
            k = fname.split('_model')[0].capitalize() # caps for consistency with label refs
            print('Loading %s from: %s' %(k,fpath))
            self.base_models[k] = make_set_model(fpath, max_len = self.max_length)
    
    def load_model_sets(self,set_files_pattern):
        print('Loading ambiguous subsystem sets models...')
        sets_info = {}
        for fpath in natural_sorted(glob(os.path.join(set_files_pattern, '*.h5'))):
            path, fname = os.path.split(fpath)
            k = int(fname.split('set')[1].split('_subsystems')[0])
            sets_info[k] = {'file': fpath, 'classes': self.set_class_list[k]}
        self.model_sets = {}
        for k in sets_info:
            print('Loading %s from: %s' %(k,sets_info[k]['file']))
            self.model_sets[k] = make_set_model(sets_info[k]['file'], max_len= self.max_length) 
        
    def save_output_files(self, chunk_idx, save_path='.', delimiter = '\t'):
        c_level = ['Superclass', 'Class', 'Subclass', 'Subsystem']  
        map_level = {v: i+1 for i,v in enumerate(c_level)}
        annotation_LCA_str = delimiter.join(['Sequence ID']+c_level+['Confidence'])
        full_annotation_str = delimiter.join(['Sequence ID','Full Annotation'])
        for k,v in self.protein_chunk_prediction_summary.items():
            #seq_id = self.sequences[k]
            # LCA
            empty_line = ['']*(len(c_level)+2)
            empty_line[0] = k
            level, label, confidence = v[-1]
            for col, lab in zip(c_level, label.split('>')):
                empty_line[map_level[col]] = lab
            # confidence 
            empty_line[ -1 ] = '%.4g' %v[-1][-1]
            self.outfile.write(delimiter.join(empty_line))
            self.outfile.write("\n")
            #self.outfull.write(delimiter.join( [k, str(v)]))
            #self.outfull.write("\n")

    def save_output_summary(self,chunk_idx, save_path='.', delimiter = '\t'):
        c_level = ['Superclass', 'Class', 'Subclass', 'Subsystem']  
        map_level = {v: i for i,v in enumerate(c_level)}
        header = delimiter.join(c_level + ['Count'])
        annotation_summary = header
        for level in c_level[::-1]:
            for label_idx in self.labels_dict.get(level,[]) :
                label = self.model_label_refs[level][label_idx]
                count = len(self.labels_dict[level][label_idx]['input_index'])
                empty_line = ['']*(len(c_level)+1)
                for col, lab in zip(c_level, label.split('>')):
                    empty_line[map_level[col]] = lab
                empty_line[-1] = str(count)
            self.outsumm.write(delimiter.join(empty_line))
            self.outsumm.write("\n")

    def save_discarded(self,chunk_idx, save_path='.'):
        # indices left unannotated at Superclass level
        unannotated_idx = self.send_up.get('Superclass',set())
        out_str = 'Sequence ID'
        for i in unannotated_idx:
            self.outdrop.write(self.sequences['feature.patric_id'].loc[i])
            self.outdrop.write("\n")

    def predict_all(self, save_path = '.', delimiter = '\t'):
        os.makedirs(save_path, exist_ok=True)
        chunk_count = 0
        while True:
            print(chunk_count, end = ',')
            self.sequences = self._sequence_file.get_chunk()
            if not self.sequences:
                break
            self.predict_chunk()
            self.save_output_files(chunk_count, save_path = save_path, delimiter = delimiter)
    #        self.save_output_summary(chunk_count, save_path = save_path, delimiter = delimiter)
    #        self.save_discarded(chunk_count, save_path = save_path)
    #       self.output_DataFrame = self.make_output_DataFrame()
    #       start = chunk_count*self._sequence_file.batch_size
    #       self.output_DataFrame.to_csv(os.path.join(save_path, 'seq_predictions_%d-%d.csv' %(start, start+ len(self.sequences) ) ))
            chunk_count += 1
    
    def predict_all_old(self, save_path = '.'):
        chunk_count = 0
        while True:
            print(chunk_count, end = ',')
            self.sequences = self._sequence_file.get_chunk()
            if len(self.sequences) == 0:
                break
            self.predict_chunk()
            self.output_DataFrame = self.make_output_DataFrame()
            start = chunk_count*self._sequence_file.batch_size
            self.output_DataFrame.to_csv(os.path.join(save_path, 'seq_predictions_%d-%d.csv' %(start, start+ len(self.sequences) ) ))
            chunk_count += 1
        
    def predict_chunk(self):
        # self.sequences = self._sequence_file.get_chunk()
        # 3. build hashtable of classification 
        #     1. keys are subsystem labels
        #     2. store both confidence level and index of the input
        #     3. for non-confused classes, have a low confidence list, storing indices of results below the confidence threshold 
        self.labels_dict, self.send_up = {}, {}
        self.labels_dict['Subsystem'], self.send_up['Subsystem'] = self.predict_one_model(
                                                                       self.sequences, 
                                                                       self.base_models['Subsystem'], check_subsys_sets=True
                                                                   )
        # 4. pass all inputs classified into the confused subsystems to the sets containing those subsystems
        # 5. update classification hashtable, correcting the results for the confused subsystems after using the sets
        # 6. add the low confidence results of confused classes to lox-confidence list 
#         labels_dict, send_up = 
        self.disambiguiate_subsystems(self.labels_dict['Subsystem'], self.send_up['Subsystem'])
        self.classify_upper_hierarchies()
        self.compile_prediction_summary()
        
    def classify_upper_hierarchies(self, ):
        c_level = ['Superclass', 'Class', 'Subclass', 'Subsystem'][::-1]
        if len(self.send_up['Subsystem']) == 0: return 
        for i in range(1,len(c_level)):
            self.labels_dict[c_level[i]], self.send_up[c_level[i]] = self.predict_one_model(
                                                                       #self.sequences.loc[self.send_up[c_level[i-1]]],
                                                                       dict([(key,self.sequences[key]) for key in self.send_up[c_level[i-1]]]),
                                                                       self.base_models[c_level[i]], check_subsys_sets=False
                                                                     )
            if len(self.send_up[c_level[i]]) == 0 :
                # nothing needs to be passed to higher level of hierarchy
                break
        
    def compile_prediction_summary(self):
        # compile summary of predictions
        # the low confidence ones should have their lower level predictions included
        c_level = ['Superclass', 'Class', 'Subclass', 'Subsystem'][::-1]
        self.protein_chunk_prediction_summary = {i:[] for i in self.sequences}
        for c in c_level:
            for label_index in self.labels_dict.get(c,[]):
                pc = self.labels_dict[c][label_index]
                for i, conf in zip(pc['input_index'], pc['confidence']):
                    label_name = self.model_label_refs[c][label_index] 
                    self.protein_chunk_prediction_summary[i] += [(c,label_name,conf)]

    def disambiguiate_subsystems(self, preds_dict, send_up):
        """Use model_sets to reclassify seqs classified into the ambiguous subsystems."""
        for sets_index, ss_list in self.set_class_list.items():
            reclassify_list = []
            for subsys in ss_list:
                if subsys in preds_dict:
                    reclassify_list += preds_dict[subsys]['input_index'] 
            if len(reclassify_list) > 0:
                preds_dict_set, send_up_set = self.predict_one_model(
                                                dict([(key,self.sequences[key]) for key in reclassify_list]),
                                                self.model_sets[sets_index],
                                                check_subsys_sets=False
                                              ) # no ambiguity
                #send_up += send_up_set
                #send_up = send_up.union(send_up_set)
                for s in send_up_set:
                    send_up.add(s)
                # update the ambiguous subsystems
                for i in preds_dict_set:
                    subsys = self.set_class_list[sets_index][i]
                    preds_dict[subsys] = preds_dict_set[i]
#         return preds_dict, send_up

    def predict_one_model(self, prots, model, check_subsys_sets = False, top = 1, ):
        predictions_dict = {} #{i: []  for i in range(model.output_shape[-1])}
        labels_dict = {}
        send_up_hierarchy_idx = [] # classify these with higher levels of hierarchy
        # 1. convert sequence to barcode
        test_data = np.array([barcode1(sequence, length=self.max_length) for header,sequence in prots.items()])
        # 2. classify with 'Susbsystem' model
        # subsys_model = base_models['Subsystem'] # !!! resolve naming discrepency: base model file names 'subsystem', not 'Subsystem Merged'
        predictions_initial = model.predict(test_data)
        # for input_index, prediction in enumerate(subsystem_predictions_initial):
        # for input_index, prediction in zip(prots.index, predictions_initial):
        # for input_index, prediction in enumerate(predictions_initial):
        for input_index, prediction in zip(prots, predictions_initial):
            # find the top predictions and report them with their confidence
            pred_top_idx = np.argsort(prediction)[::-1][:top]
            for label_idx in pred_top_idx:
                labels_dict.setdefault(label_idx, {'input_index':[], 'confidence': [] })
                labels_dict[label_idx]['input_index'] += [input_index] 
                labels_dict[label_idx]['confidence'] += [prediction[label_idx]] 
#               if check_subsys_sets: continue # we will disambiguate, so don't count low conf unless not in subsys_to_set_map
                if prediction[label_idx] < self.confidence_threshold:
                    if check_subsys_sets:
                        # only send up if not in sets
                        if label_idx not in self.subsys_to_sets_map:
                            #print(label_idx, input_index, prediction[label_idx])
                            send_up_hierarchy_idx += [input_index]
                            # it's subsys level and label is in the ambiguous set
                            # don't send up hierarchy! instead, we'll pass to sets
                    else: 
                        #print(label_idx, input_index, prediction[label_idx])
                        send_up_hierarchy_idx += [input_index]
        return labels_dict, set(send_up_hierarchy_idx)
        
# 7. pass the low confidence list to higher levels of hierarchy
# 8. keep a second hashtable, with keys being the input indices, and values being tuples of classification class and confidence for all levels that were checked, 
    
    def make_output_DataFrame(self):
        # to make sure no indices are skipped, we make all columns the size of the full input 
        output_dict = {k:[None]*len(self.sequences.index) for k in [
                'Superclass', 'Class', 'Subclass', 'Subsystem','Confidence', 
                'Full_classification']}
        c_level = ['Superclass', 'Class', 'Subclass', 'Subsystem']  
        for i, idx in enumerate(self.sequences):
            output_dict['Full_classification'][i] = self.protein_chunk_prediction_summary[idx]
            level, label, confidence = self.protein_chunk_prediction_summary[idx][-1]
            # break down the last classification and put it under the right columns
            output_dict['Confidence'][i] = confidence 
            for col, lab in zip(c_level, label.split('>')):
                output_dict[col][i] = lab

        return self.sequences.join(pd.DataFrame(output_dict, index = self.sequences.index)) 
        
        
        
