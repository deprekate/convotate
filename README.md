## Introduction
CONVOTATE is a tool to classify prokaryotic genes using neural network models. It reads in (translated) fasta files and returns predictions for function based on the Subsystems ontology. In our benchmarks, high confidence (>0.99) predictions returned AUC scores >0.999.

Advantages of CONVOTATE:
1. Fast: 20,000-70,000 annotations per minute with GPU, 600 per minute on a 13-inch macbook pro laptop.
2. Low resource: <110MB disk space and scalable in RAM (can run with <1GB RAM if necessary).
3. As it is a predictions tool, it is able to return a "best guess" for any input sequence irrespective of whether there are homology matches in other databases. 


### Installation and GPU configuration
CONVOTATE needs 107MB disk space.

We recommend running CONVOTATE from a conda environment, with python3.6. If you have a GPU, the simplest install process uses keras-gpu, which is distributed under conda, not PyPI. If you are using a CPU, the conda tensorflow will likely be faster than the pip tensorflow.  If you are using an old version of OSX, please see Troubleshooting.

To install CONVOTATE:
```sh
 git clone git@github.com:deprekate/convotate.git
 cd convotate
```

There are two options for required packages, depending on whether you will be using CPU or GPU for computation. If you have a GPU, use it: CONVOTATE will produce 20-70k annotations per minute (depending on hardware). 
1. CPU: ```conda install keras pandas numpy```
2. GPU: ```conda install keras-gpu pandas numpy```

TensorFlow is also a requirement but will automatically install when you install Keras. If you have a GPU and the software isn't making use of it, it's likely the system is using tensorflow-cpu (see Troubleshooting).

## Usage
### Example
CONVOTATE reads in sequences from a standard faa (single or multifasta) file e.g.:
```
>fig|100053.8.peg.7
MLDLRYITENTEDLKKVLELRGFKEVGIIDELKSIIQRKRELQREADVFREERNKVSKEVGRIKQSGGDITEISASVKLVGEKIKEIETKLEQEENALININLGLPNILDPKVPNGKSEHDNIVQYEVGKIPSFSFLPKPHFEIGEALNWINFEK
>fig|100053.8.peg.58
MQICRGKETCKAGRVFKMEKQNRTEQISKLQKESLDILVIGGGSTGTGAAFDAAKRGYKTALIEKKDFASGTSSRSTKLIHGGVRYLAQFHFKLIHEALTERQRLLENAPHLVKPLKFLLPAYRFYERPYYGIGLTLYDI
```
Commands to run CONVOTATE on included sample data (or replace tests/small.fasta with /path/to/your/file.fasta):
```sh
python3 convotate.py tests/small.fasta 
```

CONVOTATE takes the input sequences, and runs predictions at the lowest hierarchical level (Subsystems). If a result is below a confidence threshhold, it will send that sequence up the hierarchy to be classified at the next level as so:

<img src="fig1.png" alt="convotate" width="300"/>


The main output (filename_LCA.txt) contains per-read predicted functions in lowest common ancestor form, and should look like:
```
Sequence ID	Superclass	Class	Subclass	Subsystem	Confidence
fig|100053.8.peg.7	Protein Processing	Protein Synthesis	Aminoacyl-tRNA-synthetases		1
fig|100053.8.peg.58	Metabolism, Energy	Fatty Acids, Lipids, and Isoprenoids	Phospholipids		1
fig|100053.8.peg.59	Protein Processing	Protein Synthesis	nan	Universal GTPases	0.9976
fig|100053.8.peg.84	Metabolism, Energy	Metabolite damage and its repair or mitigation	nan	Nudix proteins (nucleoside triphosphate hydrolases)	1
...
```
There will also be a "discarded" file for sequences which did not meet the confidence threshold (filename_discarded.txt). Finally, there is a complete output file (filename_complete.txt). The complete output contains annotations/confidence scores for every model output of every input sequence:

```
Sequence ID	Full Annotation
>fig|100053.8.peg.7	[('Subsystem', 'Protein Processing>Protein Synthesis>Aminoacyl-tRNA-synthetases>tRNA aminoacylation, Ser', 1.0)]
>fig|100053.8.peg.58	[('Subsystem', 'Metabolism, Energy>Fatty Acids, Lipids, and Isoprenoids>Phospholipids>Glycerolipid and Glycerophospholipid Metabolism in Bacteria', 0.9953047)]
>fig|100053.8.peg.101	[('Subsystem', 'Metabolism, Energy>Amino Acids and Derivatives>Arginine; urea cycle, creatine, polyamines>Arginine biosynthesis via N-acetyl-L-citrulline!Arginine biosynthesis', 0.5832784), ('Subclass', 'Metabolism, Energy>Amino Acids and Derivatives>Proline and 4-hydroxyproline', 0.7927507), ('Class', 'Metabolism, Energy>Amino Acids and Derivatives', 0.92275023), ('Superclass', 'Metabolism, Energy', 0.97914463)]
...
```

### Options
There are some configurable options: 
```
-o OUTFILE, --outfile OUTFILE
                        where to write the output [stdout]
-b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of sequences to run at a time [1000]
-m MAX_LENGTH, --max_length MAX_LENGTH
                        maximum sequence length - sequences will be truncated beyond this point. [1950]
-c CONFIDENCE_THRESHOLD, --confidence CONFIDENCE_THRESHOLD
                        confidence threshold cutoff, between 0 and 1. [0.99]
``` 
Notes:
1. Batch size can help with RAM management. A batch size of 10000 (default) will use ~2.7 GB of RAM; a batch size of 100k takes ~17GB. Increase or decrease as desired. 
1. There is a maximum sequence length option, which truncates sequences after a certain point (default 1950). We have observed classifier performance doesn't decrease at this stage and it reduces computation time. 
1. We would generally recommend keeping the confidence threshold at 0.99 as it keeps the AUC score >0.999. If you want a "best guess" for a mystery sequence you can reduce this to 0, or simply look at the complete output file.

We do not have a minimum length as if sequences are too short/ambiguous they will return a low confidence result regardless.

## Troubleshooting

### OSX compatibility
Older versions of OSX have difficulty running TensorFlow.  It is recommended to use Python 3.6 and TensorFlow 1.9 for compatibility.

First you can create a python3.6 environment and then install the correct package versions.
```sh
conda create -n envconvotate python=3.6 anaconda
conda activate envconvotate
conda install tensorflow=1.9.0 keras pandas numpy
```

And then in your environment, you can run `convotate` using python3.6
```sh
python convotate.py tests/small_testset.faa
```

### Linking tensorflow to the GPU
TensorFlow needs a GPU with CUDA compute capability >=3.0 (most relatively modern GPUs should have this). To force GPU usage, _uninstall all versions of Keras and TensorFlow_, or create a new clean conda environment. Then use `conda install keras-gpu` for your TensorFlow/Keras installation. To check if it has worked, in your active environment, use:
```sh
python3
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```
which will output devices used by Tensorflow. If you can see a GPU there, Tensorflow should default to the GPU. If the GPU does not have the required compute capability it will print a warning. 

#### HPCs

GPU usage on HPC hardware may be less straightforward but should still be achievable. We have successfully tested CONVOTATE on two HPCs and found that they required specific versions of python and TensorFlow to suit the CUDA version on the cluster. Our recommendation is to get the GPU/CUDA information, and find the compatible versions of tensorflow and python. E.g. one of the clusters required the combination of python 3.5.1 with tensorflow <=1.10. 