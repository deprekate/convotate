Introduction
------------

MODEL is a tool to classify genes.

To install `MODEL`,
```sh
 git clone git@github.com:deprekate/model_package.git
 cd model_package; make
```

MODEL Example
--------------

Run on included sample data:
```sh
./model.py tests/small.fasta 
```
Output is the predicted functions, and should look like
```sh
Sequence ID	Superclass	Class	Subclass	Subsystem	Confidence
fig|100053.8.peg.7	Protein Processing	Protein Synthesis	Aminoacyl-tRNA-synthetases		1
fig|100053.8.peg.58	Metabolism, Energy	Fatty Acids, Lipids, and Isoprenoids	Phospholipids		1
fig|100053.8.peg.59	Protein Processing	Protein Synthesis	nan	Universal GTPases	0.9976
fig|100053.8.peg.84	Metabolism, Energy	Metabolite damage and its repair or mitigation	nan	Nudix proteins (nucleoside triphosphate hydrolases)	1
...
```

Trouble Shooting
--------------
Older versions of OSX have difficulty running TensorFlow.  It is recommended to use Python 3.6 and TensorFlow 1.9 for compatibility.
