Introduction
------------

CONVOTATE is a tool to classify genes.

To install `CONVOTATE`,
```sh
 git clone git@github.com:deprekate/convotate.git
 cd convotate; make
```

CONVOTATE Example
--------------

Run on included sample data:
```sh
python3 convotate.py tests/small.fasta 
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

First you can install the older version of TensorFlow from the direct url using pip; then fetch the other requirements using pip.
```
python3.6 -m pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.9.0-py3-none-any.whl
python3.6 -m pip install -r requirements.txt
```
