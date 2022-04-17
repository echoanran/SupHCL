# Pursuing Knowledge Consistency: Supervised Hierarchical ontrastive Learning for Facial Action Unit Recognition

<p align="center">
<img src="images/overview.png" width="88%" />
</p>

## Dependencies
* python >= 3.6
* torch >= 1.10.0
* requirements.txt
```bash
$ pip install -r requirements.txt
```
* torchlight
```bash
$ cd $INSTALL_DIR/torchlight
$ python setup.py install
```

## Data Preparation
### Step 1: Download datasets
First, request for the access of the two AU benchmark datasets: [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://mohammadmahoor.com/disfa/).

### Step 2: Preprocess raw data
Preprocess the downloaded datasets using [Dlib](http://dlib.net/) (related functions are provided in `$INSTALL_DIR/au_lib/face_ops.py`):
* Detect face and facial landmarks
* Align the cropped faces according to the computed coordinates of eye centers
* Resize faces to (256, 256)

### Step 3: Split dataset for subject-exclusive 3-fold cross-validation
Split the subject IDs into 3 folds randomly (an example is provided in `$INSTALL_DIR/data/splits_*.txt`)

### Step 4: Generate feeder input files
Our dataloader `$INSTALL_DIR/feeder/feeder_SupHCL.py` requires two data files (an example is given in `$INSTALL_DIR/data/bp4d_example`):
* `label_path`: the path to file which contains labels ('.pkl' data), [N, 1, num_class]
* `image_path`: the path to file which contains image paths ('.pkl' data), [N, 1]

## Compute PCC matrices & prototypes
We provide the computed files in ./misc

## Training 
```bash
$ cd $INSTALL_DIR
$ python run.py
```