# HACK

# Environments
> conda env create -f hack.yaml

## Data Preparation
### Step 1: raw data preprocessing
Require access to the two datasets and download the data:
* [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
* [DISFA](http://www.engr.du.edu/mmahoor/DISFA.htm)

Preprocess datasets using Dlib:
* Detect face and facial landmark
* Align the cropped faces according to the computed coordinates of eye centers
* Resize faces to (256, 256)

### Step2: random a 3-fold subject-exclusive split
E.g., for BP4D
* `Fold-0`: F004 F019 F005 M012 M006 F011 F022 M018 F003 F014 M002 M014 F017 M010 
* `Fold-1`: M008 F007 F021 F010 F020 F016 M001 F023 M007 M013 M009 M017 F006 F013 
* `Fold-2`: M011 M003 F008 F001 M004 F012 F015 M005 F018 F002 M016 M015 F009 

### Step 3: feeder input generation
For dataloader, ./feeder/feeder_HACK.py requires two data files
Generate these files for each fold

* `label_path`: the path to file which contains labels ('.pkl' data), [N, num_class]
* `image_path`: the path to file which contains image paths ('.pkl' data), [N, 1]

## Compute meta
We provide an example in ./misc

## Run 
```bash
python run.py
```
