## Temporal Shift Module for Jester Gesture Recognition 

According to [mit official code](https://github.com/mit-han-lab/temporal-shift-module), 
we reduce and modify some codes for jester dataset.

### Prerequisites

* Python 3.6
* PyTorch 1.2
* Opencv 3.4
* Other packages can be found in ```requirements.txt```

### Data Preparation

Firstly, we need to download the [Jester](https://20bn.com/datasets/jester/v1) dataset. 
Then, we process the data and generate corresponding labels. 
Finally, we get category.txt, train_videofolder.txt, val_videofolder.txt and test_videofolder.txt documents.

`python3 datas/generate_label.py`

### Train and Validate

`bash train.sh`

After total training epochs, you can get result.csv, 
that is the test result document, including video number and corresponding label.

### Reference

[paper links](https://arxiv.org/abs/1811.08383)
