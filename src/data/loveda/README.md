Download the LoveDa dataset from ```https://zenodo.org/records/5706578``` and unzip the dataset in this folder.

You can do so, for example, by running:

```wget https://zenodo.org/records/5706578```
```unzip files-archive```

(after this, ```unzip Train.zip Val.zip Test.zip```).

The extracted images in the folders `Train` and `Val` can be used to evaluate geographic transfer.

To be able to evaluate random split mode, redistribute the data using the ```train_val_split.py```.