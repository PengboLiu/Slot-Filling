# Slot-Filling
**Slot Filling Using RNN with ATIS data.**
## Requirement
Python3.6  
PyTorch  
progressbar
## Dataset
Airline Travel Information System(ATIS) dataset.  

Here is an example sentence and its labels from the dataset:

  Show | flights | from |   Boston | to |  New | York|today
  ---   | --- | --- |   --- | --- |  --- | ---|    ---
 O | O | O |B-dept | O|B-arr|I-arr|B-date
## Result
**Bi-GRU**

      | Precision | Recall | F1 |
---   | -- | --- |-- |
train-set|99.77|99.83|99.8|
test-set|94.78|94.75|94.76|
