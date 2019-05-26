# digit-recognizer
[Kaggle Competition](https://www.kaggle.com/c/digit-recognizer/overview)

## Competition Description
MNIST ("Modified National Institute of Standards and Technology") is the  classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. In this competition, the goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. 

## Setup
```pip install -r requirements.txt```

## Run Script

```
cd scripts
python main.p
```

## Command Line Interface
**-d (--dirpath)**
> Dataset directory path, default to `datasets/`

**-n (--n_train)**
> Train dataset size, default to all file.

**-t (--n_test)**
> Test dataset size, default to all file.

**-e (--epochs)** 
> Number of epochs, default to 3
  
**-b (--batch_size)**
> Batch size, default to 86

## Sumbmissions
Create a submissions file in the `submissions/` directory.

## Sources
https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
