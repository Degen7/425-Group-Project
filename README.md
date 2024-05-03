### 1. BassetModel.py:
This Python script defines a deep learning model for classifying DNA sequences. The model, `DNASeqClassifier`, uses several convolutional, batch normalization, pooling, and dropout layers followed by fully connected layers. It also includes functions `train_model` for training the model and `validate_model` for evaluating its performance. Additionally, it defines a `DNADataset` class to handle the dataset used for training and validation, which is compatible with PyTorch's `Dataset` class.

### 2. DataPreparation.ipynb:
This Jupyter Notebook script is used for preparing DNA sequence data. It processes genomic data files to extract sequences, manipulate them, and prepare them into a structured format suitable for machine learning applications. The script includes functions for reading chromosome sequences, processing data frames to adjust sequence positions, and generating datasets with labeled positive and negative examples. It also includes utility functions for one-hot encoding of sequences and saving the prepared data in a pickle file for future use.

### 3. RunBasset.ipynb:
This Jupyter Notebook script is designed to execute the model training and validation processes using the deep learning model defined in `BassetModel.py`. It includes steps to load prepared data, set up data loaders for training and testing datasets, and configure the training environment. The script handles the training process using custom functions, performs validation to assess the model's performance, and saves the trained model to disk. It also includes a function to execute other notebooks, which is used to ensure that data preparation is completed before model training begins.
