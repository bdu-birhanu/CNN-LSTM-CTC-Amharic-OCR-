# CNN-LSTM-CTC Amharic OCR
## End-to-End learning
- This method uses CNN-LSTM-CTC networks.
- To train and test this model, please use the ADOCR dataset from http://www.dfki.uni-kl.de/~belay/. However, to play with the code you may also use the sample database given, together with the source code, in this directory.
- Both the text-line images and corresponding ground truth are given in numpy format.


### To run the code with Terminal use the following info (Note: if you want to train using the whole data, make sure that your pc has enough memory >=32 GB RAM, otherwise you need to write your own code for datagenerator).
```
# Load and Pre-process data
python data_loader.py

# Train
python train_model.py

# Test and results
python test_model.py
```
## Some issues to know
1. The test environment is
    - Python 2.7
    - Keras 2.2.4
    - tensorflow 1.14
