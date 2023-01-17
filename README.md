# CNN-LSTM-CTC Amharic OCR
## End-to-End learning
- This method uses CNN-LSTM-CTC networks.
- To train and test this model, please use the ADOCR dataset from <s> http://www.dfki.uni-kl.de/~belay/</s> . https://bdu-birhanu.github.io/amharic.html (new link) However, to play with the code you may also use the sample database given, together with the source code, in this directory.
- Both the text-line images and corresponding ground truth are given in numpy format.


### To run the code with Terminal use the following info:
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
