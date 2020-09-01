from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import editdistance

from data_loader import x_testp, y_testp, x_testpg, y_testpg, x_testvg, y_testvg 
'''
Here, x_testp,x_testpg, x_testvg denote printed text-image with powergeez font,
synthetic image with powergeez font and visual geez font respectivily

The following program returns the CER of printed text-line image onl,y and then
 you can follow the same steps for the synthetic images
'''
model = load_model('model_test.hdf5')
y_pred=model.predict(x_testp)

#the CTC decoer
y_decode = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])

for i in range(10):
    # print the first 10 predictions
    print("Prediction :", [j for j in y_decode[i] if j != -1], " -- Label : ", y_testp[i])

#=========== compute editdistance and returne CER ====================================

true=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_testp)):
    x=[j for j in y_testp[i] if j!=0]
    true.append(x)
pred=[]# to stor the pdicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
for i in range(len(y_decode)):
    x=[j for j in y_decode[i] if j not in(0,-1)]
    pred.append(x)

cer=0
for(i,j) in zip(true,pred):
    x=editdistance.eval(i,j)
    cer=cer+x
err=cer
x=0
for i in range(len(true)):
    x=x+len(true[i])
totalchar=x
cerp=(float(err)/totalchar)*100
print(" The charcter error rate (CER) for printed testset is "+ str(cerp))

#to visualiza sample text-line image( the image at index 1
x_testp_orig=x_testp.reshape(x_testp.shape[0],x_testp.shape[1],x_testp.shape[2])
fig, ax = plt.subplots()
i=ax.imshow(x_testp_orig[1],cmap='Greys_r')# todisply the backgrouund to be white
plt.show()
