from PIL import Image
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
import  glob,cv2
import numpy as np
from sklearn.cross_validation import train_test_split

num_class=280# number of unique charcter 
dim=(128,32)# image dimension
maxlen=32 #the max strign length og GT 
def load_dataset():
    '''
    This function loads the training and test datset ( we have three differnt test sets)
    and returns the following arguments. you may download the image from 
    http://www.dfki.uni-kl.de/~belay/ and store in the same directory
    train_imagei --> training text-line images
    train_texi -->  Ground truth of training data
    test_imagep----> test set of printed text-line image with a power-geez font 
    test_imagepg----> test set of synthetic text-line image with power-geez font
    test_imagevg----> test set of  synthetic text-line with visual geez font
    test_textp----> Ground truth for printed text-line image with a power-geez font 
    test_textpg----> Ground truth for synthetic text-line image with power-geez font
    test_textvg----> Ground truth for synthetic text-line with Visualgeez font

we recommend you to run this code with full dataset directly if you computer have >=32 GB RAM
Otherwise, you need to write  your own Data-generator code ( will do it soon).
to check how it works you could use the give sample text line image.

   '''   
    train_imagei=np.load('./sample_dataset/X_trainp_pg_vg.npy')
    train_texi=np.load('./sample_dataset/y_trainp_pg_vg.npy')
    test_imagep=np.load('./sample_dataset/x_testp.npy')
    test_textp=np.load('./sample_dataset/y_testp.npy')
    test_imagepg=np.load('./sample_dataset/X_test_pg.npy')
    test_textpg=np.load('./sample_dataset/y_test_pg.npy')
    test_imagevg=np.load('./sample_dataset/X_test_vg.npy')
    test_textvg=np.load('./sample_dataset/y_test_vg.npy')
    return train_imagei, train_texi, test_imagep, test_textp, test_imagepg, test_textpg, test_imagevg, test_textvg

#the following two functions are employed for test sets and trainsets separetly just for simplcity
def preprocess_traindata():
    ''' 
    input: a 2D shape text-line image (h,w)
    output:  returns 3D shape image format (h,w,1)

    Plus this function randomly splits the training and validation set
    This function also computes list of length for both training and validation images and GT
      '''
    im_train=load_dataset()
    train_imagei = im_train[0]
    train_texi = im_train[1]
        
    im_train=[]
    for i in train_imagei:
        im_train.append(im_resize(i))
    
    im_train=np.array(im_train)

    train_image, val_image, train_tex, val_tex = train_test_split(im_train, train_texi, test_size=0.07)

     
    X_train=train_image.reshape(train_image.shape[0], train_image.shape[1], train_image.shape[2],1) #[samplesize,32,128,1]
    X_val=val_image.reshape(val_image.shape[0], val_image.shape[1], val_image.shape[2],1)
    y_train=train_tex
    y_val=val_tex

    nb_train = len(X_train)
    nb_val = len(X_val)
    #create list of input lengths
    #the +31 here is just a kind pad to make the size of the image equal to the out put of your LSTMs
    x_train_len = np.asarray([len(X_train[i])+31 for i in range(nb_train)])
    x_val_len = np.asarray([len(X_val[i])+31 for i in range(nb_val)])
 
    y_train_len = np.asarray([len(y_train[i]) for i in range(nb_train)])
    y_val_len = np.asarray([len(y_val[i]) for i in range(nb_val)])

    training_img = np.array(X_train)
    train_input_length = np.array(x_train_len)
    train_label_length = np.array(y_train_len)

    valid_img = np.array(X_val)
    valid_input_length = np.array(x_val_len)
    valid_label_length = np.array(y_val_len)

    return training_img, y_train, train_input_length, train_label_length, valid_img, y_val, valid_input_length, valid_label_length

def preprocess_testdata():
    '''
    this function helps to manipulate the test samples
    input: 2D test image
    output: 3D image formats
    '''
    im_test=load_dataset()
    test_imagep = im_test[2]
    test_imagepg = im_test[4]
    test_imagevg = im_test[6]
    y_testp = im_test[3]
    y_testpg = im_test[5]
    y_testvg = im_test[7]

    im_testp=[]
    for i in test_imagep:
        im_testp.append(im_resize(i))
    im_testpg=[]
    for i in test_imagepg:
        im_testpg.append(im_resize(i))
    im_testvg=[]
    for i in test_imagevg:
        im_testvg.append(im_resize(i))

    im_testp=np.array(im_testp)
    im_testpg=np.array(im_testpg)
    im_testvg=np.array(im_testvg)
  
    X_testp=im_testp.reshape(im_testp.shape[0],im_testp.shape[1],im_testp.shape[2],1)
    X_testpg=im_testpg.reshape(im_testpg.shape[0],im_testpg.shape[1],im_testpg.shape[2],1)
    X_testpvg=im_testvg.reshape(im_testvg.shape[0],im_testvg.shape[1],im_testvg.shape[2],1)

    return X_testp, X_testpg, X_testpvg, y_testp, y_testpg, y_testvg

  
def im_resize(input_image):
    '''
   resize the image if you want , otherwise you can use as it is.
    '''
    f=cv2.transpose(input_image)# since images in the original dataset are transposed
    im_resize=cv2.resize(f,dim)
    return im_resize

'''
all set of text images and GT
'''
train=preprocess_traindata()
x_train=train[0]
y_train=train[1]
x_train_length=train[2]
y_train_length=train[3]

x_val=train[4]
y_val=train[5]
x_val_length=train[6]
y_val_length=train[7]

test=preprocess_testdata()
x_testp= test[0]
y_testp= test[3]
x_testpg= test[2]
y_testpg= test[4]
x_testvg= test[2]
y_testvg= test[5]
print("data_loading is compeletd")
