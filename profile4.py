import pickle
import gspread
import skimage
import numpy as np
from glob import glob
from skimage import util
from skimage.io import imread
from google.colab import drive
from skimage.filters import gabor
from skimage.feature import local_binary_pattern
from oauth2client.service_account import ServiceAccountCredentials as account
label = list()#generating testing data by extracting features from all images
#https://github.com/iitkliv/dlvcnptel/blob/master/lecture3.ipynb
drive.mount('/content/drive', force_remount=True)
scope = ['https://www.googleapis.com/auth/drive']
creds=account.from_json_keyfile_name('client_secret0.json',scope)
client=gspread.authorize(creds)
sheet=client.open('legislators').sheet1
trainingdata=glob('/content/drive/MyDrive/trainingdataset/*')#total trainingdata
#generating training data by extracting features from all images
trainfeats=np.zeros((len(trainingdata),9))
for tr in range(len(trainingdata)):
    print(str(tr+1)+'/'+str(len(trainingdata)))
    img_arr=np.array(imread(trainingdata[tr],as_gray=True))#converting to grayscale and array
    feat_lbp=local_binary_pattern(img_arr,8,1,'uniform')#finding lbp
    lbp_hist,_=np.histogram(feat_lbp,8)#energy and Entropy of lbp feature
    lbp_hist=np.array(lbp_hist,dtype=float)
    lbp_prob=np.divide(lbp_hist,np.sum(lbp_hist))
    lbp_energy=np.nansum(lbp_prob**2)
    lbp_entropy=-np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))
    #finding glcm features from cooccurance matrix
    gcomat=skimage.feature.greycomatrix(util.img_as_ubyte(img_arr)//32,[2],[0])#cooccurance matrix
    contrast=skimage.feature.greycoprops(gcomat,prop='contrast')
    dissimilarity=skimage.feature.greycoprops(gcomat,prop='dissimilarity')
    homogeneity=skimage.feature.greycoprops(gcomat,prop='homogeneity')
    energy=skimage.feature.greycoprops(gcomat,prop='energy')
    correlation=skimage.feature.greycoprops(gcomat,prop='correlation')
    gaborfilt_real,gaborfilt_imag=gabor(img_arr,frequency=0.6)#gabor filter
    gabor_hist,_=np.histogram(gaborfilt_real,8)#energy and Entropy of Gabor filter response
    gabor_hist=np.array(gabor_hist,dtype=float)
    gabor_prob=np.divide(gabor_hist,np.sum(gabor_hist))
    gabor_energy=np.nansum(gabor_prob**2)
    gabor_entropy=-np.nansum(np.multiply(gabor_prob,np.log2(gabor_prob)))
    feat_glcm1=np.array([contrast[0][0],dissimilarity[0][0],homogeneity[0][0]])
    feat_glcm2=np.array([energy[0][0],correlation[0][0]])
    feat_glcm3=np.array([lbp_energy,lbp_entropy,gabor_energy,gabor_entropy])
    concat_feat=np.concatenate((feat_glcm1,feat_glcm2,feat_glcm3))#concatenating features(2+5+2)
    trainfeats[tr,:]=concat_feat#stacking features vectors for each image
    label.append(sheet.get_all_records()[tr]['statusno'])#class label
trainlabel = np.array(label)
#normalizing the train features to the range 01
trmaxs = np.amax(trainfeats,axis=0)#finding maximum along each column
trmins = np.amin(trainfeats,axis=0)#finding maximum along each column
trmaxs_rep = np.tile(trmaxs,(len(trainingdata),1))#repeating the maximum value along the rows
trmins_rep = np.tile(trmins,(len(trainingdata),1))#repeating the minimum value along the rows
trainfeatsnorm = np.divide(trainfeats-trmins_rep,trmaxs_rep)#element-wise division
#saving normalized training data and labels
#saving normalized testing data and labels
with open("trainfeats.pckl", "wb") as f: pickle.dump(trainfeatsnorm, f)
with open("trainlabel.pckl", "wb") as f: pickle.dump(trainlabel, f)
