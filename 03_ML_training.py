
##**************************************************************************************##
##                     Step1. Load Packages and Input Data                              ##          
##**************************************************************************************##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,metrics
from sklearn.svm import SVC,LinearSVC 
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import matthews_corrcoef,auc, roc_curve,plot_roc_curve, plot_precision_recall_curve,classification_report, confusion_matrix,average_precision_score, precision_recall_curve
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import imblearn
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

############################# Step2: input data processing #####################

## giessen data
gi_data = np.load("/gi_CIP_FCGR200/alt_cnn_input.npy") 
gi_pheno = pd.read_csv("CIP_gi_pheno.csv",index_col=0)   
gi_data.shape,gi_pheno.shape
gi_data2 = gi_data.reshape(900,40000)
gi_pheno2 = gi_pheno.values
gi_pheno3 = gi_pheno2.reshape(900,)
gi_data2.shape,gi_pheno3.shape
X = gi_data2
y = gi_pheno3
X.shape,y.shape

## pubdata
pub_data = np.load("/pub_CIP_FCGR200/alt_cnn_input.npy") 
pub_pheno = pd.read_csv("CIP_pub_pheno.csv",index_col=0)    
pub_data.shape
pub_data2 = pub_data.reshape(1496,40000) 
pub_pheno2 = pub_pheno.values
pub_pheno3 = pub_pheno2.reshape(1496,)
pub_data2.shape,pub_pheno3.shape
x_test = pub_data2
y_test = pub_pheno3

undersample = RandomUnderSampler(sampling_strategy='majority')
pub_x_under,pub_y_under=undersample.fit_resample(pub_data2,pub_pheno3)
print(Counter(pub_y_under))

##**************************************************************************************##
##                     Step2. Training and evaluation of RF,LR, SVM                     ##          
##**************************************************************************************##

## cross validation
cv = StratifiedKFold(n_splits=5)
rf = RandomForestClassifier(n_estimators=200, random_state=0)
lr = LogisticRegression(solver = 'lbfgs',max_iter=1000)
svm = SVC(kernel='linear', probability=True)

##*************** F1 + ROC curve
rf_tprs = []
rf_prs = []
rf_roc_aucs = []
rf_pr_aucs = []
rf_f1_matrix_out = []
rf_f1_report_out = []
rf_MCC_out = []
rf_pred_cls_out = []
rf_pred_prob_out = []
rf_y_test_out = []
rf_mean_fpr = np.linspace(0, 1, 100)
rf_mean_recall = np.linspace(0, 1, 100)
## LR
lr_tprs = []
lr_prs = []
lr_roc_aucs = []
lr_pr_aucs = []

lr_f1_matrix_out = []
lr_f1_report_out = []
lr_MCC_out = []
lr_pred_cls_out = []
lr_pred_prob_out = []
lr_y_test_out = []
lr_mean_fpr = np.linspace(0, 1, 100)
lr_mean_recall = np.linspace(0, 1, 100)


## SVM
svm_tprs = []
svm_prs = []
svm_roc_aucs = []
svm_pr_aucs = []

svm_f1_matrix_out = []
svm_f1_report_out = []
svm_MCC_out = []
svm_pred_cls_out = []
svm_pred_prob_out = []
svm_y_test_out = []
svm_mean_fpr = np.linspace(0, 1, 100)
svm_mean_recall = np.linspace(0, 1, 100)


fig,[ax1,ax2,ax3] = plt.subplots(nrows=1,ncols=3,figsize=(15, 4))
for i, (train, test) in enumerate(cv.split(X, y)):
    ## train the new model
    rf.fit(X[train], y[train])
    ## roc curve
    rf_viz = plot_roc_curve(rf, X[test], y[test],name='K fold {}'.format(i),alpha=0.3, lw=1,ax=ax1)
    rf_interp_tpr = np.interp(rf_mean_fpr, rf_viz.fpr, rf_viz.tpr)
    rf_interp_tpr[0] = 0.0
    rf_tprs.append(rf_interp_tpr)
    rf_roc_aucs.append(rf_viz.roc_auc) 
    ## evaluation metrics 	
    rf_pred_cls = rf.predict(X[test])
    rf_pred_prob = rf.predict_proba(X[test])[:,1]
    rf_f1_matrix = confusion_matrix(y[test],rf_pred_cls)
    rf_f1_report = classification_report(y[test],rf_pred_cls)
    rf_MCC = matthews_corrcoef(y[test],rf_pred_cls)
    ### save evalu_metrics out
    rf_pred_cls_out.append(rf_pred_cls)
    rf_pred_prob_out.append(rf_pred_prob)
    rf_f1_matrix_out.append(rf_f1_matrix)
    rf_f1_report_out.append(rf_f1_report)
    rf_MCC_out.append(rf_MCC)
    rf_y_test_out.append(y[test])
    
    ## LR
    lr.fit(X[train], y[train])
    ## roc curve
    lr_viz = plot_roc_curve(lr, X[test], y[test],name='K fold {}'.format(i),alpha=0.3, lw=1,ax=ax2)
    lr_interp_tpr = np.interp(lr_mean_fpr, lr_viz.fpr, lr_viz.tpr)
    lr_interp_tpr[0] = 0.0
    lr_tprs.append(lr_interp_tpr)
    lr_roc_aucs.append(lr_viz.roc_auc) 
    ## evaluation metrics 	
    lr_pred_cls = lr.predict(X[test])
    lr_pred_prob = lr.predict_proba(X[test])[:,1]
    lr_f1_matrix = confusion_matrix(y[test],lr_pred_cls)
    lr_f1_report = classification_report(y[test],lr_pred_cls)
    lr_MCC = matthews_corrcoef(y[test],lr_pred_cls)
    ### save evalu_metrics out
    lr_pred_cls_out.append(lr_pred_cls)
    lr_pred_prob_out.append(lr_pred_prob)
    lr_f1_matrix_out.append(lr_f1_matrix)
    lr_f1_report_out.append(lr_f1_report)
    lr_MCC_out.append(lr_MCC)
    lr_y_test_out.append(y[test])
    
    ## SVM
    svm.fit(X[train], y[train])
    ## roc curve
    svm_viz = plot_roc_curve(svm, X[test], y[test],name='K fold {}'.format(i),alpha=0.3, lw=1,ax=ax3)
    svm_interp_tpr = np.interp(svm_mean_fpr, svm_viz.fpr, svm_viz.tpr)
    svm_interp_tpr[0] = 0.0
    svm_tprs.append(svm_interp_tpr)
    svm_roc_aucs.append(svm_viz.roc_auc) 
    ## evaluation metrics 	
    svm_pred_cls = svm.predict(X[test])
    svm_pred_prob = svm.predict_proba(X[test])[:,1]
    svm_f1_matrix = confusion_matrix(y[test],svm_pred_cls)
    svm_f1_report = classification_report(y[test],svm_pred_cls)
    svm_MCC = matthews_corrcoef(y[test],svm_pred_cls)
    ### save evalu_metrics out
    svm_pred_cls_out.append(svm_pred_cls)
    svm_pred_prob_out.append(svm_pred_prob)
    svm_f1_matrix_out.append(svm_f1_matrix)
    svm_f1_report_out.append(svm_f1_report)
    svm_MCC_out.append(svm_MCC)
    svm_y_test_out.append(y[test])
    
#### save predit_prob out
np.save("CIP_gi_FCGR_RF_y_pred_prob_out.npy",rf_pred_prob_out)
np.save("CIP_gi_FCGR_RF_y_test_out.npy",rf_y_test_out)
np.save("CIP_gi_FCGR_LR_y_pred_prob_out.npy",lr_pred_prob_out)
np.save("CIP_gi_FCGR_LR_y_test_out.npy",lr_y_test_out)
np.save("CIP_gi_FCGR_SVM_y_pred_prob_out.npy",svm_pred_prob_out)
np.save("CIP_gi_FCGR_SVM_y_test_out.npy",svm_y_test_out)

#### evaluation
rf_eva_pred_prob = rf.predict_proba(pub_data2)[:,1]
lr_eva_pred_prob = lr.predict_proba(pub_data2)[:,1]
svm_eva_pred_prob = svm.predict_proba(pub_data2)[:,1]
np.save("CIP_FCGR_RF_test_y_pred_prob.npy",rf_eva_pred_prob)
np.save("CIP_FCGR_LR_test_y_pred_prob.npy",lr_eva_pred_prob)
np.save("CIP_FCGR_SVM_test_y_pred_prob.npy",svm_eva_pred_prob)
np.save("CIP_FCGR_test_y_out.npy",pub_pheno3)

#### evaluation for under sample
#pub_x_under,pub_y_under
rf_eva_under_pred_prob = rf.predict_proba(pub_x_under)[:,1]
lr_eva_under_pred_prob = lr.predict_proba(pub_x_under)[:,1]
svm_eva_under_pred_prob = svm.predict_proba(pub_x_under)[:,1]

##**************************************************************************************##
##                     Step3. Training and evaluation of CNN                            ##          
##**************************************************************************************##
############################# Step1: load pacakge #####################
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras import activations
from sklearn.model_selection import KFold,StratifiedKFold
from keras.layers import Dense,Dropout, Flatten, Conv1D, Conv2D, MaxPooling1D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import BatchNormalization

############################# Step2: load metrics function #####################
### F1 score, precision, recall and accuracy metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

############################# Step3: input data processing #####################
X.shape,y.shape,pub_data2.shape,pub_pheno3.shape
#((900, 40000),(900,), (1496, 40000),  (1496,))
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
#((720, 40000), (180, 40000), (720,), (180,))

inputs = x_train.reshape(720,200,200,1)
inputs = inputs.astype('float32')
targets = to_categorical(y_train)
inputs.shape,targets.shape

x_test2 = x_test.reshape(180,200,200,1)
x_test2 = x_test2.astype('float32')
y_test2 = to_categorical(y_test)

pub_x_test = pub_data2.reshape(1496,200,200,1)
pub_x_test = pub_x_test.astype('float32')
pub_y_test = pub_pheno3

############################# Step4: model training #####################
batch_size = 8
no_classes = 2
no_epochs = 50
verbosity = 1
num_folds = 5

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)
# K-fold Cross Validation model evaluation
fold_no = 1
model_history=[]
for train, test in kfold.split(inputs, targets):
 model = Sequential()
 model.add(Conv2D(filters=8, kernel_size=3,activation='relu', input_shape=(200,200,1)))
 model.add(BatchNormalization())
 model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'))
 #model.add(BatchNormalization())
 model.add(MaxPooling2D(pool_size=(2)))
 model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
 model.add(BatchNormalization())
 model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
 #model.add(BatchNormalization())
 model.add(MaxPooling2D(pool_size=(2)))
 model.add(Flatten())
 model.add(Dense(128, activation='relu'))
 model.add(Dropout(0.2))
 model.add(Dense(2,activation='softmax'))
 # Compile the model
 model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc',f1_m,precision_m, recall_m])
 # Generate a print
 print('--------------------------------')
 print(f'Training for fold {fold_no} ...')
 ## checkpoint for saving model
 filepath="CIP_gi_FCGR_CNN_weights.best.hdf5"
 checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
 callbacks_list = [checkpoint]
 # Fit data to model
 train_model = model.fit(inputs[train], targets[train],batch_size=batch_size,epochs=no_epochs,callbacks=callbacks_list,verbose=verbosity,validation_data=(inputs[test], targets[test]))
 model_history.append(train_model.history)
 # Increase fold number
 fold_no = fold_no + 1

########## (2) save model
model.save_weights('CIP_gi_FCGR_CNN.model.h5')  

# save model history
from pandas.core.frame import DataFrame
model_out = DataFrame(model_history)
model_out.to_csv("CIP_gi_FCGR_CNN_model_history_out.csv",index=False)

############# Evaluation on pub data
### ROC
y_pred_keras = model.predict_proba(pub_x_test)

### evaluation for under-sample

undersample = RandomUnderSampler(sampling_strategy='majority')
pub_x_under,pub_y_under=undersample.fit_resample(pub_data2,pub_pheno3)
print(Counter(pub_y_under))
pub_x_under = pub_x_under.reshape(534,200,200,1)
pub_x_under = pub_x_under.astype('float32')
y_pred_keras = model.predict_proba(pub_x_under)
