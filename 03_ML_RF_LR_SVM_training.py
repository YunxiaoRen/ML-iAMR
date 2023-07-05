##**************************************************************************************##
##                     Step1. Load Packages and Input Data                              ##          
##**************************************************************************************##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,metrics
from sklearn.svm import SVC,LinearSVC 
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.metrics import matthews_corrcoef,auc, roc_curve,plot_roc_curve, plot_precision_recall_curve,classification_report, confusion_matrix,average_precision_score, precision_recall_curve
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter


############################# input data processing #####################
data = pd.read_csv("data.csv") 
pheno = pd.read_csv("pheno.csv",index_col=0)   
data.shape,pheno.shape
data2 = data.values
pheno2 = pheno.values.reshape(pheno.shape[0],)
data2.shape,pheno2.shape
X_train, X_test, y_train, y_test = train_test_split(data2, pheno2, test_size= 0.20, stratify= pheno2, random_state= 123)
X = X_train
y = y_train
X.shape,y.shape

##**************************************************************************************##
##                     Step2. Training and evaluation of RF,LR, SVM                     ##          
##**************************************************************************************##

## cross validation
cv = StratifiedKFold(n_splits=5)
rf = RandomForestClassifier(n_estimators=200, random_state=0)
lr = LogisticRegression(solver = 'lbfgs',max_iter=1000)
svm = SVC(kernel='linear', probability=True)

##*************** F1 + ROC curve
## RF
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
    

### save mcc and f1-report
rf_mcc = pd.DataFrame(rf_MCC_out) 
lr_mcc = pd.DataFrame(lr_MCC_out) 
svm_mcc = pd.DataFrame(svm_MCC_out) 
all_mcc = pd.concat([rf_mcc,lr_mcc,svm_mcc],axis=1)
all_mcc.columns = ['RF_MCC','LR_MCC','SVM_MCC']
all_mcc.to_csv("Train_MCC.csv",index=False)   

rf_f1_report = pd.DataFrame(rf_f1_report_out)
rf_f1_report.columns=['RF'] 
lr_f1_report = pd.DataFrame(lr_f1_report_out) 
lr_f1_report.columns=['LR']
svm_f1_report = pd.DataFrame(svm_f1_report_out) 
svm_f1_report.columns=['SVM']
all_f1_report = pd.concat([rf_f1_report,lr_f1_report,svm_f1_report],axis=0)   
all_f1_report.to_csv("Train_F1_report.csv",index=False)   


###************************** plot training results and save mean tpr and fpr **********************************
######### RF
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
rf_mean_tpr = np.mean(rf_tprs, axis=0)
rf_mean_tpr[-1] = 1.0
rf_mean_auc = auc(rf_mean_fpr, rf_mean_tpr)
rf_std_auc = np.std(rf_roc_aucs)
ax1.plot(rf_mean_fpr, rf_mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (rf_mean_auc, rf_std_auc),lw=2, alpha=.8)
rf_std_tpr = np.std(rf_tprs, axis=0)
rf_tprs_upper = np.minimum(rf_mean_tpr + rf_std_tpr, 1)
rf_tprs_lower = np.maximum(rf_mean_tpr - rf_std_tpr, 0)
ax1.fill_between(rf_mean_fpr, rf_tprs_lower, rf_tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="RF ROC Curve")
ax1.legend(loc="lower right",prop = {'size':6})

#plt.savefig("Train_RF_ROC.pdf")  
### save mean FPR and TPR
#np.savetxt("Train_RF_mean_fpr.csv",rf_mean_fpr,delimiter=",")
#np.savetxt("Train_RF_mean_tpr.csv",rf_mean_tpr,delimiter=",")

######### LR
ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
lr_mean_tpr = np.mean(lr_tprs, axis=0)
lr_mean_tpr[-1] = 1.0
lr_mean_auc = auc(lr_mean_fpr, lr_mean_tpr)
lr_std_auc = np.std(lr_roc_aucs)
ax2.plot(lr_mean_fpr, lr_mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (lr_mean_auc, lr_std_auc),lw=2, alpha=.8)
lr_std_tpr = np.std(lr_tprs, axis=0)
lr_tprs_upper = np.minimum(lr_mean_tpr + lr_std_tpr, 1)
lr_tprs_lower = np.maximum(lr_mean_tpr - lr_std_tpr, 0)
ax2.fill_between(lr_mean_fpr, lr_tprs_lower, lr_tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="LR ROC Curve")
ax2.legend(loc="lower right",prop = {'size':6})
#plt.savefig("Train_LR_ROC.pdf")  
### save mean FPR and TPR
#np.savetxt("Train_LR_mean_fpr.csv",lr_mean_fpr,delimiter=",")
#np.savetxt("Train_LR_mean_tpr.csv",lr_mean_tpr,delimiter=",")

######### SVM
ax3.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
svm_mean_tpr = np.mean(svm_tprs, axis=0)
svm_mean_tpr[-1] = 1.0
svm_mean_auc = auc(svm_mean_fpr, svm_mean_tpr)
svm_std_auc = np.std(svm_roc_aucs)
ax3.plot(svm_mean_fpr, svm_mean_tpr, color='b',label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (svm_mean_auc, svm_std_auc),lw=2, alpha=.8)
svm_std_tpr = np.std(svm_tprs, axis=0)
svm_tprs_upper = np.minimum(svm_mean_tpr + svm_std_tpr, 1)
svm_tprs_lower = np.maximum(svm_mean_tpr - svm_std_tpr, 0)
ax3.fill_between(svm_mean_fpr, svm_tprs_lower, svm_tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
ax3.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],title="SVM ROC Curve")
ax3.legend(loc="lower right",prop = {'size':6})
plt.savefig("Train_RF_LR_SVM_ROC.pdf")  
### save mean FPR and TPR
#np.savetxt("Train_SVM_mean_fpr.csv",svm_mean_fpr,delimiter=",")
#np.savetxt("Train_SVM_mean_tpr.csv",svm_mean_tpr,delimiter=",")


####******************** Evaluation on test data ************************************************
# pre_class
rf_eva_pred = rf.predict(X_test)
lr_eva_pred = lr.predict(X_test)
svm_eva_pred = svm.predict(X_test)
# pre_prob
rf_eva_pred_prob = rf.predict_proba(X_test)[:,1]
lr_eva_pred_prob = lr.predict_proba(X_test)[:,1]
svm_eva_pred_prob = svm.predict_proba(X_test)[:,1]


## save f1 report,confusion_matrix, MCC
f=open("Test_RF_LR_SVM_eva_metrics.csv","w")
# f1 report
print(classification_report(y_test,rf_eva_pred),file=f)
print(classification_report(y_test,lr_eva_pred),file=f)
print(classification_report(y_test,svm_eva_pred),file=f)
# confusion_matrix
print(confusion_matrix(y_test,rf_eva_pred),file=f)
print(confusion_matrix(y_test,lr_eva_pred),file=f)
print(confusion_matrix(y_test,svm_eva_pred),file=f)
# MCC
print(matthews_corrcoef(y_test,rf_eva_pred),file=f)
print(matthews_corrcoef(y_test,lr_eva_pred),file=f)
print(matthews_corrcoef(y_test,svm_eva_pred),file=f)
f.close()


## Plot ROC curve and PR curve
## ROC
fig,[ax4,ax5,ax6] = plt.subplots(nrows=1,ncols=3,figsize=(15, 4))
ax4.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
rf_fpr, rf_tpr,thresholds = roc_curve(y_test,rf_eva_pred_prob)
rf_roc_auc = metrics.auc(rf_fpr,rf_tpr)
ax4.plot(rf_fpr,rf_tpr,label= 'ROC (AUC = {:.3f} )'.format(rf_roc_auc))
ax4.set(xlabel='False Postive Rate',ylabel='True Positive Rate',title="RF ROC Curve")
ax4.legend(loc='lower right')

ax5.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
lr_fpr, lr_tpr,thresholds = roc_curve(y_test,lr_eva_pred_prob)
lr_roc_auc = metrics.auc(lr_fpr,lr_tpr)
ax5.plot(lr_fpr,lr_tpr,label= 'ROC (AUC = {:.3f} )'.format(lr_roc_auc))
ax5.set(xlabel='False Postive Rate',ylabel='True Positive Rate',title="LR ROC Curve")
ax5.legend(loc='lower right')

ax6.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
svm_fpr, svm_tpr,thresholds = roc_curve(y_test,svm_eva_pred_prob)
svm_roc_auc = metrics.auc(svm_fpr,svm_tpr)
ax6.plot(svm_fpr,svm_tpr,label= 'ROC (AUC = {:.3f} )'.format(svm_roc_auc))
ax6.set(xlabel='False Postive Rate',ylabel='True Positive Rate',title="SVM ROC Curve")
ax6.legend(loc='lower right')
plt.savefig("Test_RF_LR_SVM_ROC_curve.pdf",bbox_inches='tight')


###### PR curve
fig,[ax7,ax8,ax9] = plt.subplots(nrows=1,ncols=3,figsize=(15, 4))
rf_precision,rf_recall,thresholds = precision_recall_curve(y_test,rf_eva_pred_prob)
rf_pr_auc = metrics.auc(rf_recall,rf_precision)
ax7.plot([0,1],[1,0],linestyle="--",lw=2,color='k',alpha=.8)
ax7.plot(rf_recall,rf_precision,label= '(AUCPR = {:.3f})'.format(rf_pr_auc))
ax7.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel='Recall',ylabel='Precision',title="RF Precision-Recall Curve")
ax7.legend(loc='lower left')

lr_precision,lr_recall,thresholds = precision_recall_curve(y_test,lr_eva_pred_prob)
lr_pr_auc = metrics.auc(lr_recall,lr_precision)
ax8.plot([0,1],[1,0],linestyle="--",lw=2,color='k',alpha=.8)
ax8.plot(lr_recall,lr_precision,label= '(AUCPR = {:.3f})'.format(lr_pr_auc))
ax8.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel='Recall',ylabel='Precision',title="LR Precision-Recall Curve")
ax8.legend(loc='lower left')

svm_precision,svm_recall,thresholds = precision_recall_curve(y_test,svm_eva_pred_prob)
svm_pr_auc = metrics.auc(svm_recall,svm_precision)
ax9.plot([0,1],[1,0],linestyle="--",lw=2,color='k',alpha=.8)
ax9.plot(svm_recall,svm_precision,label= '(AUCPR = {:.3f})'.format(svm_pr_auc))
ax9.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel='Recall',ylabel='Precision',title="SVM Precision-Recall Curve")
ax9.legend(loc='lower left')
plt.savefig("Test_RF_LR_SVM_PR_curve.pdf",bbox_inches='tight')

