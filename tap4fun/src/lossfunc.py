from sklearn.metrics import precision_recall_curve
import pandas as pd

def maxRecall(preds,dtrain): #preds是结果（概率值），dtrain是个带label的DMatrix
    labels=dtrain.get_label() #提取label
    preds=1-preds
    precision,recall,threshold=precision_recall_curve(labels,preds,pos_label=0)
    pr=pd.DataFrame({'precision':precision,'recall':recall})
    return 'Max Recall:',pr[pr.precision>=0.97].recall.max()

def custom_loss(y_pre,D_label): #别人的自定义损失函数
    label=D_label.get_label()
    penalty=2.0
    grad=-label/y_pre+penalty*(1-label)/(1-y_pre) #梯度
    hess=label/(y_pre**2)+penalty*(1-label)/(1-y_pre)**2 #2阶导
    return grad,hess    