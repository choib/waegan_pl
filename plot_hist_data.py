import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import auc
pd.options.plotting.backend = "plotly"

#%matplotlib inline

#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1013ict0_SNUH_ICT0.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1011ict1_SNUH_ICT1.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1011ict2_SNUH_ICT2.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1007_SNUH_ICT2.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1011ict3_SNUH_ICT3.csv')
df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1014ict3_SNUH_ICT3.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1007_SNUH_ICT4.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1011ict5_SNUH_ICT5.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/100821ict6_SNUH_ICT6.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/1008ict7_SNUH_ICT7.csv')
#df=  pd.read_csv('/workspace/PyTorch-GAN/wgan-mteg/csv/100821_SNUH_ICT8.csv')

#plt.plot(xlin,auc,color="C3")
x = df.FPR
y = df.TPR
auc = 0.5*(1-x + y)
plt.scatter(x,y,color="C2")
plt.scatter(x,auc,color="C3")
plt.show()

df.plot.scatter(y='uncertainty',x='nz f')#, ylim=(0.00,0.01))#,logx=True,logy=True)
df.plot.scatter(x='FPR',y='TPR')
df.plot.scatter(x='TNR',y='TPR')
df.plot.scatter(x='f1',y='TNR')
df['PPV'].mean()
df.hist(column='f1')
df['f1'].mean()
df.hist(column='ACC')
df['ACC'].mean()
df.hist(column='iou')
df['iou'].mean()
df.hist(column='iou bb')
df['iou bb'].mean()
# plt.hist(df['f1'])
# plt.xlabel('F1 Score')
# plt.xticks([0,0.25,0.5,0.75,1])
# plt.ylabel('Number of images')
# plt.title('Histogram of F1 score')

# plt.show()