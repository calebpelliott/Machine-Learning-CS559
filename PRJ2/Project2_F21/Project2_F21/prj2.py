import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv('F21_CS559_Project2.csv')

def my_grad(X,Y,Z):
    eta=0.1 #.01,.001
    c = 0.01
    ckn = 0.0
    ckm = 0.0
    ck = 0.0
    gradient_w1=0; gradient_w2=0
    gradient_c=0
    gradient_ckn=0
    gradient_ckm=0
    gradient_ck=0
    for i in range(0,10000):
        #y = w1*X**2+w2*X
        e1 = np.exp(-((X+ckn)**2 + (Y+ckm)**2))
        e14 = np.exp(-(X+ckn)**2 - (Y+ckm)**2)
        e11= np.exp(-(X+ckn)**2)
        e12= np.exp(-(Y+ckm)**2)
        e13=e11*e12
        ee = sum(e1-e13)
        if ee == 0:
            print('c')
        e2 = np.exp(-((X+ck)**2))
        e3 = np.exp(-((Y+ck)**2))
        z_pred = -c*(e1 + e2 + e3)
        error = (Z-z_pred)

        #handle gradients
        gradient_ck = np.mean(-((e2+e3)*error))
        gradient_ckn = np.mean(-(e11*error))
        gradient_ckm = np.mean(-(e12*error))
        gradient_c = np.mean(-(z_pred*error))
        #gradient_w1 = np.mean(-X**2*error)
        #gradient_w2 = np.mean(-X*error)

        #update constants
        #w1 = w1-eta*gradient_w1
        #w2 = w2-eta*gradient_w2
        c = c-eta*gradient_c
        ckn = ckn-eta*gradient_ckn
        ckm = ckm-eta*gradient_ckm
        ck = ck-eta*gradient_ck
        mse = mean_squared_error(Z, z_pred)
        if mse<=0.05:
            #print(w1,w2,np.mean(error**2)/2)
            break
    return [c, ck, ckn, ckm]

for k in [0,1,2,3]:
    # Train the model using the training sets
    kdf = df.loc[df['class'] == k]
    my_grad(kdf['x'],kdf['y'],kdf['z'])