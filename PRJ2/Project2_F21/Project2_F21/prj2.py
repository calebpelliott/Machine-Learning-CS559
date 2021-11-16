import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./PRJ2/Project2_F21/Project2_F21/F21_CS559_Project2.csv')

def my_grad(X,Y,Z):
    eta=0.1 #.01,.001
    #a
    c = 1#0.0#1#0.0
    #b
    ckn = 0.0
    #c
    ckm = 0.0
    #f
    ck = 0.0
    gradient_w1=0; gradient_w2=0
    gradient_c=0
    gradient_ckn=0
    gradient_ckm=0
    gradient_ck=0
    mse = 100
    for i in range(0,10000):
        e1 = np.exp(-((X+ckn)**2 + (Y+ckm)**2))
        e2 = np.exp(-((X+ck)**2))
        e3 = np.exp(-((Y+ck)**2))
        
        z_pred = -c*(e1 + e2 + e3)
        error = (Z-z_pred)

        #handle gradients
        partial_c = (e1 + e2 + e3)
        gradient_c = np.mean((partial_c*error))

        partial_ckn = 2*c*(ckn+X)*np.exp((-1*((ckn+X)**2))-((ckm+Y)**2))
        gradient_ckn = np.mean(-(partial_ckn*error))

        partial_ckm = 2*c*(ckm+Y)*np.exp((-1*((ckn+X)**2))-((ckm+Y)**2))
        gradient_ckm = np.mean(-(partial_ckm*error))
        
        partial_ck = -2*np.exp(-1*((ck+X)**2))*(ck+X)
        partial_ck += -2*np.exp(-1*((ck+Y)**2))*(ck+Y)
        partial_ck *= -c
        gradient_ck = np.mean(-(partial_ck*error))

        #gradient_ckn = np.mean(-(e11*error))
        #gradient_ckm = np.mean(-(e12*error))
        #gradient_c = np.mean(-(z_pred*error))
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
        new_mse = np.mean(error**2)/2
        rmse = mean_squared_error(Z, z_pred, squared=False)
        if mse<=0.05:
            #print(w1,w2,np.mean(error**2)/2)
            break
    return [c, ck, ckn, ckm, mse]

for k in [0,1,2,3]:#[3,2,1,0]:
    # Train the model using the training sets
    kdf = df.loc[df['class'] == k]
    [c, ck, ckn, ckm, mse] = my_grad(kdf['x'],kdf['y'],kdf['z'])
    print(f"Class: {k}\n  C:{c}\n  Ck:{ck}\n  Ckn:{ckn}\n  Ckm:{ckm}\n  MSE:{mse}\n\n")