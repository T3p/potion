import pandas as pd
import glob
from potion.envs.lqr import LQR
import numpy as np

def addJ(name, sigma=1, gamma=0.9, horizon=10):
    files = [f for f in glob.glob("*.csv") if f.startswith(name + '_')]
    dfs =  [pd.read_csv(file, index_col=False) for file in files]
      
    lq = LQR()
    lq.gamma = gamma
    lq.horizon = horizon
      
    for file, df in zip(files, dfs):
        param = df['param0']
        js = [lq.computeJ(p, sigma) for p in param]
        df['ThPerf'] = js
        df.to_csv(file, index=False, header=True)
        
def addJmulti(name, nparams=3, sigma=1, gamma=0.9, horizon=10):
    files = [f for f in glob.glob("*.csv") if f.startswith(name + '_')]
    dfs =  [pd.read_csv(file, index_col=False) for file in files]
      
    lq = LQR()
    lq.gamma = gamma
    lq.horizon = horizon
      
    for file, df in zip(files, dfs):
        param = np.reshape(df['param0'].to_numpy(), (len(df['param0']),1,1))
        for i in range(1, nparams):
            np.dstack((param, np.reshape(df['param%d' % i].to_numpy(), (len(df['param%d'%i]),1,1))))
        js = [lq.computeJ(p, sigma) for p in param]
        print(js)
        df['ThPerf'] = js
        df.to_csv(file, index=False, header=True)