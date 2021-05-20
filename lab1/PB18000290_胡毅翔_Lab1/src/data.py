import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pl=pd.read_csv(r'adult.csv')
pl.head()
t=pd.DataFrame(pl[['age','sex','race','marital_status','occupation']],columns=['age','sex','race','marital_status','occupation'])
t.to_csv("test.csv",index=False,sep=',')