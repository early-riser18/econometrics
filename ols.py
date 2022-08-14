import pandas as pd
import numpy as np
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, geom_abline

class ols_univariate:
    def __init__(self, data, x, y):
        self.data = data
        self.x = x
        self.y = y
        self.beta1 = None
        self.beta0 = None
        self.sst = None
        self.sse = None
        self.ssr = None
        
    def compute(self, x):
        return self.beta0 + self.beta1 * x
    
    
    
    def regress(self):
        '''
        Takes the regressor and the regressed variables as panda series and store the output regression function and relevant properties
        '''
        
        X = self.data[f'{self.x}'].mean()
        Y = self.data[f'{self.y}'].mean()

        beta1_numerator = self.data.apply(lambda c: (
        c[f'{self.x}'] - X) 
        *(c[f'{self.y}'] - Y),
        axis=1).sum()
        
        
        beta1_denominator =  self.data.apply(lambda c: (c[f'{self.x}'] - X)**2, axis=1).sum()
        
        beta1 = beta1_numerator /beta1_denominator 
        beta0 = Y - (beta1*X)
        
        self.beta1 = beta1
        self.beta0 = beta0
        
        SST = self.data.apply(lambda c: (c[f'{self.y}'] - Y)**2, axis=1).sum() 
        SSE = self.data.apply(lambda c: (self.compute(c[f'{self.x}']) - Y)**2, axis=1).sum() 
        SSR = self.data.apply(lambda c: (c[f'{self.y}'] - self.compute(c[f'{self.x}']))**2, axis=1).sum()
        
        self.sst = SST
        self.sse = SSE
        self.ssr = SSR

    
    def show_plot(self):
        Y = self.data[f'{self.y}'].mean()
        return (
        ggplot(self.data, aes(x=self.x,y=self.y)) 
         + geom_point() 
         + geom_abline(aes(slope=self.beta1, intercept=self.beta0 ),color='blue')
         + geom_abline(aes(slope=0, intercept=Y), color='red')

        )