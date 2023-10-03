# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:29:31 2023

@author: XYZW
"""
import numpy as np
import scipy.stats as stats
import calibration_piecewise_expo as PE
#%%
def spread_CDS(N,r,recov,T,freq,expo_rvs,t = 0,RPV01 = False,components = False):
    ns = np.shape(expo_rvs)[0]
    times = [i/freq for i in range(0,int(freq*T)+1)]
    no = int(T*freq)
    default_leg_sims = [(1-recov)*N*np.exp(-r*(expo_rvs[i]-t))*(expo_rvs[i]<T)
                        *(expo_rvs[i]>t) for i in range(ns)]
    premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) 
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    if RPV01 == False and components == False:
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims)
    elif RPV01 == False and components == True:
        spread = np.mean(default_leg_sims)/np.mean(premium_leg_sims)
        default_leg = np.mean(default_leg_sims)
        premium_leg = np.mean(premium_leg_sims)
        return spread,default_leg,premium_leg
    elif RPV01 == True and components == False:
        rpv01 = np.mean(premium_leg_sims)/N
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims),rpv01
    elif RPV01==True and components==True:
        rpv01 = np.mean(premium_leg_sims)/N
        spread = np.mean(default_leg_sims)/np.mean(premium_leg_sims)
        default_leg = np.mean(default_leg_sims)
        premium_leg = np.mean(premium_leg_sims)
        return spread,default_leg,premium_leg,rpv01
    
def spread_CDS_accrued(N,r,recov,T,freq,expo_rvs,t = 0,RPV01 = True,components = False):
    """
    Include accrued premium in the CDS contract. 
    
    
    """
    ns = np.shape(expo_rvs)[0]
    times = [i/freq for i in range(0,int(freq*T)+1)]
    no = int(T*freq)
    default_leg_sims = [(1-recov)*N*np.exp(-r*(expo_rvs[i]-t))*(expo_rvs[i]<T)
                        *(expo_rvs[i]>t) for i in range(ns)]
    premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) + N*(
        expo_rvs[j]>times[i-1])*(expo_rvs[j]<times[i])* (expo_rvs[j]-times[i-1])
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    if RPV01 == False and components == False:
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims)
    elif RPV01 == False and components == True:
        spread = np.mean(default_leg_sims)/np.mean(premium_leg_sims)
        default_leg = np.mean(default_leg_sims)
        premium_leg = np.mean(premium_leg_sims)
        return spread,default_leg,premium_leg
    elif RPV01 == True and components == False:
        rpv01 = np.mean(premium_leg_sims)/N
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims),rpv01
    elif RPV01==True and components==True:
        rpv01 = np.mean(premium_leg_sims)/N
        spread = np.mean(default_leg_sims)/np.mean(premium_leg_sims)
        default_leg = np.mean(default_leg_sims)
        premium_leg = np.mean(premium_leg_sims)
        return spread,default_leg,premium_leg,rpv01
    
    
def spread_CDS3(N,r,recov,T,freq,unif_rvs,lbd,gamma = 1.0, t = 0, 
                model = 'exponential',RPV01 = False):
    ns = np.shape(unif_rvs)[0]
    if str.lower(model) in ['exponential','expo']:
        rvs = -np.log(1-unif_rvs)/lbd
    elif str.lower(model) in ['weibull','we']:
        rvs = (-np.log(unif_rvs)/lbd)**(1/gamma)
    elif str.lower(model) in ['lognormal','log-normal']:
        rvs = np.exp(stats.isf(unif_rvs)/gamma)/lbd
    elif str.lower(model) in ['loglogistic','log-logistic','log logistic']:
        rvs = ((1-unif_rvs)/(unif_rvs*lbd))**gamma
    elif str.lower(model) in ['piecewise exponential','pe']:
        PE_obj = PE.piecewise_exponential(lbd[0],lbd[1])
        rvs = PE_obj.rvs(unif_rvs)
    
    times = [i/freq for i in range(0,int(freq*T)+1)]
    no = int(T*freq)
    default_leg_sims = [(1-recov)*N*np.exp(-r*(rvs[i]-t))*(rvs[i]<T)
                        *(rvs[i]>t) for i in range(ns)]
    premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (rvs[j]>times[i])*np.exp(-r*(times[i]-t)) 
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    if RPV01 == False:
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims)
    else:
        rpv01 = np.mean(premium_leg_sims)/N
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims),rpv01
    
    
#%%
def binary_CDS_spread(N,r,recov,T,freq,lbd,unif_rvs,H,t = 0, RPV01 = False,accrued = 'No'):
    """
    Find the spread of a binary CDS
    
    Input/Parameters:
        N: notional
        
        r: risk free rate
        
        recov: recovery rate
        
        T: expiry of the binary CDS
        
        lbd: 
    """
    ns = np.shape(unif_rvs)[0]
    expo_rvs = -np.log(1-unif_rvs)/lbd
    times = [i/freq for i in range(0,int(freq*T)+1)]
    no = int(T*freq)
    default_leg_sims = [H*np.exp(-r*(expo_rvs[i]-t))*(expo_rvs[i]<T)
                        *(expo_rvs[i]>t) for i in range(ns)]
    if accrued == 'No':
        premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) 
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    else:
        premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                    (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) + N*(
            expo_rvs[j]>times[i-1])*(expo_rvs[j]<times[i])* (expo_rvs[j]-times[i-1])
                    for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    if RPV01==False:
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims)
    else:
        rpv01 = np.mean(premium_leg_sims)/N
        spread = np.mean(default_leg_sims)/np.mean(premium_leg_sims)
        return spread,rpv01
    
def binary_CDS_value(N,r,recov,T,freq,lbd,unif_rvs,H,spread,t = 0):
    ns = np.shape(unif_rvs)[0]
    expo_rvs = -np.log(1-unif_rvs)/lbd
    times = [i/freq for i in range(0,int(freq*T)+1)]
    no = int(T*freq)
    default_leg_sims = [H*np.exp(-r*(expo_rvs[i]-t))*(expo_rvs[i]<T)
                        *(expo_rvs[i]>t) for i in range(ns)]
    premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) 
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    return np.mean(premium_leg_sims)*spread-np.mean(default_leg_sims)
    
def spread_CDS2(N,r,recov,T,freq,lbd,t = 0,ns =1000,RPV01 = False):
    unif_rvs = stats.uniform.rvs(0,1,size = 1000)
    expo_rvs = -np.log(1-unif_rvs)/lbd
    times = [i/freq for i in range(0,int(freq*T)+1)]
    no = int(T*freq)
    default_leg_sims = [(1-recov)*N*np.exp(-r*(expo_rvs[i]-t))*(expo_rvs[i]<T)
                        *(expo_rvs[i]>t) for i in range(ns)]
    premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) 
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    if RPV01==False:
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims)
    else:
        rpv01 = np.mean(premium_leg_sims)/N
        spread = np.mean(default_leg_sims)/np.mean(premium_leg_sims)
        return spread,rpv01
    
def deferred_spread_CDS(start,T,freq,R,lbd,rf,t = 0):
    N = int((T-start)*freq)
    times = [0]+[start+i/freq for i in range(1,N+1)]
    #print(times)
    deferred_spread = (1-R)*lbd/(lbd+rf)*(1-np.exp(-(rf+lbd)*(T-t)))/np.dot(
        np.diff(times)*(np.array(times)[1:N+1]>t),
        np.exp(-(rf+lbd)*np.array(times)[1:N+1]*(np.array(times)[1:N+1]>t)))
    return deferred_spread

def deferred_spread_CDS2(notional,start,T,freq,R,rf,expo_rvs,t = 0,RPV01 = False):
    N = int((T-start)*freq)
    ns = np.shape(expo_rvs)[0]
    times = [start+i/freq for i in range(1,N+1)]
    default_leg_sims = [(1-R)*notional* np.exp(-rf*expo_rvs[i])*(expo_rvs[i]<T) 
                        for i in range(ns)]
    times = [0]+times
    premium_leg_sims = [sum([notional*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-rf*(times[i]-t)) 
                for i in range(1,N+1) if times[i]>t]) for j in range(ns)]
    if RPV01 == False:
        return np.mean(default_leg_sims)/np.mean(premium_leg_sims)
    else:
        rpv01 = np.mean(premium_leg_sims)/notional
        spread = np.mean(default_leg_sims)/np.mean(premium_leg_sims)
        return spread,rpv01

def value_deferred_spread_CDS(notional,start,T,freq,R,rf,unif_rvs,lbd,spread,t = 0,
                              model = 'exponential'):
    N = int((T-start)*freq)
    expo_rvs = -np.log(1-unif_rvs)/lbd
    ns = np.shape(expo_rvs)[0]
    times = [start+i/freq for i in range(1,N+1)]
    default_leg_sims = [(1-R)*notional* np.exp(-rf*expo_rvs[i])*(expo_rvs[i]<T) 
                        for i in range(ns)]
    times = [0]+times
    premium_leg_sims = [sum([notional*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-rf*(times[i]-t)) 
                for i in range(1,N+1) if times[i]>t]) for j in range(ns)]
    return np.mean(premium_leg_sims)*spread-np.mean(default_leg_sims)


def value_CDS2(N,r,recov,T,freq,lbd,spread,unif_rvs,gamma = 1.0,t = 0,ns = 1000,
               CDS_type = 'seller', RPV01=True,model = 'exponential'):
    """
    Parameters:
        N: notional outstanding on which the CDS is applied
        
        recov: recovery rate 
        
        r: risk_free_rate applied to all cash-flows
        
        freq: frequency of payments
        
        lbd: intensity parameters
        
        spread: the premium actually paid in the CDS value
        
        gamma: scale parameter. Useful in all distributions except Exponential
        
        unif_rvs: the uniform sample from which one should start
        
        Model: By default is exponential. 
                It can also be Weibull, Log-Logistic, Log-Normal, Gompertz
    Returns:
        Necessary upfront in a non-standard CDS whose initiation value is 
        different from zero due to different spread than the usual market spread. 
    """
    times = [i/freq for i in range(1,int(freq*T)+1)]
    
    no = int(T*freq)
    
    if str.lower(model)=='exponential':
        expo_rvs = -np.log(1-unif_rvs)/lbd
    elif str.lower(model)=='weibull':
        expo_rvs = (-np.log(unif_rvs)/lbd)**(1/gamma)
    elif str.lower(model) in ['lognormal','log-normal']:
        expo_rvs = np.exp(stats.isf(unif_rvs)/gamma)/lbd
    elif str.lower(model) in ['loglogistic','log-logistic','log logistic']:
        expo_rvs = ((1-unif_rvs)/(unif_rvs*lbd))**gamma
    elif str.lower(model) in ['gompertz']:
        expo_rvs = np.log(1-np.log(unif_rvs)/lbd)/gamma
    
    default_leg_sims = [(1-recov)*N*np.exp(-r*(expo_rvs[i]-t))*(expo_rvs[i]<T)
                        *(expo_rvs[i]>t) for i in range(ns)]
    
    times = [0]+times
    
    premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) 
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    
    if RPV01==False:
        return spread*np.mean(premium_leg_sims)-np.mean(default_leg_sims)
    else:
        rpv01 = np.mean(premium_leg_sims)/N
        return spread*rpv01*N-np.mean(default_leg_sims),rpv01
    
    
def value_CDS(N,r,recov,T,freq,expo_rvs,spread,t = 0,CDS_type = 'seller',
              RPV01 = False):
    """
    Parameters:
        N: notional, double number
        
        r: interest rate
        
        recov: recovery rate
        
        T: length of a contract (in years)
        
        freq: frequency of premium payments / year
        
        spread: value of the premium (in percentage)
        
        expo_rvs: set of random variates
    """
    ns = np.shape(expo_rvs)[0]
    
    times = [i/freq for i in range(1,int(freq*T)+1)]
    
    no = int(T*freq)
    
    default_leg_sims = [(1-recov)*N*np.exp(-r*(expo_rvs[i]-t))*(expo_rvs[i]<T)
                        *(expo_rvs[i]>t) for i in range(ns)]
    
    times = [0]+times
    
    premium_leg_sims = [sum([N*(times[i]-times[i-1])*
                (expo_rvs[j]>times[i])*np.exp(-r*(times[i]-t)) 
                for i in range(1,no+1) if times[i]>t]) for j in range(ns)]
    
    if RPV01==False:
        return spread*np.mean(premium_leg_sims)-np.mean(default_leg_sims)
    else:
        rpv01 = np.mean(premium_leg_sims)/N
        return spread*rpv01*N-np.mean(default_leg_sims),rpv01