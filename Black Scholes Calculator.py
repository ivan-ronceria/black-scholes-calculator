# -*- coding: utf-8 -*-
"""
@author: Ivan Ronceria
"""
import numpy as np
import math
from datetime import date
from scipy.stats import norm


s = float(4255.15)
k = float(4251.92938062)
t = (date(2021, 9, 16) - date.today()).days
rf = 0.0001
v = 0.11936253



def jarrow_rudd(s, k, t, v, rf, cp, am = False, n = 1000):
    
    """ 
    Price an option using the Jarrow-Rudd binomial model.
    
    s: initial stock price
    k: strike price
    t: expiration time
    v: volatility
    rf: risk-free rate
    cp: +1/-1 for call/put
    am: True/False for American/European
    n: number of binomial steps
    """
    
    #Calculations
    h = t/n
    u = math.exp((rf - 0.5 * math.pow(v, 2)) * h + v * math.sqrt(h))
    d = math.exp((rf - 0.5 * math.pow(v, 2)) * h - v * math.sqrt(h))
    drift = math.exp(rf * h)
    q  = (drift - d)/ (u - d)
    
    #Terminal Prices
    stkval = np.zeros((n + 1, n + 1))
    optval = np.zeros((n + 1, n + 1))
    stkval[0,0] = s
    for i in range(1, n + 1):
        stkval[i, 0] = stkval[i - 1, 0] * u
        for j in range(1, i + 1):
            stkval[i, j] = stkval[i - 1, j - 1] * d
            
    #Backwards recursion for option price ("Discounting cash flows of each possibility")
    for j in range(n + 1):
        optval[n,j] = max(0, cp * (stkval[n,j] - k))
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            optval[i,j] = (q * optval[i + 1, j] + (1 - q) * optval[i + 1, j + 1]/drift)
            if am:
                optval[i,j] = max(optval[i,j], cp * (stkval[i, j] - k))

    return optval[0,0]
    

def black_scholes(stock_price, strike_price, riskless_rate, vol, ttm, call_or_put):
    """
    Parameters
    ----------
    stock_price : The underlying price at t = 0
    strike_price : The strike price of the contract
    riskless_rate : The risk-free rate
    vol : The volatility (sigma) of the stock. Many ways to estimate, but we can use implied volatility
    ttm : Time to maturity. The date of the maturity minus the initial date of entering in the position
    call_or_put : Positive 1 to indicate a call, negative 1 for a put

    Returns
    -------
    bsm_price : Returns the fair value for a vanilla European Call/Put option under the assumptions
        of the no dividend BSM model.

    """
    d_1 = (np.log(stock_price/strike_price) + (riskless_rate + ((pow(vol,2)/2))) * ttm) / (vol * math.sqrt(ttm))
    
    d_2 = d_1 - vol * math.sqrt(ttm)
    
    bsm_price = call_or_put * (stock_price * norm.cdf(call_or_put * d_1) - strike_price * math.exp(-1 * riskless_rate * ttm) * norm.cdf(call_or_put * d_2))
    
    return bsm_price


price_jr = jarrow_rudd(s, k, t, v, rf, 1, am = False)
price_bsm = black_scholes(s, k, rf, v, t, 1)

print(price_jr)
print(price_bsm)

