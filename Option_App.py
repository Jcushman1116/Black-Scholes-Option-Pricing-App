# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 00:25:36 2025

@author: jcush
"""

from math import log, sqrt, exp
from scipy.stats import norm
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Function Defintions 

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def option_heatmap(K, T, r, option_type="call"):
    spot_price = np.linspace(80, 120, 21)
    volatilities = np.linspace(0.1, 0.5, 21)
    price_matrix = [
        [black_scholes(S, K, T, r, sigma, option_type) for S in spot_price]
        for sigma in volatilities
    ]
    df_prices = pd.DataFrame(price_matrix, index=volatilities, columns=spot_price)
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(df_prices, cmap='coolwarm', annot = True, fmt=".2f", xticklabels=5, yticklabels=5, ax=ax)
    ax.set_title(f"{option_type.capitalize()} Option Prices Heatmap")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    return fig

#def PnL_heatmap

#heatmap number 2
#for i, vol in enumerate(vol_range):
 #   for j, spot in enumerate(spot_range):
  #      call_price = black_scholes(spot, K, T, r, vol, option_type="call")
   ##    call_pnl[i, j] = call_price - purchase_call_price
     #   put_pnl[i, j] = put_price - purchase_put_price

def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def delta(S, K, T, r, sigma, option_type="call"):
    d_1 = d1(S, K, T, r, sigma)
    return norm.cdf(d_1) if option_type == "call" else norm.cdf(d_1) - 1
 
def gamma(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return norm.pdf(d_1) / (S * sigma * sqrt(T))

def vega(S, K, T, r, sigma):
    d_1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(d_1) * sqrt(T)

def theta(S, K, T, r, sigma, option_type="call"):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d_1 - sigma * sqrt(T)
    if option_type == "call":
        return (-S * norm.pdf(d_1) * sigma / (2 * sqrt(T)) -
                r * K * exp(-r * T) * norm.cdf(d_2))
    else:
        return (-S * norm.pdf(d_1) * sigma / (2 * sqrt(T)) +
                r * K * exp(-r * T) * norm.cdf(-d_2))

def rho(S, K, T, r, sigma, option_type="call"):
    d_2 = d1(S, K, T, r, sigma) - sigma * sqrt(T)
    if option_type == "call":
        return K * T * exp(-r * T) * norm.cdf(d_2)
    else:
        return -K * T * exp(-r * T) * norm.cdf(-d_2)


st.title("Black-Scholes Option Pricing Calculator (European Only)")
st.header("Base Option Inputs")

col1, col2, col3, col4, col5 = st.columns(5)
S = col1.number_input("Market Price", min_value=0.0, value = 100.0, step = 1.0)
K = col2.number_input("Strike Price", min_value=0.0, value = 100.0, step = 1.0)
T = col3.number_input("Time to Maturity", min_value=0.01, value =1.0 , step= 0.01)
sigma = col4.number_input("Volatility", min_value= 0.01, value = 0.2, step = 0.01)
r = col5.number_input("Risk Free Rate", min_value = 0.01, value = 0.05, step = 0.01)

#Call and Put Variables
call = black_scholes(S, K, T, r, sigma, option_type = "call")
put = black_scholes(S, K, T, r, sigma, option_type = "put")

col1, col2 = st.columns(2)
col1.metric("Call Option Price", value = str(call.round(2)))
col2.metric("Put Option Price", value = str(put.round(2)))

st.header("Shock Parameters") 
spot_shock = st.slider("Spot Price Shock (%)", min_value = -99, max_value = 100, value= 0) 
vol_shock = st.slider("Volatility Shcok (%)", min_value = -99, max_value = 100, value= 0)

s_shock = S * (1 + spot_shock/100)
v_shock = sigma * (1 + vol_shock/100)

s_call = black_scholes(s_shock, K, T, r, v_shock, option_type= "call")
s_put = black_scholes(s_shock, K, T,r, v_shock, option_type= "put")

st.divider()
col1.metric("Call Option (Shocked)", value = str(s_call.round(2)))
col2.metric("Put Option (Shocked)", value = str(s_put.round(2)))



c_delta = delta(S, K, T, r, sigma, option_type= "call")
c_gamma = gamma(S, K, T, r, sigma)
c_vega = vega(S, K, T, r, sigma)
c_theta = theta(S, K, T, r, sigma, option_type= "call")  
c_rho = rho(S, K, T, r, sigma, option_type= "call")

p_delta = delta(S, K, T, r, sigma, option_type= "put")
p_gamma = gamma(S, K, T, r, sigma,)
p_vega = vega(S, K, T, r, sigma)
p_theta = theta(S, K, T, r, sigma, option_type= "put")  
p_rho = rho(S, K, T, r, sigma, option_type= "put")


c_s_delta = delta(s_shock,K,T,r, v_shock, option_type= "call")
c_s_gamma = gamma(s_shock,K,T,r, v_shock)
c_s_vega = vega(s_shock,K,T,r, v_shock)
c_s_theta = theta(s_shock,K,T,r, v_shock,option_type= "call")
c_s_rho = rho(s_shock,K,T,r, v_shock,option_type= "call")

p_s_delta = delta(s_shock,K,T,r, v_shock, option_type= "put")
p_s_gamma = gamma(s_shock,K,T,r, v_shock)
p_s_vega = vega(s_shock,K,T,r, v_shock)
p_s_theta = theta(s_shock,K,T,r, v_shock,option_type= "put")
p_s_rho = rho(s_shock,K,T,r, v_shock,option_type= "put")

st.header("Call Option Greeks") 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", value = str(c_delta.round(2)))
col2.metric("Gamma", value = str(c_gamma.round(2)))
col3.metric("Vega", value = str(c_vega.round(2)))
col4.metric("Theta", value = str(c_theta.round(2)))
col5.metric("Rho", value = str(c_rho.round(2)))

st.header("Call Option Greeks (Shocked)") 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", value = str(c_s_delta.round(2)))
col2.metric("Gamma", value = str(c_s_gamma.round(2)))
col3.metric("Vega", value = str(c_s_vega.round(2)))
col4.metric("Theta", value = str(c_s_theta.round(2)))
col5.metric("Rho", value = str(c_s_rho.round(2)))

st.header("Put Option Greeks") 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", value = str(p_delta.round(2)))
col2.metric("Gamma", value = str(p_gamma.round(2)))
col3.metric("Vega", value = str(p_vega.round(2)))
col4.metric("Theta", value = str(p_theta.round(2)))
col5.metric("Rho", value = str(p_rho.round(2)))

st.header("Put Option Greeks (Shocked)") 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", value = str(p_s_delta.round(2)))
col2.metric("Gamma", value = str(p_s_gamma.round(2)))
col3.metric("Vega", value = str(p_s_vega.round(2)))
col4.metric("Theta", value = str(p_s_theta.round(2)))
col5.metric("Rho", value = str(p_s_rho.round(2)))
st.divider()

st.header("Option Price Heatmap")
heatmap_option = st.selectbox("Select Option Type: " , ["call", "put"])
heatmap = option_heatmap(K, T, r, option_type= heatmap_option)
st.pyplot(heatmap)

# The next thing to do is make it so that each thing is configurable for call/put from the start
# One box at the top that give calls or puts and then everyhting follows suit
#This will help avoid the cluster of data and hopefully run time as well. 
 
#after that fix is addind a PnL Heatmap. 
#Things to consider. should the heatmap be dynamic and adjust to the shocks automatically 
#OR is it better to have 2 seperate graphs








