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
option_type = st.selectbox("Select Option Type: " , ["call", "put"])


col1, col2, col3, col4, col5 = st.columns(5)
S = col1.number_input("Market Price", min_value=0.0, value = 100.0, step = 1.0)
K = col2.number_input("Strike Price", min_value=0.0, value = 100.0, step = 1.0)
T = col3.number_input("Time to Maturity", min_value=0.01, value =1.0 , step= 0.01)
sigma = col4.number_input("Volatility", min_value= 0.01, value = 0.2, step = 0.01)
r = col5.number_input("Risk Free Rate", min_value = 0.01, value = 0.05, step = 0.01)

#Call and Put Variables
option_price = black_scholes(S, K, T, r, sigma, option_type)
st.header("Shock Parameters") 
spot_shock = st.slider("Spot Price Shock (%)", min_value = -99, max_value = 100, value= 0) 
vol_shock = st.slider("Volatility Shcok (%)", min_value = -99, max_value = 100, value= 0)

s_shock = S * (1 + spot_shock/100)
v_shock = sigma * (1 + vol_shock/100)

s_option = black_scholes(s_shock, K, T, r, v_shock, option_type)

col1, col2 = st.columns(2)

if option_type == "call": 
    col1.metric("Call Option Price", value = str(option_price.round(2)))
    col2.metric("Call Option Price (Shocked)", value = str(s_option.round(2)))
else: 
    col1.metric("Put Option Price", value = str(option_price.round(2)))
    col2.metric("Put Option Price (Shocked)", value = str(s_option.round(2)))
    

st.divider()
delta_value = delta(S, K, T, r, sigma, option_type)
gamma_value = gamma(S, K, T, r, sigma)
vega_value = vega(S, K, T, r, sigma)
theta_value = theta(S, K, T, r, sigma, option_type)  
rho_value = rho(S, K, T, r, sigma, option_type)

s_delta = delta(s_shock,K,T,r, v_shock, option_type)
s_gamma = gamma(s_shock,K,T,r, v_shock)
s_vega = vega(s_shock,K,T,r, v_shock)
s_theta = theta(s_shock,K,T,r, v_shock,option_type)
s_rho = rho(s_shock,K,T,r, v_shock,option_type)

if option_type == "call": 
    st.header("Call Option Greeks") 
else: 
    st.header("Put Option Greeks")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", value = str(delta_value.round(2)))
col2.metric("Gamma", value = str(gamma_value.round(2)))
col3.metric("Vega", value = str(vega_value.round(2)))
col4.metric("Theta", value = str(theta_value.round(2)))
col5.metric("Rho", value = str(rho_value.round(2)))

if option_type == "call": 
    st.header("Call Option Greeks (Shocked)") 
else: 
    st.header("Put Option Greeks (Shocked)")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta", value = str(s_delta.round(2)))
col2.metric("Gamma", value = str(s_gamma.round(2)))
col3.metric("Vega", value = str(s_vega.round(2)))
col4.metric("Theta", value = str(s_theta.round(2)))
col5.metric("Rho", value = str(s_rho.round(2)))

if option_type == "call": 
    st.header("Call Option Price Heatmap") 
else: 
    st.header("Put Option Price Heatmap")
heatmap = option_heatmap(K, T, r, option_type)
st.pyplot(heatmap)

# The next thing to do is make it so that each thing is configurable for call/put from the start
# One box at the top that give calls or puts and then everyhting follows suit
#This will help avoid the cluster of data and hopefully run time as well. 
 
#after that fix is addind a PnL Heatmap. 
#Things to consider. should the heatmap be dynamic and adjust to the shocks automatically 
#OR is it better to have 2 seperate graphs








