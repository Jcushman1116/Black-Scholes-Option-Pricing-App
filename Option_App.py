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

def option_heatmap(S, K, T, r, sigma, option_type="call", n_points = 11):
    spot_price = np.linspace(S * 0.8, S * 1.2, n_points) 
    volatilities = np.linspace(max(0.01, sigma * 0.5), sigma * 1.5, n_points)
    price_matrix = [
        [black_scholes(S, K, T, r, sigma, option_type) for S in spot_price]
        for sigma in volatilities
    ]
    df_prices = pd.DataFrame(price_matrix, index=volatilities.round(2), columns=spot_price.round(2))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_prices, cmap='mako', annot = True, fmt=".2f", xticklabels=2, yticklabels=2, ax=ax)
    ax.set_title(f"{option_type.capitalize()} Option Prices Heatmap")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    return fig

def PnL_heatmap(S,K,T,r,sigma, option_type= "call", purchase_price = 0, position= "long", n_points= 11):
    spot_price = np.linspace(S * 0.8, S * 1.2, n_points)
    volatilities = np.linspace(max(0.01, sigma * 0.5), sigma * 1.5, n_points)
    
    if position == 'long':
        sign = 1
    else: 
        sign = -1 
    
    pnl_matrix = [[sign * (black_scholes(S_new, K, T, r, sigma_new, option_type) - purchase_price) 
                  for S_new in spot_price]
                  for sigma_new in volatilities]

    df_pnl = pd.DataFrame(pnl_matrix, index=volatilities.round(2), columns=spot_price.round(2))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_pnl, cmap='RdGn', center = 0.0, annot = True, fmt=".2f", xticklabels=2, yticklabels=2, ax=ax)
    ax.set_title(f"{option_type.capitalize()} Option PnL Heatmap")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    return fig


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
position = st.selectbox("Select Postion Type: ", ["long","short"])


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

n_points = st.slider("Heatmap Grid Size", min_value=5, max_value=25, value=11)
if option_type == "call": 
    st.header("Call Option Price Heatmap") 
else: 
    st.header("Put Option Price Heatmap")
heatmap = option_heatmap(s_shock,K, T, r, v_shock,option_type, n_points)
st.pyplot(heatmap)

st.divider()

if option_type == "call":
    st.header("Call Option PnL Hetamap")
else:
    st.header("Put Option PnL Hetamap")

purchase_price = st.number_input("Puchase Price", min_value= 0.0, value= 0.0, step = 1.0)

Heatmap_2 = PnL_heatmap(s_shock, K, T, r, v_shock, option_type, purchase_price, position, n_points)
st.pyplot(Heatmap_2)

'''
To Do: 
    - ensure that the PnL graph is workign properly with an example and add more text to descibe the program with examples
'''







