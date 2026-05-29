import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from src.SSVI import black_scholes_price, calculate_greeks, ssvi_variance_total, get_theta_t, get_phi_theta_power
from src.rates_model import nelson_siegel

def pricer_hors_grille_ssvi(S, K, T, params_ns, params_ssvi):
    """
    Price une option avec r(t) issu de Nelson-Siegel 
    et sigma(k,t) issu de la surface SSVI.
    """
    b0, b1, b2, tau = params_ns
    r_t = nelson_siegel(T, b0, b1, b2, tau)
    
    forward = S * np.exp(r_t * T)
    k = np.log(K / forward)
    
    theta_t = get_theta_t(T, params_ssvi['kappa'], params_ssvi['v0'], params_ssvi['v_inf'])
    phi_t = get_phi_theta_power(theta_t, params_ssvi['eta'], params_ssvi['lmbda'])
    
    w_pred = ssvi_variance_total(k, theta_t, params_ssvi['rho'], phi_t)
    sigma_ssvi = np.sqrt(max(1e-9, w_pred / T))
    
    prix = black_scholes_price(S, K, T, r_t, sigma_ssvi, 'call')
    grecques = calculate_greeks(S, K, T, r_t, sigma_ssvi, 'call')
    
    return prix, sigma_ssvi, r_t, grecques

def tracer_payoff_bull_call_spread(K1, K2, prime_net):
    """Trace le payoff à l'échéance du Bull Call Spread."""
    S_range = np.linspace(K1 * 0.8, K2 * 1.2, 500)
    
    # Payoff = Max(S - K1, 0) - Max(S - K2, 0) - Prime Payée
    payoff = np.maximum(S_range - K1, 0) - np.maximum(S_range - K2, 0) - prime_net
    
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, payoff, label="Payoff à l'échéance (T=0)", color="darkblue", lw=2.5)
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)
    plt.axvline(K1, color="red", linestyle=":", label=f"Strike K1 Achat ({K1})")
    plt.axvline(K2, color="green", linestyle=":", label=f"Strike K2 Vente ({K2})")
    
    plt.title("Payoff d'un Bull Call Spread Européen (Hors-Grille)", fontsize=12, fontweight='bold')
    plt.xlabel("Prix du Sous-jacent (Spot) à l'échéance", fontsize=10)
    plt.ylabel("Gain / Perte (USD)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def optimiser_portefeuille_immunise(grecques_produit, g_fut, g_opt2, g_opt3):
    """
    Trouve les poids (w_fut, w_opt2, w_opt3) pour rendre le portefeuille 
    global Delta-Gamma-Vega neutre par optimisation quadratique.
    """
    # Portefeuille global (auto-financé) = -1 * Produit + w_fut*Future + w_opt2*Opt2 + w_opt3*Opt3
    def objectif(w):
        w_fut, w_o2, w_o3 = w
        
        delta_global = -grecques_produit['delta'] + w_fut * g_fut['delta'] + w_o2 * g_opt2['delta'] + w_o3 * g_opt3['delta']
        gamma_global = -grecques_produit['gamma'] + w_fut * g_fut['gamma'] + w_o2 * g_opt2['gamma'] + w_o3 * g_opt3['gamma']
        vega_global  = -grecques_produit['vega']  + w_fut * g_fut['vega']  + w_o2 * g_opt2['vega']  + w_o3 * g_opt3['vega']
        
        # Fonction de coût avec facteurs d'échelle pour équilibrer les résidus
        return (delta_global**2) * 1e4 + (gamma_global**2) * 1e8 + (vega_global**2) * 1e4

    res = minimize(objectif, [0.0, 0.0, 0.0], method='BFGS')
    return res.x