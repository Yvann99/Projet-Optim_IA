import numpy as np
import pandas as pd
from scipy.optimize import minimize

def extract_implicit_rates(df_options, df_futures):
    """
    Extrait les taux r_T en utilisant la parité Call-Put :
    C - P = (F - K) * exp(-r * T)
    """
    results = []
    
    # On itère par maturité (T)
    for expiry in df_options['expiration_timestamp'].unique():
        # Sélection des options et du futur pour cette échéance
        opts = df_options[df_options['expiration_timestamp'] == expiry]
        fut = df_futures[df_futures['expiration_timestamp'] == expiry]
        
        if fut.empty:
            continue
            
        F = fut['mid_price'].iloc[0]
        T = (expiry - pd.Timestamp.now().timestamp() * 1000) / (365 * 24 * 3600 * 1000) # T en années
        
        if T <= 0: continue

        # On cherche les couples Call/Put de même strike
        for strike in opts['strike'].unique():
            c = opts[(opts['strike'] == strike) & (opts['option_type'] == 'call')]
            p = opts[(opts['strike'] == strike) & (opts['option_type'] == 'put')]
            
            if not c.empty and not p.empty:
                C_price = c['mid_price'].iloc[0]
                P_price = p['mid_price'].iloc[0]
                
                # Formule isolant r : r = -ln((C - P) / (F - K)) / T
                try:
                    val = (C_price - P_price) / (F - strike)
                    if val > 0:
                        r = -np.log(val) / T
                        results.append({'T': T, 'r': r})
                except:
                    continue
                    
    return pd.DataFrame(results).groupby('T').mean().reset_index()

def nelson_siegel(T, beta0, beta1, beta2, tau):
    """Modèle Nelson-Siegel pour lisser la courbe"""
    return beta0 + (beta1 + beta2) * (tau / T) * (1 - np.exp(-T / tau)) - beta2 * np.exp(-T / tau)

def calibrate_nelson_siegel(df_rates):
    """Trouve les paramètres beta et tau minimisant l'erreur quadratique"""
    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0: return 1e9
        preds = nelson_siegel(df_rates['T'], b0, b1, b2, tau)
        return np.sum((df_rates['r'] - preds)**2)
    
    # Initialisation raisonnable
    res = minimize(objective, [0.05, -0.02, 0.02, 1.0])
    return res.x # Retourne b0, b1, b2, tau