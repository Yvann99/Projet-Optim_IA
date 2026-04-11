import numpy as np
import pandas as pd
from scipy.optimize import minimize

def extract_implicit_rates(df_options, df_futures):
    """
    Extrait r_T via la parité Call-Put : C - P = (F - K) * exp(-rT) [cite: 20, 21]
    """
    results = []
    
    # Utilisation du dernier prix de futur comme proxy du spot
    perp = df_futures[df_futures['instrument_name'].str.contains('PERPETUAL')]
    spot_price = perp['mid_price'].iloc[0] if not perp.empty else df_futures['mid_price'].iloc[0]

    for expiry in df_options['expiration_timestamp'].unique():
        opts = df_options[df_options['expiration_timestamp'] == expiry]
        fut = df_futures[df_futures['expiration_timestamp'] == expiry]
        
        if fut.empty: continue
            
        F_usd = fut['mid_price'].iloc[0] 
        T = (expiry - pd.Timestamp.now().timestamp() * 1000) / (365 * 24 * 3600 * 1000)
        
        if T <= 0.002: continue 

        for strike in opts['strike'].unique():
            c = opts[(opts['strike'] == strike) & (opts['option_type'] == 'call')]
            p = opts[(opts['strike'] == strike) & (opts['option_type'] == 'put')]
            
            if not c.empty and not p.empty:
                C_usd = c['mid_price'].iloc[0] * spot_price
                P_usd = p['mid_price'].iloc[0] * spot_price
                
                try:
                    denom = F_usd - strike
                    num = C_usd - P_usd
                    if abs(denom) > 0.01:
                        ratio = num / denom
                        if 0.5 < ratio < 1.5: 
                            r = -np.log(ratio) / T
                            if -0.2 < r < 1.0: # Filtre des taux réalistes
                                results.append({'T': T, 'r': r})
                except:
                    continue
                    
    df_raw = pd.DataFrame(results)
    return df_raw.groupby('T')['r'].median().reset_index() if not df_raw.empty else df_raw

def nelson_siegel(T, beta0, beta1, beta2, tau):
    """Modèle Nelson-Siegel pour lisser la courbe """
    return beta0 + (beta1 + beta2) * (tau / T) * (1 - np.exp(-T / tau)) - beta2 * np.exp(-T / tau)

def calibrate_nelson_siegel(df_rates):
    """Optimisation pour coller aux valeurs empiriques du marché """
    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0: return 1e9
        preds = nelson_siegel(df_rates['T'], b0, b1, b2, tau)
        return np.sum((df_rates['r'] - preds)**2)
    
    # Initialisation : beta0=taux long terme, beta1=pente, beta2=courbure
    res = minimize(objective, [0.05, -0.02, 0.02, 1.0])
    return res.x