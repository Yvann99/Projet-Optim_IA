import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.data_process import get_deribit_data, process_and_filter_options
from src.rates_model import (
    extract_implicit_rates, 
    calibrate_nelson_siegel, 
    nelson_siegel, 
    evaluate_ns_performance
)
from src.SSVI import (
    get_implied_vol, 
    calibrate_ssvi_slice, 
    get_ssvi_price
)

if __name__ == "__main__":
    CURRENCY = "BTC"
    
    # ==========================================================
    # PHASE 1 : RÉCUPÉRATION ET NETTOYAGE
    # ==========================================================
    print(f"--- Phase 1 : Récupération et Nettoyage ({CURRENCY}) ---")
    options_raw, futures_raw = get_deribit_data(CURRENCY)
    
    perp = futures_raw[futures_raw['instrument_name'].str.contains('PERPETUAL')]
    current_spot = perp['mid_price'].iloc[0] if not perp.empty else futures_raw['mid_price'].iloc[0]
    print(f"Prix Spot détecté : {current_spot} USD")
    
    options_cleaned = process_and_filter_options(options_raw, current_spot)
    print(f"Options après filtrage : {len(options_cleaned)} / {len(options_raw)}")

    # ==========================================================
    # PHASE 2 : COURBE DES TAUX (NELSON-SIEGEL)
    # ==========================================================
    print(f"\n--- Phase 2 : Reconstruction de la courbe des taux ---")
    raw_rates = extract_implicit_rates(options_cleaned, futures_raw)
    
    if raw_rates.empty:
        print("Erreur : Impossible d'extraire les taux. Fin du programme."); exit()

    params_ns = calibrate_nelson_siegel(raw_rates)
    b0, b1, b2, tau = params_ns
    mse_ns, mae_ns, df_rates_comp = evaluate_ns_performance(raw_rates, params_ns)
    print(f"Précision Nelson-Siegel (MAE) : {mae_ns:.6f}")

    # ==========================================================
    # PHASE 3 : VOLATILITÉ IMPLICITE ET SSVI
    # ==========================================================
    print(f"\n--- Phase 3 : Calibration de la Volatilité SSVI ---")
    
    results_vols = []
    # 1. Calcul de l'IV pour chaque option (Newton-Raphson / Dichotomie)
    for _, row in options_cleaned.iterrows():
        r_t = nelson_siegel(row['T'], b0, b1, b2, tau)
        iv = get_implied_vol(row['mid_price_usd'], current_spot, row['strike'], row['T'], r_t, row['option_type'])
        
        if not np.isnan(iv) and iv > 0:
            forward = current_spot * np.exp(r_t * row['T'])
            k = np.log(row['strike'] / forward)
            w_market = (iv**2) * row['T']
            results_vols.append({
                'T': row['T'], 'k': k, 'iv': iv, 'w': w_market, 
                'strike': row['strike'], 'option_type': row['option_type'],
                'market_price': row['mid_price_usd'], 'r': r_t
            })

    df_vols = pd.DataFrame(results_vols)
    
    # 2. Calibration SSVI par maturité et Comparaison des Prix
    final_comparison = []
    ssvi_params_storage = {} # Pour stocker rho et phi par maturité

    for t in sorted(df_vols['T'].unique()):
        slice_data = df_vols[df_vols['T'] == t]
        if len(slice_data) < 3: continue # Besoin de points pour calibrer
        
        # Trouver Theta ATM (variance au strike le plus proche du forward)
        theta_atm = slice_data.iloc[abs(slice_data['k']).argmin()]['w']
        
        # Calibration rho, phi
        rho, phi = calibrate_ssvi_slice(slice_data['k'].values, slice_data['w'].values, theta_atm)
        ssvi_params_storage[t] = {'theta': theta_atm, 'rho': rho, 'phi': phi}
        
        print(f"Maturité T={t:.3f} | Rho={rho:.2f} | Phi={phi:.2f}")

        # 3. Calcul des prix SSVI pour comparaison
        for _, row in slice_data.iterrows():
            p_ssvi = get_ssvi_price(current_spot, row['strike'], t, row['r'], theta_atm, rho, phi, row['option_type'])
            final_comparison.append({
                'T': t, 'Strike': row['strike'], 'Type': row['option_type'],
                'Market_Price': row['market_price'], 'SSVI_Price': p_ssvi,
                'Error_USD': p_ssvi - row['market_price']
            })

   
    # RÉSULTATS FINAUX ET EXPORT
    df_final = pd.DataFrame(final_comparison)
    mae_price = np.mean(np.abs(df_final['Error_USD']))
    
    print(f"\n--- Analyse de Performance Finale ---")
    print(f"Erreur Moyenne sur les prix (MAE) : {mae_price:.2f} USD")
    print("\nÉchantillon des écarts (10 premières lignes) :")
    print(df_final[['T', 'Strike', 'Market_Price', 'SSVI_Price', 'Error_USD']].head(10).to_string(index=False))

    # Sauvegarde des résultats pour le GitHub
    df_final.to_csv("resultats_calibration_finale.csv", index=False)
    print("\nDonnées sauvegardées dans 'resultats_calibration_finale.csv'")