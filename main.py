import numpy as np
import pandas as pd
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
    get_ssvi_price,
    ssvi_variance_total,
    calculate_greeks
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
    print(f"Prix Spot détecté : {current_spot:.2f} USD")
    
    options_cleaned = process_and_filter_options(options_raw, current_spot)
    print(f"Options après filtrage : {len(options_cleaned)} / {len(options_raw)}")

    # ==========================================================
    # PHASE 2 : COURBE DES TAUX (NELSON-SIEGEL)
    # ==========================================================
    print(f"\n--- Phase 2 : Reconstruction de la courbe des taux ---")
    raw_rates = extract_implicit_rates(options_cleaned, futures_raw)
    
    if raw_rates.empty:
        print("Erreur : Impossible d'extraire les taux."); exit()

    params_ns = calibrate_nelson_siegel(raw_rates)
    b0, b1, b2, tau = params_ns
    mse_ns, mae_ns, _ = evaluate_ns_performance(raw_rates, params_ns)
    print(f"Précision Nelson-Siegel (MAE) : {mae_ns:.6f}")

    # ==========================================================
    # PHASE 3 : VOLATILITÉ IMPLICITE ET SSVI
    # ==========================================================
    print(f"\n--- Phase 3 : Calibration de la Volatilité SSVI ---")
    
    vols_list = []
    for _, row in options_cleaned.iterrows():
        # Utilisation de 'T' (calculé dans data_process)
        r_t = nelson_siegel(row['T'], b0, b1, b2, tau)
        iv = get_implied_vol(row['mid_price_usd'], current_spot, row['strike'], row['T'], r_t, row['option_type'])
        
        if not np.isnan(iv) and iv > 0:
            forward = current_spot * np.exp(r_t * row['T'])
            k = np.log(row['strike'] / forward)
            vols_list.append({
                'T': row['T'], 'k': k, 'iv': iv, 'w': (iv**2) * row['T'], 
                'strike': row['strike'], 'option_type': row['option_type'],
                'market_price': row['mid_price_usd'], 'r': r_t
            })

    df_vols = pd.DataFrame(vols_list)
    ssvi_params_storage = {} 

    for t in sorted(df_vols['T'].unique()):
        slice_data = df_vols[df_vols['T'] == t]
        if len(slice_data) < 3: continue
        
        # Variance ATM
        theta_atm = slice_data.iloc[abs(slice_data['k']).argmin()]['w']
        
        # Calibration
        rho, phi = calibrate_ssvi_slice(slice_data['k'].values, slice_data['w'].values, theta_atm)
        ssvi_params_storage[t] = {'theta': theta_atm, 'rho': rho, 'phi': phi}
        print(f"Maturité T={t:.3f} | Rho={rho:.2f} | Phi={phi:.2f}")

    # ==========================================================
    # PHASE 4 : CALCUL DES GRECQUES ET EXPORT FINAL
    # ==========================================================
    print(f"\n--- Phase 4 : Calcul des Grecques (via SSVI) ---")
    
    final_data = []
    for _, row in df_vols.iterrows():
        t = row['T']
        if t in ssvi_params_storage:
            p = ssvi_params_storage[t]
            
            # 1. Calcul de la volatilité lissée par le modèle SSVI
            w_ssvi = ssvi_variance_total(row['k'], p['theta'], p['rho'], p['phi'])
            sigma_ssvi = np.sqrt(max(0, w_ssvi / t))
            
            # 2. Calcul du prix théorique SSVI
            p_ssvi = get_ssvi_price(current_spot, row['strike'], t, row['r'], p['theta'], p['rho'], p['phi'], row['option_type'])
            
            # 3. Calcul des Grecques
            g = calculate_greeks(current_spot, row['strike'], t, row['r'], sigma_ssvi, row['option_type'])
            
            final_data.append({
                'T': t, 'Strike': row['strike'], 'Type': row['option_type'],
                'Market_Price': row['market_price'], 'SSVI_Price': p_ssvi,
                'IV_Market': row['iv'], 'IV_SSVI': sigma_ssvi,
                'Delta': g['delta'], 'Gamma': g['gamma'], 'Vega': g['vega'], 'Theta': g['theta']
            })

    df_final = pd.DataFrame(final_data)
    
    # Métriques de prix
    mae_price = np.mean(np.abs(df_final['SSVI_Price'] - df_final['Market_Price']))
    print(f"Erreur moyenne de pricing : {mae_price:.2f} USD")

    # Affichage des résultats
    print("\nAperçu des résultats (5 premières lignes) :")
    cols_show = ['T', 'Strike', 'Type', 'Delta', 'Gamma', 'Vega', 'Market_Price', 'SSVI_Price']
    print(df_final[cols_show].head().to_string(index=False))

    # Exportation CSV pour ton Github / Société Générale
    output_file = "analyse_options_complete.csv"
    df_final.to_csv(output_file, index=False)
    print(f"\nPipeline terminé. Résultats exportés dans '{output_file}'")