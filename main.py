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
    calibrate_full_surface_ssvi,  # Nouvelle fonction globale
    ssvi_phi_function,           # Pour reconstruire phi(theta)
    ssvi_variance_total,
    get_ssvi_price,
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
    _, mae_ns, _ = evaluate_ns_performance(raw_rates, params_ns)
    print(f"Précision Nelson-Siegel (MAE) : {mae_ns:.6f}")

    # ==========================================================
    # PHASE 3 : VOLATILITÉ IMPLICITE ET SURFACE SSVI GLOBALE
    # ==========================================================
    print(f"\n--- Phase 3 : Calibration de la Surface SSVI Globale ---")
    
    vols_list = []
    for _, row in options_cleaned.iterrows():
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

    # Calibration de la surface complète (Rho, Eta, Gamma)
    rho_surf, eta_surf, gamma_surf = calibrate_full_surface_ssvi(df_vols)
    
    print(f"Paramètres Surface Calibrés :")
    print(f"  > Rho (Skew)    : {rho_surf:.4f}")
    print(f"  > Eta (Vol-of-Vol): {eta_surf:.4f}")
    print(f"  > Gamma (Power) : {gamma_surf:.4f}")

    # Reconstruction des paramètres par maturité pour le pricing
    ssvi_params_storage = {}
    for t in sorted(df_vols['T'].unique()):
        subset_t = df_vols[df_vols['T'] == t]
        # On définit theta_t comme la variance ATM à cette maturité
        theta_t = subset_t.iloc[abs(subset_t['k']).argmin()]['w']
        # Calcul de phi(theta) selon la loi de puissance de Gatheral
        phi_t = ssvi_phi_function(theta_t, eta_surf, gamma_surf)
        
        ssvi_params_storage[t] = {'theta': theta_t, 'rho': rho_surf, 'phi': phi_t}

    # ==========================================================
    # PHASE 4 : CALCUL DES GRECQUES ET EXPORT FINAL
    # ==========================================================
    print(f"\n--- Phase 4 : Calcul des Grecques (via Surface Lissée) ---")
    
    final_data = []
    for _, row in df_vols.iterrows():
        t = row['T']
        p = ssvi_params_storage[t]
        
        # 1. Volatilité lissée issue de la surface globale
        w_ssvi = ssvi_variance_total(row['k'], p['theta'], p['rho'], p['phi'])
        sigma_ssvi = np.sqrt(max(0, w_ssvi / t))
        
        # 2. Prix théorique cohérent avec la surface
        p_ssvi = get_ssvi_price(current_spot, row['strike'], t, row['r'], p['theta'], p['rho'], p['phi'], row['option_type'])
        
        # 3. Grecques analytiques
        g = calculate_greeks(current_spot, row['strike'], t, row['r'], sigma_ssvi, row['option_type'])
        
        final_data.append({
            'T': t, 'Strike': row['strike'], 'Type': row['option_type'],
            'Market_Price': row['market_price'], 'SSVI_Price': p_ssvi,
            'IV_Market': row['iv'], 'IV_SSVI': sigma_ssvi,
            'Delta': g['delta'], 'Gamma': g['gamma'], 'Vega': g['vega'], 'Theta': g['theta']
        })

    df_final = pd.DataFrame(final_data)
    mae_price = np.mean(np.abs(df_final['SSVI_Price'] - df_final['Market_Price']))
    
    print(f"Erreur moyenne de pricing sur la surface : {mae_price:.2f} USD")
    print("\nAperçu des 5 premières lignes du rapport final :")
    print(df_final[['T', 'Strike', 'Type', 'Delta', 'IV_SSVI', 'SSVI_Price']].head().to_string(index=False))

    # Exportation finale
    df_final.to_csv("surface_vol_complete_btc.csv", index=False)
    print(f"\nFichier 'surface_vol_complete_btc.csv' généré avec succès.")