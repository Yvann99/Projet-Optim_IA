import numpy as np
import pandas as pd
import os
import matplotlib
# Configuration du backend pour éviter les problèmes d'affichage
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

from src.data_process import get_deribit_data, process_and_filter_options
from src.rates_model import (
    extract_implicit_rates, 
    calibrate_nelson_siegel, 
    nelson_siegel, 
    evaluate_ns_performance
)
from src.SSVI import (
    get_implied_vol, 
    calibrate_ssvi_structure_2steps,
    get_theta_t,
    get_phi_theta_power,
    ssvi_variance_total,
    get_ssvi_price,
    calculate_greeks
)

def plot_style_comparison(df_final):
    """
    Compare les points de marché (dots) avec les courbes SSVI (lignes) 
    pour les deux maturités les plus représentatives.
    """
    maturities = sorted(df_final['T'].unique())
    if len(maturities) < 1:
        print("Aucune donnée à afficher pour le graphique.")
        return

    # Sélection de la maturité la plus courte et d'une maturité intermédiaire
    selected_T = [maturities[0], maturities[min(len(maturities)-1, len(maturities)//2)]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, t in enumerate(selected_T):
        subset = df_final[df_final['T'] == t].sort_values('Strike')
        
        # Points réels du marché
        axes[i].scatter(subset['Strike'], subset['IV_Market'], color='red', alpha=0.5, label='Marché (Réel)')
        # Courbe lissée par le modèle
        axes[i].plot(subset['Strike'], subset['IV_SSVI'], color='blue', lw=2, label='Modèle SSVI')
        
        axes[i].set_title(f"Smile de Volatilité - T = {t*365:.1f} jours")
        axes[i].set_xlabel("Strike")
        axes[i].set_ylabel("Volatilité Implicite")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("\n[INFO] Ouverture de la fenêtre graphique... Fermez-la pour terminer le script.")
    plt.show()

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
    # PHASE 3 : CALIBRATION STRUCTURELLE SSVI
    # ==========================================================
    print(f"\n--- Phase 3 : Calibration Structurelle (Gatheral & Jacquier) ---")
    
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

    # Calibration en deux étapes
    p = calibrate_ssvi_structure_2steps(df_vols, current_spot)
    
    print(f"Paramètres ATM (Temps)  : Kappa={p['kappa']:.4f}, V0={p['v0']:.4f}, V_inf={p['v_inf']:.4f}")
    print(f"Paramètres Smile (Forme): Rho={p['rho']:.4f}, Eta={p['eta']:.4f}, Lambda={p['lmbda']:.4f}")

    # Reconstruction de la surface
    ssvi_params_storage = {}
    for t in sorted(df_vols['T'].unique()):
        theta_t = get_theta_t(t, p['kappa'], p['v0'], p['v_inf'])
        phi_t = get_phi_theta_power(theta_t, p['eta'], p['lmbda'])
        ssvi_params_storage[t] = {'theta': theta_t, 'rho': p['rho'], 'phi': phi_t}

    # ==========================================================
    # PHASE 4 : CALCUL DES RÉSULTATS ET EXPORT
    # ==========================================================
    print(f"\n--- Phase 4 : Calcul des Grecques et Comparaison ---")
    
    final_data = []
    for _, row in df_vols.iterrows():
        t = row['T']
        param_t = ssvi_params_storage[t]
        
        w_ssvi = ssvi_variance_total(row['k'], param_t['theta'], param_t['rho'], param_t['phi'])
        sigma_ssvi = np.sqrt(max(0, w_ssvi / t))
        p_ssvi = get_ssvi_price(current_spot, row['strike'], t, row['r'], 
                                param_t['theta'], param_t['rho'], param_t['phi'], row['option_type'])
        g = calculate_greeks(current_spot, row['strike'], t, row['r'], sigma_ssvi, row['option_type'])
        
        final_data.append({
            'T': t, 'Strike': row['strike'], 'Type': row['option_type'],
            'Market_Price': row['market_price'], 'SSVI_Price': p_ssvi,
            'IV_Market': row['iv'], 'IV_SSVI': sigma_ssvi,
            'Delta': g['delta'], 'Gamma': g['gamma'], 'Vega': g['vega'], 'Theta': g['theta']
        })

    df_final = pd.DataFrame(final_data)
    
    # --- AFFICHAGE DES RÉSULTATS DANS LE TERMINAL ---
    mae_price = np.mean(np.abs(df_final['SSVI_Price'] - df_final['Market_Price']))
    print(f"Erreur moyenne de pricing (MAE) : {mae_price:.2f} USD")
    print("\nAperçu des résultats (5 premières lignes) :")
    print(df_final[['T', 'Strike', 'Type', 'Delta', 'IV_SSVI', 'SSVI_Price']].head().to_string(index=False))

    # --- EXPORTATION ---
    df_final.to_csv("surface_vol_structurelle_btc.csv", index=False)
    print(f"\nFichier 'surface_vol_structurelle_btc.csv' généré avec succès.")

    # --- VISUALISATION (Dernière étape) ---
    plot_style_comparison(df_final)