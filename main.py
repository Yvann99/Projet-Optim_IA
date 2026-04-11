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

if __name__ == "__main__":
    CURRENCY = "BTC"
    
    print(f"--- Phase 1 : Récupération et Nettoyage ({CURRENCY}) ---")
    
    # 1. Collecte des données via l'API
    options_raw, futures_raw = get_deribit_data(CURRENCY)
    
    # 2. Identification du prix Spot (via le Future Perpétuel)
    perp = futures_raw[futures_raw['instrument_name'].str.contains('PERPETUAL')]
    current_spot = perp['mid_price'].iloc[0] if not perp.empty else futures_raw['mid_price'].iloc[0]
    print(f"Prix Spot détecté : {current_spot} USD")
    
    # 3. Filtrage (Spread 25%, Moneyness, Arbitrage Statistique)
    options_cleaned = process_and_filter_options(options_raw, current_spot)
    print(f"Options après filtrage : {len(options_cleaned)} / {len(options_raw)}")

    print(f"\n--- Phase 2 : Reconstruction de la courbe des taux ---")
    
    # 4. Extraction des taux bruts par parité Call-Put
    raw_rates = extract_implicit_rates(options_cleaned, futures_raw)
    
    if not raw_rates.empty:
        # 5. Calibration du modèle Nelson-Siegel
        print(f"Calibration Nelson-Siegel sur {len(raw_rates)} maturités...")
        params = calibrate_nelson_siegel(raw_rates)
        b0, b1, b2, tau = params
        
        # 6. Évaluation de la précision (Comparaison Empirique vs Modèle)
        mse, mae, comparison_df = evaluate_ns_performance(raw_rates, params)
        
        print(f"Mean Absolute Error (MAE) : {mae:.6f}")
        print("\nTableau comparatif des taux :")
        print(comparison_df[['T', 'r', 'r_ns', 'error']].to_string(index=False))
        
        # 7. Génération de la courbe lissée pour le graphique
        T_smooth = np.linspace(raw_rates['T'].min(), raw_rates['T'].max(), 200)
        r_smooth = nelson_siegel(T_smooth, b0, b1, b2, tau)
        
        # 8. Visualisation et Sauvegarde
        plt.figure(figsize=(10, 6))
        plt.scatter(raw_rates['T'], raw_rates['r'] * 100, color='red', label='Taux Marché (Empiriques)')
        plt.plot(T_smooth, r_smooth * 100, color='blue', label='Lissage Nelson-Siegel')
        
        plt.title(f"Structure par terme des taux {CURRENCY} - Modèle Nelson-Siegel")
        plt.xlabel("Maturité T (Années)")
        plt.ylabel("Taux d'intérêt (%)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_plot = 'courbe_taux_nelson_siegel.png'
        plt.savefig(output_plot)
        plt.close()
        
        print(f"\nGraphique sauvegardé : {output_plot}")
        # Commande pour ouvrir automatiquement l'image sur Mac
        os.system(f"open {output_plot}")
        
    else:
        print("Erreur : Impossible d'extraire suffisamment de points de taux pour la calibration.")