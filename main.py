import pandas as pd
# On importe les fonctions depuis le script de traitement data_process.py

from src.data_process import get_deribit_data, process_and_filter_options
if __name__ == "__main__":
    CURRENCY = "BTC"
    
    # 1. Récupération brute
    options_raw, futures_raw = get_deribit_data(CURRENCY)
    
    # 2. Extraction du spot dynamique depuis les futures
    # On cherche le prix mid du perpétuel
    perp = futures_raw[futures_raw['instrument_name'].str.contains('PERPETUAL')]
    current_spot = perp['mid_price'].iloc[0] if not perp.empty else 65000
    
    # 3. Traitement et Filtrage (Phase 1 du projet)
    options_cleaned = process_and_filter_options(options_raw, current_spot)
    
    # 4. Affichage des résultats pour ton rapport technique
    print(f"--- Phase 1 : Pipeline Deribit {CURRENCY} ---")
    print(f"Prix Spot détecté : {current_spot}")
    print(f"Instruments totaux : {len(options_raw) + len(futures_raw)}")
    print(f"Options après filtres (Spread 25% + Arbitrage) : {len(options_cleaned)}")
    
    # Aperçu pour ton extraction des résultats (attendu dans le projet)
    print(options_cleaned[['instrument_name', 'strike', 'mid_price', 'spread']].head())

    # Sauvegarde optionnelle pour la Phase 2 (Nelson-Siegel)
    # options_cleaned.to_csv("data_cleaned.csv")