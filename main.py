import matplotlib.pyplot as plt
from src.data_process import get_deribit_data, process_and_filter_options
from src.rates_model import extract_implicit_rates, calibrate_nelson_siegel, nelson_siegel

if __name__ == "__main__":
    # --- PHASE 1 ---
    options_raw, futures_raw = get_deribit_data("BTC")
    # Simulation du spot (à dynamiser avec le futur perpétuel si possible)
    spot = futures_raw['mid_price'].iloc[0] 
    options_cleaned = process_and_filter_options(options_raw, spot)
    
    # --- PHASE 2 : COURBE DES TAUX ---
    print("Extraction des taux implicites...")
    raw_rates = extract_implicit_rates(options_cleaned, futures_raw)
    
    if not raw_rates.empty:
        print(f"Calibration Nelson-Siegel sur {len(raw_rates)} maturités...")
        b0, b1, b2, tau = calibrate_nelson_siegel(raw_rates)
        
        # Comparaison valeurs empiriques vs Nelson-Siegel [cite: 25]
        T_smooth = np.linspace(raw_rates['T'].min(), raw_rates['T'].max(), 100)
        r_smooth = nelson_siegel(T_smooth, b0, b1, b2, tau)
        
        plt.figure(figsize=(10, 5))
        plt.scatter(raw_rates['T'], raw_rates['r'], color='red', label='Taux Bruts (Marché)')
        plt.plot(T_smooth, r_smooth, label='Courbe Nelson-Siegel (Lissée)')
        plt.title("Structure par terme des taux implicites BTC")
        plt.xlabel("Maturité T (Années)")
        plt.ylabel("Taux d'intérêt r")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Erreur : Pas assez de données pour extraire les taux.")