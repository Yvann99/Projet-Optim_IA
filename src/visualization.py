import matplotlib.pyplot as plt
import numpy as np

def plot_style_comparison(df_final):
    """
    Compare les points de marché (dots) avec les courbes SSVI (lignes) 
    pour les deux maturités les plus représentatives.
    """
    maturities = sorted(df_final['T'].unique())
    # On choisit une maturité courte et une longue
    selected_T = [maturities[0], maturities[len(maturities)//2]]
    
    plt.figure(figsize=(12, 6))
    
    for i, t in enumerate(selected_T):
        subset = df_final[df_final['T'] == t].sort_values('Strike')
        
        plt.subplot(1, 2, i+1)
        # Points réels
        plt.scatter(subset['Strike'], subset['IV_Market'], color='red', alpha=0.5, label='Marché (Réel)')
        # Courbe SSVI
        plt.plot(subset['Strike'], subset['IV_SSVI'], color='blue', lw=2, label='Modèle SSVI')
        
        plt.title(f"Smile de Volatilité - T = {t*365:.1f} jours")
        plt.xlabel("Strike")
        plt.ylabel("Volatilité Implicite")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()