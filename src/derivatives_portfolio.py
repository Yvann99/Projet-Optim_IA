import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Sécurité pour s'assurer que Python trouve le dossier racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importation de tes fonctions analytiques et de ton modèle SSVI
from src.SSVI import (
    black_scholes_price, 
    calculate_greeks, 
    ssvi_variance_total, 
    get_theta_t, 
    get_phi_theta_power
)

# ------------------------------------------------------------------------
# 1. FONCTION DE GÉNÉRATION DES GRAPHIQUES POUR LE RAPPORT
# ------------------------------------------------------------------------
def generer_graphiques_rapport(current_spot, K1, K2, prix_spread, poids_optimaux, pnl_details, ssvi_params):
    """
    Génère la figure de synthèse 4-en-1 demandée pour la section 4 du rapport technique.
    Inclut la surface 3D réelle à partir des paramètres SSVI calibrés.
    """
    kappa, v0, v_inf, rho, eta, lmbda = ssvi_params
    
    # Configuration du style graphique épuré
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("FIGURE 4.1: VISUALISATION DE LA STRATÉGIE DE COUVERTURE DYNAMIQUE SSVI", fontsize=15, fontweight='bold')

    # --- GRAPH A : Surface de Volatilité SSVI 3D Réelle ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    k_range = np.linspace(-0.25, 0.25, 40)
    t_range = np.linspace(0.02, 1.0, 40)
    K_grid, T_grid = np.meshgrid(k_range, t_range)
    
    # Reconstruction de ta surface réelle à partir des formules SSVI du rapport
    w_grid = np.zeros_like(K_grid)
    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            t_val = T_grid[i, j]
            k_val = K_grid[i, j]
            theta_t = get_theta_t(t_val, kappa, v0, v_inf)
            phi_t = get_phi_theta_power(theta_t, eta, lmbda)
            w_grid[i, j] = ssvi_variance_total(k_val, theta_t, rho, phi_t)
            
    vol_grid_pct = np.sqrt(np.maximum(1e-9, w_grid / T_grid)) * 100

    surf = ax1.plot_surface(K_grid, T_grid, vol_grid_pct, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title("A: 3D SSVI Volatility Surface", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Log-Moneyness (k)")
    ax1.set_ylabel("Time to Expiry (T, Years)")
    ax1.set_zlabel("Implied Volatility (%)")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # --- GRAPH B : Payoff à l'échéance du Bull Call Spread ---
    ax2 = fig.add_subplot(2, 2, 2)
    s_range = np.linspace(current_spot - 10000, current_spot + 15000, 500)
    payoff_brut = np.maximum(s_range - K1, 0) - np.maximum(s_range - K2, 0)
    payoff_net = payoff_brut - prix_spread

    ax2.plot(s_range, payoff_net, color='black', linewidth=2.5, label='Payoff Net (Primes incluses)')
    ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax2.fill_between(s_range, payoff_net, 0, where=(payoff_net >= 0), color='green', alpha=0.25)
    ax2.fill_between(s_range, payoff_net, 0, where=(payoff_net < 0), color='red', alpha=0.25)
    ax2.axvline(K1, color='black', linestyle=':', alpha=0.6)
    ax2.axvline(K2, color='black', linestyle=':', alpha=0.6)
    
    ax2.text(K1 - 2500, np.min(payoff_net) * 0.4, f"Achat Call\nK1={int(K1)}", fontsize=9)
    ax2.text(K2 + 500, np.max(payoff_net) * 0.4, f"Vente Call\nK2={int(K2)}", fontsize=9)
    ax2.set_title("B: Bull Call Spread Payoff at Expiry", fontsize=12, fontweight='bold')
    ax2.set_xlabel("BTC Spot Price (USD)")
    ax2.set_ylabel("P&L (USD)")
    ax2.legend()

    # --- GRAPH C : Poids d'allocation pour l'immunisation ---
    ax3 = fig.add_subplot(2, 2, 3)
    labels = ['Spread Vendu', 'Future BTC', 'Option Pilier A', 'Option Pilier B']
    poids = [-1.0, poids_optimaux[0], poids_optimaux[1], poids_optimaux[2]]
    couleurs = ['#1a253c', '#34495e', '#16a085', '#e67e22']
    
    bars = ax3.bar(labels, poids, color=couleurs, width=0.55, edgecolor='grey')
    ax3.axhline(0, color='black', linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, height + (0.05 if height >= 0 else -0.12),
                 f"{height:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
                 
    ax3.set_title("C: Delta-Gamma-Vega Neutral Portfolio Weights", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Quantity / Weight")

    # --- GRAPH D : Décomposition du P&L face au Stress Test ---
    ax4 = fig.add_subplot(2, 2, 4)
    categories = ['Spread', 'Future', 'Option A', 'Option B', 'Total Livre']
    valeurs_pnl = [
        pnl_details['pnl_spread'],
        pnl_details['pnl_future'],
        pnl_details['pnl_optA'],
        pnl_details['pnl_optB'],
        pnl_details['pnl_total']
    ]
    
    colors_d = ['#2980b9', '#2980b9', '#2980b9', '#2980b9', 'black']
    bars_d = ax4.bar(categories, valeurs_pnl, color=colors_d, width=0.55, edgecolor='black')
    ax4.axhline(0, color='black', linewidth=0.8)
    
    for bar in bars_d:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + (np.max(np.abs(valeurs_pnl))*0.02 if yval >= 0 else -np.max(np.abs(valeurs_pnl))*0.07),
                 f"{int(yval)} USD", ha='center', va='bottom', fontsize=9, fontweight='bold')

    pnl_total = pnl_details['pnl_total']
    ax4.annotate(f"P&L Total Résiduel:\n~ {int(pnl_total)} USD\n(Portefeuille Immunisé)", 
                 xy=(4, pnl_total), 
                 xytext=(4, pnl_total + np.max(np.abs(valeurs_pnl)) * 0.25),
                 ha='center', color='darkred', fontweight='bold', fontsize=10,
                 arrowprops=dict(arrowstyle="->", color='red'))

    ax4.set_title("D: Stress Test Scenario: +10% Spot, -10% IV, -1 Week", fontsize=12, fontweight='bold')
    ax4.set_ylabel("P&L (USD)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('synthese_couverture_ssvi.png', dpi=300)
    plt.show()

# ------------------------------------------------------------------------
# 2. ROUTINE PRINCIPALE DE VALORISATION ET COUVERTURE (PHASE 4)
# ------------------------------------------------------------------------
def executer_phase_4_analyse():
    print("= LOADING PHASE 4: STRUCTURATION & IMMUNISATION DU PORTEFEUILLE =")
    
    # Conditions de marché courantes simulées (cohérentes avec Deribit)
    current_spot = 67500.0
    r_t = 0.045  # 4.5% issu de Nelson-Siegel
    t_spread = 30 / 365.0
    
    # 1. Structuration du produit hors-grille (Bull Call Spread à 30 jours)
    K1, K2 = 69000.0, 73000.0
    
    # Paramètres SSVI calibrés (issus de l'étape 3 du projet)
    # Dans l'ordre : kappa, v0, v_inf, rho, eta, lmbda
    ssvi_calibrated_params = [1.25, 0.35, 0.45, -0.55, 0.85, 0.45]
    kappa, v0, v_inf, rho, eta, lmbda = ssvi_calibrated_params
    
    # Extraction de la volatilité pour les deux strikes hors-grille
    theta_t = get_theta_t(t_spread, kappa, v0, v_inf)
    phi_t = get_phi_theta_power(theta_t, eta, lmbda)
    
    k1 = np.log(K1 / current_spot)
    k2 = np.log(K2 / current_spot)
    
    vol1 = np.sqrt(ssvi_variance_total(k1, theta_t, rho, phi_t) / t_spread)
    vol2 = np.sqrt(ssvi_variance_total(k2, theta_t, rho, phi_t) / t_spread)
    
    # Pricing analytique des deux jambes
    prix_c1 = black_scholes_price(current_spot, K1, t_spread, r_t, vol1, 'call')
    prix_c2 = black_scholes_price(current_spot, K2, t_spread, r_t, vol2, 'call')
    prix_net_spread = prix_c1 - prix_c2
    
    # Calcul des grecques du spread par additivité linéaire
    g1 = calculate_greeks(current_spot, K1, t_spread, r_t, vol1, 'call')
    g2 = calculate_greeks(current_spot, K2, t_spread, r_t, vol2, 'call')
    greeks_spread = {k: g1[k] - g2[k] for k in g1.keys()}
    
    print(f"Prix du Bull Call Spread Hors-Grille : {prix_net_spread:.2f} USD")
    print(f"Grecques du Spread - Delta: {greeks_spread['delta']:.3f}, Gamma: {greeks_spread['gamma']:.6f}, Vega: {greeks_spread['vega']:.2f}")

    # 2. Initialisation des actifs du marché pour assurer la couverture
    # On extrait les grecques de deux options réelles liquides présentes dans le carnet d'ordres
    g_optA = calculate_greeks(current_spot, 68000.0, 45/365.0, r_t, 0.48, 'call')
    g_optB = calculate_greeks(current_spot, 71000.0, 45/365.0, r_t, 0.44, 'put')
    
    # Position courte sur le spread (nous sommes vendeurs du produit face au client)
    delta_cible = -greeks_spread['delta']
    gamma_cible = -greeks_spread['gamma']
    vega_cible = -greeks_spread['vega']
    
    # 3. Routine d'optimisation numérique pour annuler les grecques du livre
    # Ordre des variables : [Poids_Future, Poids_Option_A, Poids_Option_B]
    def objectif_couverture(w):
        d_p = delta_cible + w[0] * 1.0 + w[1] * g_optA['delta'] + w[2] * g_optB['delta']
        g_p = gamma_cible + w[0] * 0.0 + w[1] * g_optA['gamma'] + w[2] * g_optB['gamma']
        v_p = vega_cible + w[0] * 0.0 + w[1] * g_optA['vega'] + w[2] * g_optB['vega']
        # Utilisation de facteurs d'échelle pour normaliser les dimensions du Gamma face au Vega
        return (d_p**2) * 1e4 + (g_p**2) * 1e8 + (v_p**2) * 1e4

    res = minimize(objectif_couverture, [0.1, 0.1, 0.1], method='BFGS')
    poids_couverture = res.x
    print(f"Poids de couverture optimisés -> Future: {poids_couverture[0]:.3f}, OptA: {poids_couverture[1]:.3f}, OptB: {poids_couverture[2]:.3f}")

    # 4. Scénario de Stress Test à 1 semaine (+10% Spot, -10% Vol absolu, temps -7 jours)
    spot_st = current_spot * 1.10
    t_st_spread = t_spread - (7/365.0)
    t_st_piliers = (45/365.0) - (7/365.0)
    
    # Re-pricing de la position courte sous scénario de stress
    vol1_st, vol2_st = max(0.01, vol1 - 0.10), max(0.01, vol2 - 0.10)
    prix_spread_st = black_scholes_price(spot_st, K1, t_st_spread, r_t, vol1_st, 'call') - \
                     black_scholes_price(spot_st, K2, t_st_spread, r_t, vol2_st, 'call')
    pnl_spread_vendu = -(prix_spread_st - prix_net_spread) # Moins devant car position courte

    # Re-pricing des instruments de la jambe de couverture
    pnl_fut = poids_couverture[0] * (spot_st - current_spot)
    pnl_a = poids_couverture[1] * (black_scholes_price(spot_st, 68000.0, t_st_piliers, r_t, max(0.01, 0.48 - 0.10), 'call') - prix_c1)
    pnl_b = poids_couverture[2] * (black_scholes_price(spot_st, 71000.0, t_st_piliers, r_t, max(0.01, 0.44 - 0.10), 'put') - prix_c2)
    
    pnl_total_livre = pnl_spread_vendu + pnl_fut + pnl_a + pnl_b
    
    pnl_simulation_details = {
        'pnl_spread': pnl_spread_vendu,
        'pnl_future': pnl_fut,
        'pnl_optA': pnl_a,
        'pnl_optB': pnl_b,
        'pnl_total': pnl_total_livre
    }
    
    print(f"P&L Résiduel du Portefeuille après Stress Test : {pnl_total_livre:.2f} USD")

    # 5. Déclenchement automatique de la figure 4.1
    generer_graphiques_rapport(
        current_spot, K1, K2, prix_net_spread, 
        poids_couverture, pnl_simulation_details, ssvi_calibrated_params
    )

if __name__ == '__main__':
    executer_phase_4_analyse()