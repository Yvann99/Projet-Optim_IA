import numpy as np
import pandas as pd  # <--- Il manquait celui-là !
from scipy.stats import norm
from scipy.optimize import minimize


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calcul du prix théorique via Black-Scholes."""
    if sigma < 1e-7 or T < 1e-7:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_vega(S, K, T, r, sigma):
    """Calcule le Vega : dérivée du prix par rapport à la volatilité."""
    if sigma < 1e-7 or T < 1e-7:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def get_implied_vol(price, S, K, T, r, option_type='call'):
    """
    Extrait l'IV en utilisant Newton-Raphson avec repli sur dichotomie.
    """
    intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
    if price <= intrinsic:
        return 0.0

    #Tentative avec Newton-Raphson
    sigma_n = 0.5  # Estimation initiale (50%)
    for i in range(20):
        p = black_scholes_price(S, K, T, r, sigma_n, option_type)
        v = black_scholes_vega(S, K, T, r, sigma_n)
        
        diff = p - price
        if abs(diff) < 1e-8:
            return sigma_n
        
        # Si le Vega est trop faible, Newton-Raphson va échouer -> on passe à la dichotomie
        if abs(v) < 1e-7:
            break
            
        sigma_n = sigma_n - diff / v
        
        # Si Newton projette une valeur aberrante, on arrête
        if sigma_n <= 0 or sigma_n > 5.0:
            break
    
    # --- Repli sur la Dichotomie (Bisection) ---
    low, high = 0.0001, 5.0
    for i in range(100):
        mid = (low + high) / 2
        p_mid = black_scholes_price(S, K, T, r, mid, option_type)
        if abs(p_mid - price) < 1e-7:
            return mid
        if p_mid < price:
            low = mid
        else:
            high = mid
    return (low + high) / 2

#MODÈLE SSVI

def ssvi_variance_total(k, theta, rho, phi):
    """Formule de la variance totale SSVI."""
    return (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt((phi * k + rho)**2 + (1.0 - rho**2)))

def calibrate_ssvi_slice(k_arr, w_market_arr, theta_atm):
    """Calcule rho et phi pour une maturité donnée."""
    def objective_ssvi(params):
        rho, phi = params
        if abs(rho) >= 0.99 or phi <= 1e-6:
            return 1e12
        w_pred = ssvi_variance_total(k_arr, theta_atm, rho, phi)
        return np.sum((w_market_arr - w_pred)**2)

    res = minimize(objective_ssvi, [0.0, 0.5], bounds=[(-0.95, 0.95), (1e-4, 10)])
    return res.x
def get_ssvi_price(S, K, T, r, theta_atm, rho, phi, option_type='call'):
    """
    Calcule le prix d'une option en utilisant la variance du modèle SSVI.
    """
    # 1. Calcul du log-strike forward
    forward = S * np.exp(r * T)
    k = np.log(K / forward)
    
    # 2. Calcul de la variance totale SSVI
    w_pred = ssvi_variance_total(k, theta_atm, rho, phi)
    
    # 3. Conversion variance totale -> volatilité sigma
    # w = sigma^2 * T => sigma = sqrt(w / T)
    sigma_ssvi = np.sqrt(max(0, w_pred / T))
    
    # 4. Calcul du prix via Black-Scholes
    return black_scholes_price(S, K, T, r, sigma_ssvi, option_type)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calcule les principales grecques via le modèle Black-Scholes.
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # PDF et CDF de la loi normale
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    # 1. DELTA
    if option_type == 'call':
        delta = cdf_d1
    else:
        delta = cdf_d1 - 1

    # 2. GAMMA (Identique Call/Put)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))

    # 3. VEGA (Identique Call/Put)
    # Divisé par 100 pour avoir l'impact d'une variation de 1% de volatilité
    vega = (S * np.sqrt(T) * pdf_d1) / 100

    # 4. THETA
    # Divisé par 365 pour avoir l'impact d'un jour qui passe
    term1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        term2 = r * K * np.exp(-r * T) * cdf_d2
        theta = (term1 - term2) / 365
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }
def ssvi_phi_function(theta, eta, gamma):
    """Fonction phi de Gatheral pour le lissage temporel."""
    return eta / (pow(theta, gamma) * pow(1 + theta, 1 - gamma))

def calibrate_full_surface_ssvi(df_vols):
    """
    Calibre les paramètres η (eta), γ (gamma) et ρ (rho) sur l'ensemble 
    des maturités pour lisser la surface complète.
    """
    # On prépare les données : k (log-strike), theta (variance ATM), w (variance marché)
    maturities = sorted(df_vols['T'].unique())
    data_points = []
    
    for t in maturities:
        subset = df_vols[df_vols['T'] == t]
        theta_t = subset.iloc[abs(subset['k']).argmin()]['w']
        for _, row in subset.iterrows():
            data_points.append({
                'k': row['k'], 'w_mkt': row['w'], 'theta': theta_t
            })
    
    df_fit = pd.DataFrame(data_points)

    def objective_surface(params):
        rho, eta, gamma = params
        # Contraintes de base
        if abs(rho) >= 0.99 or eta <= 0 or not (0 <= gamma <= 0.5):
            return 1e12
        
        # Calcul de phi pour chaque point en fonction de son theta
        phi_vals = ssvi_phi_function(df_fit['theta'], eta, gamma)
        
        # Calcul de la variance prédite par SSVI
        w_pred = (df_fit['theta'] / 2) * (
            1 + rho * phi_vals * df_fit['k'] + 
            np.sqrt((phi_vals * df_fit['k'] + rho)**2 + (1 - rho**2))
        )
        
        return np.sum((df_fit['w_mkt'] - w_pred)**2)

    # Initialisation standard : rho négatif, eta autour de 1, gamma à 0.25
    res = minimize(objective_surface, [-0.4, 1.0, 0.3], 
                   bounds=[(-0.95, 0.95), (0.01, 5), (0.01, 0.5)])
    return res.x # [rho, eta, gamma]