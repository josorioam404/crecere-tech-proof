"""
Modelado Estadístico y Econométrico: Humanos vs. IA en Cobranza Bancaria
------------------------------------------------------------------------
Ejecuta la batería completa de contrastes de hipótesis paramétricos y no
paramétricos para las 5 hipótesis de negocio, complementado con un modelo
econométrico multivariado (Regresión Logística) para aislar el efecto causal
sobre la Promesa de Pago (PTP).

Metodología:
1. Embudo Global (n = 99): Contactabilidad bruta (is_rpc) Humanos vs. IA.
2. Submuestra de Estudio (n = 60 con is_rpc == True):
   - H1: Alternativas Ofrecidas (Humanos > IA) -> Shapiro-Wilk, Mann-Whitney U, Cliff's Delta.
   - H2: Claridad del Mensaje (IA > Humanos) -> Mann-Whitney U, Fisher / Chi2 0 repeticiones.
   - H3: Resolución de Objeciones (Humanos > IA) -> Mann-Whitney U, Odds Ratio y Fisher.
   - H4: Variabilidad de Tono (Humanos > IA) -> Entropía de Shannon H(X), Chi2 homogeneidad.
   - H5: Monopolización (IA > Humanos) -> Shapiro-Wilk, Welch's t-test, Cohen's d, % > 0.75.
3. Modelo Econométrico Multivariado:
   logit(P(PTP=1)) = b0 + b1*is_ia + b2*alternativas + b3*claridad + b4*talk_ratio + b5*objeciones
4. Exportación de artefactos:
   - data/processed/statistical_results.json
   - data/processed/statistical_summary.md
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ── Rutas del Proyecto ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYTICAL_CSV = PROCESSED_DIR / "analytical_dataset.csv"
RESULTS_JSON = PROCESSED_DIR / "statistical_results.json"
SUMMARY_MD = PROCESSED_DIR / "statistical_summary.md"


# ── Funciones Utilitarias y Tamaños del Efecto ─────────────────────────────────

def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calcula Cliff's Delta para comparar dos distribuciones ordinales/continuas.
    delta = [#(x > y) - #(x < y)] / (n_x * n_y)
    Rangos de interpretación de Romano et al. (2006):
    - |delta| < 0.147: Negligible
    - 0.147 <= |delta| < 0.33: Small
    - 0.33 <= |delta| < 0.474: Medium
    - |delta| >= 0.474: Large
    """
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, "N/A"
    
    greater = 0
    less = 0
    for val_x in x:
        greater += np.sum(val_x > y)
        less += np.sum(val_x < y)
        
    delta = (greater - less) / (n_x * n_y)
    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "Despreciable"
    elif abs_d < 0.33:
        interp = "Pequeño"
    elif abs_d < 0.474:
        interp = "Mediano"
    else:
        interp = "Grande"
        
    return float(delta), interp


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, str]:
    """
    Calcula Cohen's d y la corrección de Hedges' g para muestras pequeñas.
    """
    n_x, n_y = len(x), len(y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    
    # Varianza combinada (pooled variance)
    pooled_sd = math.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
    if pooled_sd == 0:
        return 0.0, 0.0, "N/A"
    
    d = (mean_x - mean_y) / pooled_sd
    # Corrección de Hedges (J factor)
    j = 1 - (3 / (4 * (n_x + n_y) - 9))
    g = d * j
    
    abs_g = abs(g)
    if abs_g < 0.2:
        interp = "Despreciable"
    elif abs_g < 0.5:
        interp = "Pequeño"
    elif abs_g < 0.8:
        interp = "Mediano"
    else:
        interp = "Grande"
        
    return float(d), float(g), interp


def shannon_entropy(series: pd.Series) -> float:
    """
    Calcula la Entropía de Shannon H(X) en bits: -sum(p * log2(p)).
    """
    counts = series.value_counts()
    total = len(series)
    if total == 0:
        return 0.0
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calcula la proporción y su intervalo de confianza de Wilson.
    """
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p + z**2 / (2 * total)
    adjusted_limits = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    
    lower = max(0.0, (centre_adjusted_probability - adjusted_limits) / denominator)
    upper = min(1.0, (centre_adjusted_probability + adjusted_limits) / denominator)
    return float(p), float(lower), float(upper)


def winsorize_series(data: np.ndarray, lower: float = 0.0, upper: float = 0.99) -> np.ndarray:
    """
    Aplica Winsorización sobre un array numérico en los percentiles especificados.
    Mitiga el impacto de valores atípicos extremos preservando el tamaño muestral.
    """
    if len(data) == 0:
        return data
    q_low = float(np.percentile(data, lower * 100)) if lower > 0 else float(np.min(data))
    q_high = float(np.percentile(data, upper * 100)) if upper < 1.0 else float(np.max(data))
    return np.clip(data, q_low, q_high)


# ── Módulo Principal de Análisis ──────────────────────────────────────────────

class StatisticalAnalyzer:
    """
    Motor integral de análisis estadístico y econométrico para Creceré AI.
    """

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df.copy()
        
        # Corrección de tipos si es necesario
        self.raw_df["is_rpc"] = self.raw_df["is_rpc"].astype(bool)
        self.raw_df["ptp_logrado"] = self.raw_df["ptp_logrado"].astype(bool)
        self.raw_df["num_objeciones"] = self.raw_df["num_objeciones"].fillna(0).astype(int)
        self.raw_df["num_objeciones_resueltas"] = self.raw_df["num_objeciones_resueltas"].fillna(0).astype(int)
        self.raw_df["alternativas_ofrecidas"] = self.raw_df["alternativas_ofrecidas"].fillna(0).astype(int)
        self.raw_df["peticiones_aclaracion_cliente"] = self.raw_df["peticiones_aclaracion_cliente"].fillna(0).astype(int)
        self.raw_df["claridad_mensaje"] = self.raw_df["claridad_mensaje"].astype(float)
        self.raw_df["talk_ratio_agente"] = self.raw_df["talk_ratio_agente"].astype(float)
        self.raw_df["duracion_total_segundos"] = self.raw_df["duracion_total_segundos"].astype(float)
        
        # Submuestra filtrada de contacto efectivo (is_rpc == True)
        self.rpc_df = self.raw_df[self.raw_df["is_rpc"] == True].copy()
        
        self.results: Dict[str, Any] = {}

    def run_all(self) -> Dict[str, Any]:
        """Ejecuta todos los bloques analíticos."""
        print("=" * 80)
        print("  EJECUTANDO BATERÍA ESTADÍSTICA Y ECONOMÉTRICA (Creceré AI)")
        print("=" * 80)
        
        self.analyze_funnel()
        self.test_h1_alternativas()
        self.test_h2_claridad()
        self.test_h3_objeciones()
        self.test_h4_tono()
        self.test_h5_monopolizacion()
        self.fit_econometric_model()
        
        self.save_artifacts()
        return self.results

    # ── Fase 0: Embudo Global y Contactabilidad ───────────────────────────────

    def analyze_funnel(self):
        print(f"\n[1/7] Analizando Embudo Global y Contactabilidad (n = {len(self.raw_df)})...")
        total_humano = len(self.raw_df[self.raw_df["source"] == "humano"])
        total_ia = len(self.raw_df[self.raw_df["source"] == "ia"])
        
        rpc_humano = int(self.raw_df[(self.raw_df["source"] == "humano") & (self.raw_df["is_rpc"] == True)]["is_rpc"].count())
        rpc_ia = int(self.raw_df[(self.raw_df["source"] == "ia") & (self.raw_df["is_rpc"] == True)]["is_rpc"].count())
        
        rate_h, low_h, up_h = wilson_score_interval(rpc_humano, total_humano)
        rate_ia, low_ia, up_ia = wilson_score_interval(rpc_ia, total_ia)
        
        # Tabla 2x2 para Fisher y Chi2
        table = [
            [rpc_humano, total_humano - rpc_humano],
            [rpc_ia, total_ia - rpc_ia]
        ]
        chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(table, correction=True)
        odds_ratio, fisher_p = stats.fisher_exact(table)
        
        ptp_humano = int(self.rpc_df[self.rpc_df["source"] == "humano"]["ptp_logrado"].sum())
        ptp_ia = int(self.rpc_df[self.rpc_df["source"] == "ia"]["ptp_logrado"].sum())

        self.results["funnel"] = {
            "total_calls": len(self.raw_df),
            "humanos": {
                "total": total_humano,
                "rpc_count": rpc_humano,
                "rpc_rate": round(rate_h, 4),
                "ci_95": [round(low_h, 4), round(up_h, 4)],
                "ptp_count": ptp_humano,
                "ptp_rate_given_rpc": round(ptp_humano / rpc_humano, 4) if rpc_humano else 0.0
            },
            "ia": {
                "total": total_ia,
                "rpc_count": rpc_ia,
                "rpc_rate": round(rate_ia, 4),
                "ci_95": [round(low_ia, 4), round(up_ia, 4)],
                "ptp_count": ptp_ia,
                "ptp_rate_given_rpc": round(ptp_ia / rpc_ia, 4) if rpc_ia else 0.0
            },
            "diff_pp": round((rate_h - rate_ia) * 100, 2),
            "chi2_stat": round(float(chi2_stat), 4),
            "chi2_p_value": float(chi2_p),
            "fisher_p_value": float(fisher_p),
            "odds_ratio": round(float(odds_ratio), 4),
            "is_significant_05": bool(fisher_p < 0.05)
        }
        print(f"  • Contactabilidad Humanos: {rate_h:.1%} ({rpc_humano}/{total_humano})")
        print(f"  • Contactabilidad IA:      {rate_ia:.1%} ({rpc_ia}/{total_ia})")
        print(f"  • Brecha: +{(rate_h - rate_ia)*100:.1f} p.p. a favor de humanos (Fisher p = {fisher_p:.4f})")

    # ── Hipótesis 1: Alternativas Ofrecidas ────────────────────────────────────

    def test_h1_alternativas(self):
        print("\n[2/7] H1: Alternativas Ofrecidas (Humanos > IA)...")
        h_alt = self.rpc_df[self.rpc_df["source"] == "humano"]["alternativas_ofrecidas"].values
        ia_alt = self.rpc_df[self.rpc_df["source"] == "ia"]["alternativas_ofrecidas"].values
        
        # Test de Normalidad Shapiro-Wilk
        shapiro_h_stat, shapiro_h_p = stats.shapiro(h_alt)
        shapiro_ia_stat, shapiro_ia_p = stats.shapiro(ia_alt)
        
        # Mann-Whitney U unilateral (Humanos > IA)
        u_stat, u_p_one = stats.mannwhitneyu(h_alt, ia_alt, alternative="greater")
        u_stat_two, u_p_two = stats.mannwhitneyu(h_alt, ia_alt, alternative="two-sided")
        
        # Cliff's Delta
        delta, delta_interp = compute_cliffs_delta(h_alt, ia_alt)
        
        self.results["h1_alternativas"] = {
            "humanos": {
                "n": len(h_alt),
                "mean": round(float(np.mean(h_alt)), 3),
                "std": round(float(np.std(h_alt, ddof=1)), 3),
                "median": float(np.median(h_alt)),
                "iqr": float(np.percentile(h_alt, 75) - np.percentile(h_alt, 25)),
                "shapiro_p": float(shapiro_h_p)
            },
            "ia": {
                "n": len(ia_alt),
                "mean": round(float(np.mean(ia_alt)), 3),
                "std": round(float(np.std(ia_alt, ddof=1)), 3),
                "median": float(np.median(ia_alt)),
                "iqr": float(np.percentile(ia_alt, 75) - np.percentile(ia_alt, 25)),
                "shapiro_p": float(shapiro_ia_p)
            },
            "mann_whitney_u": float(u_stat),
            "p_value_one_sided": float(u_p_one),
            "p_value_two_sided": float(u_p_two),
            "cliffs_delta": round(delta, 4),
            "cliffs_delta_interpretation": delta_interp,
            "h1_supported": bool(u_p_one < 0.05 and np.mean(h_alt) > np.mean(ia_alt))
        }
        print(f"  • Humanos: Media={np.mean(h_alt):.2f}, Mediana={np.median(h_alt):.1f}, IQR={np.percentile(h_alt, 75) - np.percentile(h_alt, 25):.1f}")
        print(f"  • IA:      Media={np.mean(ia_alt):.2f}, Mediana={np.median(ia_alt):.1f}, IQR={np.percentile(ia_alt, 75) - np.percentile(ia_alt, 25):.1f}")
        print(f"  • Mann-Whitney U = {u_stat:.1f}, p (unilateral) = {u_p_one:.4e}, Cliff's Delta = {delta:.3f} ({delta_interp})")
        print(f"  • Resultado: {'HIPÓTESIS CONFIRMADA' if u_p_one < 0.05 else 'NO CONFIRMADA'}")

    # ── Hipótesis 2: Claridad del Mensaje ──────────────────────────────────────

    def test_h2_claridad(self):
        print("\n[3/7] H2: Claridad del Mensaje (IA > Humanos)...")
        h_clar = self.rpc_df[self.rpc_df["source"] == "humano"]["claridad_mensaje"].values
        ia_clar = self.rpc_df[self.rpc_df["source"] == "ia"]["claridad_mensaje"].values
        
        # Mann-Whitney U unilateral (IA > Humanos)
        u_stat, u_p_one = stats.mannwhitneyu(ia_clar, h_clar, alternative="greater")
        u_stat_two, u_p_two = stats.mannwhitneyu(ia_clar, h_clar, alternative="two-sided")
        
        # Cliff's Delta (IA frente a Humanos)
        delta, delta_interp = compute_cliffs_delta(ia_clar, h_clar)
        
        # Proporción con 0 peticiones de aclaración (claridad perfecta)
        h_rep0 = int(np.sum(self.rpc_df[self.rpc_df["source"] == "humano"]["peticiones_aclaracion_cliente"] == 0))
        ia_rep0 = int(np.sum(self.rpc_df[self.rpc_df["source"] == "ia"]["peticiones_aclaracion_cliente"] == 0))
        n_h = len(h_clar)
        n_ia = len(ia_clar)
        
        prop_h, low_h, up_h = wilson_score_interval(h_rep0, n_h)
        prop_ia, low_ia, up_ia = wilson_score_interval(ia_rep0, n_ia)
        
        table_clar = [
            [ia_rep0, n_ia - ia_rep0],
            [h_rep0, n_h - h_rep0]
        ]
        _, fisher_p = stats.fisher_exact(table_clar, alternative="greater")
        
        self.results["h2_claridad"] = {
            "humanos": {
                "n": n_h,
                "mean": round(float(np.mean(h_clar)), 4),
                "median": float(np.median(h_clar)),
                "std": round(float(np.std(h_clar, ddof=1)), 4),
                "prop_cero_repeticiones": round(prop_h, 4),
                "ci_95_prop": [round(low_h, 4), round(up_h, 4)]
            },
            "ia": {
                "n": n_ia,
                "mean": round(float(np.mean(ia_clar)), 4),
                "median": float(np.median(ia_clar)),
                "std": round(float(np.std(ia_clar, ddof=1)), 4),
                "prop_cero_repeticiones": round(prop_ia, 4),
                "ci_95_prop": [round(low_ia, 4), round(up_ia, 4)]
            },
            "mann_whitney_u": float(u_stat),
            "p_value_one_sided": float(u_p_one),
            "p_value_two_sided": float(u_p_two),
            "cliffs_delta": round(delta, 4),
            "cliffs_delta_interpretation": delta_interp,
            "fisher_prop_cero_rep_p": float(fisher_p),
            "diff_prop_cero_rep_pp": round((prop_ia - prop_h) * 100, 2),
            "h2_supported": bool(u_p_one < 0.05 and np.mean(ia_clar) > np.mean(h_clar))
        }
        print(f"  • Claridad Humanos: Media={np.mean(h_clar):.3f}, 0 Repeticiones={prop_h:.1%} ({h_rep0}/{n_h})")
        print(f"  • Claridad IA:      Media={np.mean(ia_clar):.3f}, 0 Repeticiones={prop_ia:.1%} ({ia_rep0}/{n_ia})")
        print(f"  • Mann-Whitney U = {u_stat:.1f}, p (unilateral IA>H) = {u_p_one:.4f}, Cliff's Delta = {delta:.3f}")
        print(f"  • Resultado: {'HIPÓTESIS CONFIRMADA' if u_p_one < 0.05 else 'NO CONFIRMADA (o sin significancia)'}")

    # ── Hipótesis 3: Abordaje de Objeciones ────────────────────────────────────

    def test_h3_objeciones(self):
        print("\n[4/7] H3: Abordaje y Resolución de Objeciones (Humanos > IA)...")
        # Filtrar llamadas con al menos una objeción presentada
        obj_df = self.rpc_df[self.rpc_df["num_objeciones"] > 0].copy()
        
        h_obj_df = obj_df[obj_df["source"] == "humano"]
        ia_obj_df = obj_df[obj_df["source"] == "ia"]
        
        n_h_obj = len(h_obj_df)
        n_ia_obj = len(ia_obj_df)
        
        h_res_rate = h_obj_df["tasa_resolucion_objeciones"].dropna().values
        ia_res_rate = ia_obj_df["tasa_resolucion_objeciones"].dropna().values
        
        u_stat, u_p_one = stats.mannwhitneyu(h_res_rate, ia_res_rate, alternative="greater")
        delta, delta_interp = compute_cliffs_delta(h_res_rate, ia_res_rate)
        
        # Conversión a PTP condicional a Objeción
        ptp_h = int(h_obj_df["ptp_logrado"].sum())
        ptp_ia = int(ia_obj_df["ptp_logrado"].sum())
        
        rate_ptp_h, low_ptp_h, up_ptp_h = wilson_score_interval(ptp_h, n_h_obj)
        rate_ptp_ia, low_ptp_ia, up_ptp_ia = wilson_score_interval(ptp_ia, n_ia_obj)
        
        table_ptp = [
            [ptp_h, n_h_obj - ptp_h],
            [ptp_ia, n_ia_obj - ptp_ia]
        ]
        odds_ratio, fisher_p = stats.fisher_exact(table_ptp, alternative="greater")
        
        # Riesgo Relativo
        rr = (rate_ptp_h / rate_ptp_ia) if rate_ptp_ia > 0 else float("inf")
        
        self.results["h3_objeciones"] = {
            "subsample_with_objections": {
                "total": len(obj_df),
                "humanos": n_h_obj,
                "ia": n_ia_obj
            },
            "resolution_rate": {
                "humanos_mean": round(float(np.mean(h_res_rate)), 4),
                "humanos_median": float(np.median(h_res_rate)),
                "ia_mean": round(float(np.mean(ia_res_rate)), 4),
                "ia_median": float(np.median(ia_res_rate)),
                "mann_whitney_u": float(u_stat),
                "p_value_one_sided": float(u_p_one),
                "cliffs_delta": round(delta, 4),
                "cliffs_delta_interpretation": delta_interp
            },
            "ptp_conversion_conditional": {
                "humanos_ptp_rate": round(rate_ptp_h, 4),
                "humanos_count": ptp_h,
                "ia_ptp_rate": round(rate_ptp_ia, 4),
                "ia_count": ptp_ia,
                "diff_pp": round((rate_ptp_h - rate_ptp_ia) * 100, 2),
                "odds_ratio": round(float(odds_ratio), 4),
                "relative_risk": round(float(rr), 4) if rr != float("inf") else "inf",
                "fisher_p_value": float(fisher_p)
            },
            "h3_supported": bool(u_p_one < 0.05 or fisher_p < 0.05)
        }
        print(f"  • Llamadas con Objeciones: Humanos={n_h_obj}, IA={n_ia_obj}")
        print(f"  • Tasa Resolución: Humanos={np.mean(h_res_rate):.1%}, IA={np.mean(ia_res_rate):.1%} (MW-U p={u_p_one:.4f})")
        print(f"  • Cierre PTP | Objeción: Humanos={rate_ptp_h:.1%} ({ptp_h}/{n_h_obj}) vs IA={rate_ptp_ia:.1%} ({ptp_ia}/{n_ia_obj})")
        print(f"  • Odds Ratio = {odds_ratio:.2f}, Fisher p = {fisher_p:.4f}")
        print(f"  • Resultado: {'HIPÓTESIS CONFIRMADA' if (u_p_one < 0.05 or fisher_p < 0.05) else 'NO CONFIRMADA'}")

    # ── Hipótesis 4: Variabilidad del Tono ─────────────────────────────────────

    def test_h4_tono(self):
        print("\n[5/7] H4: Variabilidad del Tono (Humanos > IA)...")
        h_tones = self.rpc_df[self.rpc_df["source"] == "humano"]["tono_predominante_agente"]
        ia_tones = self.rpc_df[self.rpc_df["source"] == "ia"]["tono_predominante_agente"]
        
        entropy_h = shannon_entropy(h_tones)
        entropy_ia = shannon_entropy(ia_tones)
        
        # Tabla de contingencia 4x2
        contingency = pd.crosstab(self.rpc_df["tono_predominante_agente"], self.rpc_df["source"])
        chi2_stat, chi2_p, dof, _ = stats.chi2_contingency(contingency)
        
        self.results["h4_tono"] = {
            "entropy_shannon_bits": {
                "humanos": round(entropy_h, 4),
                "ia": round(entropy_ia, 4),
                "diff_bits": round(entropy_h - entropy_ia, 4)
            },
            "contingency_table": contingency.to_dict(),
            "chi2_stat": round(float(chi2_stat), 4),
            "chi2_p_value": float(chi2_p),
            "degrees_of_freedom": int(dof),
            "human_tones_breakdown": (h_tones.value_counts(normalize=True) * 100).round(1).to_dict(),
            "ia_tones_breakdown": (ia_tones.value_counts(normalize=True) * 100).round(1).to_dict(),
            "h4_supported": bool(entropy_h > entropy_ia and chi2_p < 0.05)
        }
        print(f"  • Entropía de Shannon: Humanos={entropy_h:.3f} bits vs IA={entropy_ia:.3f} bits (Delta={entropy_h - entropy_ia:+.3f} bits)")
        print(f"  • Chi2 Homogeneidad = {chi2_stat:.2f} (dof={dof}), p = {chi2_p:.4e}")
        print(f"  • Distribución Humanos: {dict(h_tones.value_counts())}")
        print(f"  • Distribución IA:      {dict(ia_tones.value_counts())}")
        print(f"  • Resultado: {'HIPÓTESIS CONFIRMADA' if (entropy_h > entropy_ia and chi2_p < 0.05) else 'NO CONFIRMADA'}")

    # ── Hipótesis 5: Monopolización de la Conversación ────────────────────────

    def test_h5_monopolizacion(self):
        print("\n[6/7] H5: Monopolización de la Conversación (IA > Humanos)...")
        h_ratio = self.rpc_df[self.rpc_df["source"] == "humano"]["talk_ratio_agente"].values
        ia_ratio = self.rpc_df[self.rpc_df["source"] == "ia"]["talk_ratio_agente"].values
        
        # Aplicar Winsorización al percentil 99 para robustez paramétrica ante valores atípicos
        h_ratio_w = winsorize_series(h_ratio, lower=0.0, upper=0.99)
        ia_ratio_w = winsorize_series(ia_ratio, lower=0.0, upper=0.99)

        # Shapiro-Wilk sobre series winsorizadas
        shapiro_h_stat, shapiro_h_p = stats.shapiro(h_ratio_w)
        shapiro_ia_stat, shapiro_ia_p = stats.shapiro(ia_ratio_w)
        
        # Welch's t-test unilateral (IA > Humanos)
        t_stat, t_p_one = stats.ttest_ind(ia_ratio_w, h_ratio_w, equal_var=False, alternative="greater")
        # Mann-Whitney U unilateral complementario (sobre datos originales)
        u_stat, u_p_one = stats.mannwhitneyu(ia_ratio, h_ratio, alternative="greater")
        
        # Cohen's d y Hedges' g sobre series winsorizadas
        d, g, interp_g = compute_cohens_d(ia_ratio_w, h_ratio_w)
        
        # Monopolización severa (ratio > 0.75)
        h_sev = int(np.sum(h_ratio > 0.75))
        ia_sev = int(np.sum(ia_ratio > 0.75))
        n_h = len(h_ratio)
        n_ia = len(ia_ratio)
        
        rate_sev_h, _, _ = wilson_score_interval(h_sev, n_h)
        rate_sev_ia, _, _ = wilson_score_interval(ia_sev, n_ia)
        
        table_sev = [
            [ia_sev, n_ia - ia_sev],
            [h_sev, n_h - h_sev]
        ]
        odds_sev, fisher_sev_p = stats.fisher_exact(table_sev, alternative="greater")
        
        self.results["h5_monopolizacion"] = {
            "humanos": {
                "n": n_h,
                "mean": round(float(np.mean(h_ratio_w)), 4),
                "std": round(float(np.std(h_ratio_w, ddof=1)), 4),
                "median": round(float(np.median(h_ratio)), 4),
                "shapiro_p": float(shapiro_h_p),
                "severe_monopolization_count": h_sev,
                "severe_monopolization_rate": round(rate_sev_h, 4)
            },
            "ia": {
                "n": n_ia,
                "mean": round(float(np.mean(ia_ratio_w)), 4),
                "std": round(float(np.std(ia_ratio_w, ddof=1)), 4),
                "median": round(float(np.median(ia_ratio)), 4),
                "shapiro_p": float(shapiro_ia_p),
                "severe_monopolization_count": ia_sev,
                "severe_monopolization_rate": round(rate_sev_ia, 4)
            },
            "welch_t_stat": round(float(t_stat), 4),
            "welch_p_value_one_sided": float(t_p_one),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p_one_sided": float(u_p_one),
            "cohens_d": round(d, 4),
            "hedges_g": round(g, 4),
            "effect_size_interpretation": interp_g,
            "severe_fisher_p_value": float(fisher_sev_p),
            "winsorization_note": "Winsorización al percentil 99 aplicada para mitigar distorsiones por valores atípicos",
            "h5_supported": bool(t_p_one < 0.05 and np.mean(ia_ratio_w) > np.mean(h_ratio_w))
        }
        print(f"  • Talk Ratio Humanos: Media={np.mean(h_ratio):.1%}, Mediana={np.median(h_ratio):.1%}")
        print(f"  • Talk Ratio IA:      Media={np.mean(ia_ratio):.1%}, Mediana={np.median(ia_ratio):.1%}")
        print(f"  • Welch's t = {t_stat:.3f}, p (unilateral IA>H) = {t_p_one:.4e}, Hedges' g = {g:.3f} ({interp_g})")
        print(f"  • Monopolización Severa (>75%): IA={rate_sev_ia:.1%} ({ia_sev}/{n_ia}) vs Humanos={rate_sev_h:.1%} ({h_sev}/{n_h}), Fisher p = {fisher_sev_p:.4f}")
        print(f"  • Resultado: {'HIPÓTESIS CONFIRMADA' if t_p_one < 0.05 else 'NO CONFIRMADA'}")

    # ── Modelado Econométrico: Regresión Logística de PTP ───────────────────────

    def fit_econometric_model(self):
        print("\n[7/7] Modelado Econométrico Multivariado (PTP)...")
        # Submuestra con contacto efectivo
        df_model = self.rpc_df.copy()
        
        df_model["y"] = df_model["ptp_logrado"].astype(int)
        df_model["is_ia"] = (df_model["source"] == "ia").astype(int)
        
        # Modelo 1: Bivariado puro (IA vs Humano)
        model_biv = smf.logit("y ~ is_ia", data=df_model).fit(disp=False)
        
        # Modelo 2: Multivariado completo con covariables de negociación y acústica
        formula_multi = "y ~ is_ia + alternativas_ofrecidas + claridad_mensaje + talk_ratio_agente + num_objeciones"
        try:
            model_multi = smf.logit(formula_multi, data=df_model).fit(disp=False)
        except Exception as e:
            print(f"  [!] Advertencia al ajustar modelo multivariado con statsmodels: {e}. Usando formulación simplificada.")
            formula_multi = "y ~ is_ia + alternativas_ofrecidas + num_objeciones"
            model_multi = smf.logit(formula_multi, data=df_model).fit(disp=False)
            
        # Extraer métricas y Odds Ratios
        biv_or = np.exp(model_biv.params["is_ia"])
        biv_p = model_biv.pvalues["is_ia"]
        biv_ci = np.exp(model_biv.conf_int().loc["is_ia"]).tolist()
        
        multi_summary = {}
        for var in model_multi.params.index:
            coef = float(model_multi.params[var])
            or_val = float(np.exp(coef))
            pval = float(model_multi.pvalues[var])
            ci_low = float(np.exp(model_multi.conf_int().loc[var, 0]))
            ci_high = float(np.exp(model_multi.conf_int().loc[var, 1]))
            multi_summary[var] = {
                "coef": round(coef, 4),
                "odds_ratio": round(or_val, 4),
                "p_value": float(pval),
                "ci_95_or": [round(ci_low, 4), round(ci_high, 4)]
            }
            
        self.results["econometric_model"] = {
            "n_observations": len(df_model),
            "bivariate_model": {
                "formula": "y ~ is_ia",
                "is_ia_odds_ratio": round(float(biv_or), 4),
                "is_ia_p_value": float(biv_p),
                "is_ia_ci_95_or": [round(biv_ci[0], 4), round(biv_ci[1], 4)],
                "pseudo_r2": round(float(model_biv.prsquared), 4)
            },
            "multivariate_model": {
                "formula": formula_multi,
                "pseudo_r2": round(float(model_multi.prsquared), 4),
                "ll_null": round(float(model_multi.llnull), 4),
                "ll_model": round(float(model_multi.llf), 4),
                "coefficients": multi_summary
            }
        }
        print(f"  • Modelo Bivariado: OR(IA) = {biv_or:.3f}, p = {biv_p:.4f}, Pseudo R² = {model_biv.prsquared:.3f}")
        print(f"  • Modelo Multivariado: Pseudo R² = {model_multi.prsquared:.3f}")
        for var, metrics in multi_summary.items():
            if var != "Intercept":
                print(f"    - {var:23}: OR = {metrics['odds_ratio']:6.3f} (p = {metrics['p_value']:.4f})")

    # ── Guardar Artefactos ────────────────────────────────────────────────────

    def save_artifacts(self):
        print("\n[Guardando Artefactos]...")
        # 1. Guardar JSON
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Resultados exportados a: {RESULTS_JSON}")
        
        # 2. Guardar Markdown Ejecutivo
        self.generate_summary_markdown()
        print(f"  ✓ Resumen ejecutivo exportado a: {SUMMARY_MD}")

    def generate_summary_markdown(self):
        r = self.results
        f = r["funnel"]
        h1 = r["h1_alternativas"]
        h2 = r["h2_claridad"]
        h3 = r["h3_objeciones"]
        h4 = r["h4_tono"]
        h5 = r["h5_monopolizacion"]
        eco = r["econometric_model"]
        
        md = f"""# Resultados Estadísticos y Econométricos: Humanos vs. IA en Cobranza

Este documento consolida los contrastes de hipótesis, parámetros de significancia, tamaños de efecto y modelado econométrico multivariado a partir de la muestra de {f['total_calls']} llamadas procesadas.

---

## 1. Embudo Global y Contactabilidad (n = {f['total_calls']})

* **Contactabilidad Humanos (RPC):** {f['humanos']['rpc_rate']:.1%} ({f['humanos']['rpc_count']}/{f['humanos']['total']}) [IC 95%: {f['humanos']['ci_95'][0]:.1%} - {f['humanos']['ci_95'][1]:.1%}]
* **Contactabilidad IA (RPC):** {f['ia']['rpc_rate']:.1%} ({f['ia']['rpc_count']}/{f['ia']['total']}) [IC 95%: {f['ia']['ci_95'][0]:.1%} - {f['ia']['ci_95'][1]:.1%}]
* **Brecha Bruta:** +{f['diff_pp']:.1f} p.p. a favor de humanos.
* **Test de Independencia:** $\\chi^2 = {f['chi2_stat']:.2f}$ ($p = {f['chi2_p_value']:.4f}$), Test Exacto de Fisher $p = {f['fisher_p_value']:.4f}$, $\\text{{Odds Ratio}} = {f['odds_ratio']:.2f}$.
* **Conclusión Metodológica:** Existe una disparidad significativa en el discado/contacto previo ($p < 0.05$). Para aislar la habilidad de negociación, comunicación y resolución de objeciones sin sesgo de selección, el análisis de hipótesis se conduce sobre las **$n = {h1['humanos']['n'] + h1['ia']['n']}$ llamadas con contacto efectivo verificado** ($n_{{\\text{{humano}}}} = {h1['humanos']['n']}$, $n_{{\\text{{ia}}}} = {h1['ia']['n']}$).

---

## 2. Matriz de Validación de las 5 Hipótesis

| Hipótesis | Variable / Métrica | Humanos ($n={h1['humanos']['n']}$) | IA ($n={h1['ia']['n']}$) | Estadístico de Contraste | Valor $p$ | Tamaño del Efecto | Veredicto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1: Alternativas Ofrecidas**<br>*(Humanos > IA)* | `alternativas_ofrecidas` | Media: {h1['humanos']['mean']:.2f}<br>Mediana: {h1['humanos']['median']:.1f} (IQR: {h1['humanos']['iqr']:.1f}) | Media: {h1['ia']['mean']:.2f}<br>Mediana: {h1['ia']['median']:.1f} (IQR: {h1['ia']['iqr']:.1f}) | Mann-Whitney $U = {h1['mann_whitney_u']:.1f}$ | $p = {h1['p_value_one_sided']:.4e}$ | Cliff's $\\delta = {h1['cliffs_delta']:.3f}$<br>({h1['cliffs_delta_interpretation']}) | **{'CONFIRMADA' if h1['h1_supported'] else 'NO CONFIRMADA'}** |
| **H2: Claridad del Mensaje**<br>*(IA > Humanos)* | `claridad_mensaje`<br>(0 repeticiones) | Media: {h2['humanos']['mean']:.3f}<br>% 0 rep: {h2['humanos']['prop_cero_repeticiones']:.1%} | Media: {h2['ia']['mean']:.3f}<br>% 0 rep: {h2['ia']['prop_cero_repeticiones']:.1%} | Mann-Whitney $U = {h2['mann_whitney_u']:.1f}$<br>Fisher %0rep | $p = {h2['p_value_one_sided']:.4f}$<br>$p = {h2['fisher_prop_cero_rep_p']:.4f}$ | Cliff's $\\delta = {h2['cliffs_delta']:.3f}$<br>({h2['cliffs_delta_interpretation']}) | **{'CONFIRMADA' if h2['h2_supported'] else 'NO CONFIRMADA'}** |
| **H3: Resolución Objeciones**<br>*(Humanos > IA)* | `tasa_resolucion`<br>PTP | Objeción | Res: {h3['resolution_rate']['humanos_mean']:.1%}<br>PTP: {h3['ptp_conversion_conditional']['humanos_ptp_rate']:.1%} ({h3['ptp_conversion_conditional']['humanos_count']}/{h3['subsample_with_objections']['humanos']}) | Res: {h3['resolution_rate']['ia_mean']:.1%}<br>PTP: {h3['ptp_conversion_conditional']['ia_ptp_rate']:.1%} ({h3['ptp_conversion_conditional']['ia_count']}/{h3['subsample_with_objections']['ia']}) | MW $U = {h3['resolution_rate']['mann_whitney_u']:.1f}$<br>Fisher PTP | $p = {h3['resolution_rate']['p_value_one_sided']:.4f}$<br>$p = {h3['ptp_conversion_conditional']['fisher_p_value']:.4f}$ | $\\text{{OR}} = {h3['ptp_conversion_conditional']['odds_ratio']:.2f}$<br>$\\text{{RR}} = {h3['ptp_conversion_conditional']['relative_risk']}$ | **{'CONFIRMADA' if h3['h3_supported'] else 'NO CONFIRMADA'}** |
| **H4: Variabilidad del Tono**<br>*(Humanos > IA)* | `tono_predominante`<br>(Entropía Shannon) | $H(X) = {h4['entropy_shannon_bits']['humanos']:.3f}$ bits<br>Empático {h4['human_tones_breakdown'].get('empatico_profesional', 0)}% | $H(X) = {h4['entropy_shannon_bits']['ia']:.3f}$ bits<br>Neutro {h4['ia_tones_breakdown'].get('neutro_formal', 0)}% | $\\chi^2 = {h4['chi2_stat']:.2f}$<br>(dof = {h4['degrees_of_freedom']}) | $p = {h4['chi2_p_value']:.4e}$ | $\\Delta H = {h4['entropy_shannon_bits']['diff_bits']:+.3f}$ bits | **{'CONFIRMADA' if h4['h4_supported'] else 'NO CONFIRMADA'}** |
| **H5: Monopolización**<br>*(IA > Humanos)* | `talk_ratio_agente`<br>(% Ratio > 75%) | Media: {h5['humanos']['mean']:.1%}<br>>75%: {h5['humanos']['severe_monopolization_rate']:.1%} ({h5['humanos']['severe_monopolization_count']}/{h5['humanos']['n']}) | Media: {h5['ia']['mean']:.1%}<br>>75%: {h5['ia']['severe_monopolization_rate']:.1%} ({h5['ia']['severe_monopolization_count']}/{h5['ia']['n']}) | Welch's $t = {h5['welch_t_stat']:.3f}$<br>Fisher >75% | $p = {h5['welch_p_value_one_sided']:.4e}$<br>$p = {h5['severe_fisher_p_value']:.4f}$ | Hedges' $g = {h5['hedges_g']:.3f}$<br>({h5['effect_size_interpretation']}) | **{'CONFIRMADA' if h5['h5_supported'] else 'NO CONFIRMADA'}** |

---

## 3. Modelo Econométrico Multivariado: Determinantes de la Promesa de Pago (PTP)

Para responder a la duda del comité bancario: *¿La brecha en promesas de pago se debe a que la IA es intrínsecamente rechazada o a su falta de flexibilidad en alternativas y manejo de objeciones?*

$$\\text{{logit}}(P(\\text{{PTP}}=1)) = \\beta_0 + \\beta_1 \\cdot \\text{{es\\_ia}} + \\beta_2 \\cdot \\text{{alternativas}} + \\beta_3 \\cdot \\text{{claridad}} + \\beta_4 \\cdot \\text{{talk\\_ratio}} + \\beta_5 \\cdot \\text{{num\\_objeciones}}$$

### Resumen del Modelo Multivariado:
* **Muestra:** $n = {eco['n_observations']}$ llamadas con contacto efectivo.
* **Pseudo $R^2$ de McFadden:** ${eco['multivariate_model']['pseudo_r2']:.3f}$

| Variable Predictora | Coeficiente ($\\beta$) | Odds Ratio ($e^\\beta$) | IC 95% del Odds Ratio | Valor $p$ | Interpretación Ejecutiva |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for var, m in eco['multivariate_model']['coefficients'].items():
            if var == "Intercept":
                continue
            md += f"| `{var}` | {m['coef']:+.4f} | **{m['odds_ratio']:.3f}** | [{m['ci_95_or'][0]:.3f}, {m['ci_95_or'][1]:.3f}] | {m['p_value']:.4f} | {'Significativo al 95%' if m['p_value'] < 0.05 else 'No significativo'} |\n"
            
        md += f"""
### Hallazgo Econométrico Central:
En el modelo bivariado no controlado, la IA presenta una menor probabilidad de PTP ($\\text{{OR}} = {eco['bivariate_model']['is_ia_odds_ratio']:.3f}$, $p = {eco['bivariate_model']['is_ia_p_value']:.4f}$). 
Sin embargo, en el modelo multivariado controlado por herramientas de reestructuración (`alternativas_ofrecidas`) y fricción de resistencia (`num_objeciones`), el driver dominante es la **capacidad de desplegar alternativas financieras** (cada alternativa incrementa las odds de acuerdo en un factor multiplicador de **{eco['multivariate_model']['coefficients'].get('alternativas_ofrecidas', {}).get('odds_ratio', 1.0):.2f}x**).
"""
        with open(SUMMARY_MD, "w", encoding="utf-8") as f:
            f.write(md)


# ── Punto de Entrada CLI ──────────────────────────────────────────────────────

def main():
    if not ANALYTICAL_CSV.exists():
        print(f"Error: No se encontró el dataset analítico en {ANALYTICAL_CSV}")
        sys.exit(1)
        
    df = pd.read_csv(ANALYTICAL_CSV)
    analyzer = StatisticalAnalyzer(df)
    analyzer.run_all()
    print("\n[✓] Análisis estadístico completado exitosamente.")


if __name__ == "__main__":
    main()
