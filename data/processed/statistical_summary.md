# Resultados Estadísticos y Econométricos: Humanos vs. IA en Cobranza

Este documento consolida los contrastes de hipótesis, parámetros de significancia, tamaños de efecto y modelado econométrico multivariado a partir de la muestra de 100 llamadas procesadas.

---

## 1. Embudo Global y Contactabilidad (n = 100)

* **Contactabilidad Humanos (RPC):** 74.0% (37/50) [IC 95%: 60.5% - 84.1%]
* **Contactabilidad IA (RPC):** 48.0% (24/50) [IC 95%: 34.8% - 61.5%]
* **Brecha Bruta:** +26.0 p.p. a favor de humanos.
* **Test de Independencia:** $\chi^2 = 6.05$ ($p = 0.0139$), Test Exacto de Fisher $p = 0.0134$, $\text{Odds Ratio} = 3.08$.
* **Conclusión Metodológica:** Existe una disparidad significativa en el discado/contacto previo ($p < 0.05$). Para aislar la habilidad de negociación, comunicación y resolución de objeciones sin sesgo de selección, el análisis de hipótesis se conduce sobre las **$n = 61$ llamadas con contacto efectivo verificado** ($n_{\text{humano}} = 37$, $n_{\text{ia}} = 24$).

---

## 2. Matriz de Validación de las 5 Hipótesis

| Hipótesis | Variable / Métrica | Humanos ($n=37$) | IA ($n=24$) | Estadístico de Contraste | Valor $p$ | Tamaño del Efecto | Veredicto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1: Alternativas Ofrecidas**<br>*(Humanos > IA)* | `alternativas_ofrecidas` | Media: 1.95<br>Mediana: 2.0 (IQR: 3.0) | Media: 1.88<br>Mediana: 2.0 (IQR: 0.0) | Mann-Whitney $U = 387.0$ | $p = 8.1251e-01$ | Cliff's $\delta = -0.128$<br>(Despreciable) | **NO CONFIRMADA** |
| **H2: Claridad del Mensaje**<br>*(IA > Humanos)* | `claridad_mensaje`<br>(0 repeticiones) | Media: 0.714<br>% 0 rep: 56.8% | Media: 0.585<br>% 0 rep: 33.3% | Mann-Whitney $U = 363.5$<br>Fisher %0rep | $p = 0.8988$<br>$p = 0.9806$ | Cliff's $\delta = -0.181$<br>(Pequeño) | **NO CONFIRMADA** |
| **H3: Resolución Objeciones**<br>*(Humanos > IA)* | `tasa_resolucion`<br>PTP | Objeción | Res: 67.3%<br>PTP: 53.8% (14/26) | Res: 41.2%<br>PTP: 42.1% (8/19) | MW $U = 316.5$<br>Fisher PTP | $p = 0.0393$<br>$p = 0.3173$ | $\text{OR} = 1.60$<br>$\text{RR} = 1.2788$ | **CONFIRMADA** |
| **H4: Variabilidad del Tono**<br>*(Humanos > IA)* | `tono_predominante`<br>(Entropía Shannon) | $H(X) = 1.249$ bits<br>Empático 48.6% | $H(X) = 1.196$ bits<br>Neutro 54.2% | $\chi^2 = 0.40$<br>(dof = 2) | $p = 8.1871e-01$ | $\Delta H = +0.052$ bits | **NO CONFIRMADA** |
| **H5: Monopolización**<br>*(IA > Humanos)* | `talk_ratio_agente`<br>(% Ratio > 75%) | Media: 49.5%<br>>75%: 0.0% (0/37) | Media: 46.2%<br>>75%: 0.0% (0/24) | Welch's $t = -1.057$<br>Fisher >75% | $p = 8.5203e-01$<br>$p = 1.0000$ | Hedges' $g = -0.279$<br>(Pequeño) | **NO CONFIRMADA** |

---

## 3. Modelo Econométrico Multivariado: Determinantes de la Promesa de Pago (PTP)

Para responder a la duda del comité bancario: *¿La brecha en promesas de pago se debe a que la IA es intrínsecamente rechazada o a su falta de flexibilidad en alternativas y manejo de objeciones?*

$$\text{logit}(P(\text{PTP}=1)) = \beta_0 + \beta_1 \cdot \text{es\_ia} + \beta_2 \cdot \text{alternativas} + \beta_3 \cdot \text{claridad} + \beta_4 \cdot \text{talk\_ratio} + \beta_5 \cdot \text{num\_objeciones}$$

### Resumen del Modelo Multivariado:
* **Muestra:** $n = 61$ llamadas con contacto efectivo.
* **Pseudo $R^2$ de McFadden:** $0.102$

| Variable Predictora | Coeficiente ($\beta$) | Odds Ratio ($e^\beta$) | IC 95% del Odds Ratio | Valor $p$ | Interpretación Ejecutiva |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `is_ia` | -0.8807 | **0.414** | [0.127, 1.356] | 0.1453 | No significativo |
| `alternativas_ofrecidas` | +0.7338 | **2.083** | [1.026, 4.229] | 0.0422 | Significativo al 95% |
| `claridad_mensaje` | -0.3370 | **0.714** | [0.128, 3.973] | 0.7004 | No significativo |
| `talk_ratio_agente` | -1.0996 | **0.333** | [0.002, 53.932] | 0.6718 | No significativo |
| `num_objeciones` | -0.4227 | **0.655** | [0.448, 0.958] | 0.0293 | Significativo al 95% |

### Hallazgo Econométrico Central:
En el modelo bivariado no controlado, la IA presenta una menor probabilidad de PTP ($\text{OR} = 0.609$, $p = 0.3495$). 
Sin embargo, en el modelo multivariado controlado por herramientas de reestructuración (`alternativas_ofrecidas`) y fricción de resistencia (`num_objeciones`), el driver dominante es la **capacidad de desplegar alternativas financieras** (cada alternativa incrementa las odds de acuerdo en un factor multiplicador de **2.08x**).
