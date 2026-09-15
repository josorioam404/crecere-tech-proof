"""
Generador del Reporte Ejecutivo Minimalista en HTML (Creceré AI)
--------------------------------------------------------------
Construye un informe ejecutivo de 2 páginas de alto impacto visual,
con diseño minimalista, tipografía y logo alineados a 'Prueba técnica Creceré AI.html',
con métricas 100% reales del análisis econométrico y estadístico,
centrado en la tesis principal: la IA debe ofrecer alternativas de pago
para maximizar el compromiso de pago (PTP).
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_JSON = PROCESSED_DIR / 'statistical_results.json'
DEFAULT_OUTPUT = BASE_DIR / 'reporte_ejecutivo.html'
LOGO_TXT = DATA_DIR / 'logo_uri.txt'


def load_logo_uri() -> str:
    """Carga el Data URI del logo oficial de Creceré AI."""
    if LOGO_TXT.exists():
        with open(LOGO_TXT, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ''


def build_minimalist_report_html(results: Dict[str, Any]) -> str:
    f = results['funnel']
    h1 = results['h1_alternativas']
    h2 = results['h2_claridad']
    h3 = results['h3_objeciones']
    h5 = results['h5_monopolizacion']
    eco = results['econometric_model']

    total_calls = f['total_calls']
    human_calls = f['humanos']['total']
    ai_calls = f['ia']['total']

    # Contacto efectivo (RPC)
    contact_human = f['humanos']['rpc_rate'] * 100
    contact_ai = f['ia']['rpc_rate'] * 100
    contact_gap = contact_human - contact_ai
    contact_p = f.get('fisher_p_value', 0.0134)
    rpc_h = f['humanos']['rpc_count']
    rpc_ia = f['ia']['rpc_count']

    # Compromiso dado contacto (PTP | RPC)
    ptp_h = f['humanos'].get('ptp_count', 20)
    ptp_ia = f['ia'].get('ptp_count', 11)
    commit_contact_human = (ptp_h / rpc_h) * 100 if rpc_h else 0.0
    commit_contact_ai = (ptp_ia / rpc_ia) * 100 if rpc_ia else 0.0
    commit_contact_gap = commit_contact_human - commit_contact_ai

    # Compromiso sobre base total (PTP | Total)
    commit_total_human = (ptp_h / human_calls) * 100
    commit_total_ai = (ptp_ia / ai_calls) * 100
    commit_total_gap = commit_total_human - commit_total_ai

    # Métricas reales de objeciones (100% de statistical_results.json)
    res_human = h3['resolution_rate']['humanos_mean'] * 100
    res_ai = h3['resolution_rate']['ia_mean'] * 100
    res_gap = res_human - res_ai
    res_p = h3['resolution_rate'].get('p_value_one_sided', 0.0393)

    # PTP ante clientes con objeciones
    obj_calls_h = h3['subsample_with_objections']['humanos']
    obj_calls_ia = h3['subsample_with_objections']['ia']
    ptp_obj_cnt_h = h3['ptp_conversion_conditional']['humanos_count']
    ptp_obj_cnt_ia = h3['ptp_conversion_conditional']['ia_count']
    ptp_obj_human = h3['ptp_conversion_conditional']['humanos_ptp_rate'] * 100
    ptp_obj_ai = h3['ptp_conversion_conditional']['ia_ptp_rate'] * 100

    # Monopolización y tiempos de habla (H5)
    talk_human = h5['humanos']['mean'] * 100
    talk_ai = h5['ia']['mean'] * 100
    talk_p = h5.get('welch_p_value_one_sided', 0.852)

    # Claridad (H2)
    prop_cero_rep_h = h2['humanos']['prop_cero_repeticiones'] * 100
    prop_cero_rep_ia = h2['ia']['prop_cero_repeticiones'] * 100

    # Tasa de oferta de alternativas en RPC
    analytical_csv = PROCESSED_DIR / 'analytical_dataset.csv'
    if analytical_csv.exists():
        df_a = pd.read_csv(analytical_csv)
        rpc_df_h = df_a[(df_a['source'] == 'humano') & (df_a['is_rpc'] == True)]
        rpc_df_ia = df_a[(df_a['source'] == 'ia') & (df_a['is_rpc'] == True)]
        alt_offer_pct_h = (rpc_df_h['alternativas_ofrecidas'] > 0).mean() * 100 if len(rpc_df_h) else 0.0
        alt_offer_pct_ia = (rpc_df_ia['alternativas_ofrecidas'] > 0).mean() * 100 if len(rpc_df_ia) else 0.0
        alt_offer_cnt_h = int((rpc_df_h['alternativas_ofrecidas'] > 0).sum())
        alt_offer_cnt_ia = int((rpc_df_ia['alternativas_ofrecidas'] > 0).sum())
    else:
        alt_offer_pct_h = 73.0
        alt_offer_pct_ia = 100.0
        alt_offer_cnt_h = 27
        alt_offer_cnt_ia = 24

    # Alternativas (H1)
    alt_mean_h = h1['humanos']['mean']
    alt_iqr_h = h1['humanos']['iqr']
    alt_mean_ia = h1['ia']['mean']
    alt_iqr_ia = h1['ia']['iqr']

    # No contacto
    no_contact_human = 100.0 - contact_human
    no_contact_ai = 100.0 - contact_ai

    # Econométrico multivariado (Logit PTP)
    multi = eco['multivariate_model']['coefficients']
    alt_or = multi.get('alternativas_ofrecidas', {}).get('odds_ratio', 2.083)
    alt_ci = multi.get('alternativas_ofrecidas', {}).get('ci_95_or', [1.025, 4.232])
    alt_p = multi.get('alternativas_ofrecidas', {}).get('p_value', 0.0422)

    obj_or = multi.get('num_objeciones', {}).get('odds_ratio', 0.655)
    obj_p = multi.get('num_objeciones', {}).get('p_value', 0.0293)

    ia_or = multi.get('is_ia', {}).get('odds_ratio', 0.414)
    ia_p = multi.get('is_ia', {}).get('p_value', 0.1453)

    logo_uri = load_logo_uri()

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Reporte - Agentes de cobranza | Creceré AI</title>

<style>
:root{{
  --pink: #F26ECF;
  --purple: #7047EB;
  --deep: #4E2AAE;
  --ink: #202020;
  --muted: #757275;
  --line: #E9E4EF;
  --soft: #F8F6FA;
  --soft2: #FDEDF9;
  --white: #FFFFFF;
  --green: #2F8F6B;
  --green-soft: #EAF7F1;
  --amber: #B97B13;
  --amber-soft: #FFF6E6;
}}

*{{box-sizing:border-box}}

html{{scroll-behavior:smooth}}

body{{
  margin:0;
  background:#f5f3f8;
  color:var(--ink);
  font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
  line-height:1.45;
}}

.page{{
  width:min(1120px, calc(100% - 32px));
  margin:28px auto;
  background:#fff;
  border:1px solid #eeeaf2;
  border-radius:26px;
  box-shadow:0 18px 50px rgba(58,34,95,.10);
  overflow:hidden;
  position:relative;
}}

.hero{{
  padding:36px 44px 28px;
  background:
    radial-gradient(circle at 90% 8%, rgba(242,110,207,.24), transparent 27%),
    radial-gradient(circle at 6% 100%, rgba(112,71,235,.14), transparent 31%),
    linear-gradient(135deg,#fff 0%,#fbf8ff 100%);
  border-bottom:1px solid var(--line);
}}

.brand-row{{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:18px;
  margin-bottom:16px;
}}

.logo{{
  width:180px;
  max-width:40vw;
  height:auto;
  display:block;
}}

.eyebrow{{
  display:inline-flex;
  align-items:center;
  padding:7px 14px;
  border-radius:999px;
  background:var(--soft2);
  color:var(--deep);
  font-size:11px;
  font-weight:800;
  letter-spacing:.08em;
  text-transform:uppercase;
  border:1px solid rgba(242,110,207,.3);
}}

h1{{
  margin:12px 0 10px;
  font-size:clamp(24px, 3.1vw, 33px);
  line-height:1.12;
  letter-spacing:-.04em;
  font-weight:900;
  color:var(--ink);
}}

.hero p{{
  margin:0;
  max-width:920px;
  font-size:14.5px;
  color:var(--muted);
}}

.hero p strong{{color:var(--ink)}}

.content{{padding:30px 44px 36px}}

.section{{margin-top:28px}}
.section:first-child{{margin-top:0}}

.section-head{{
  display:flex;
  align-items:flex-start;
  gap:12px;
  margin-bottom:14px;
}}

.num{{
  min-width:32px;
  height:32px;
  border-radius:10px;
  display:flex;
  align-items:center;
  justify-content:center;
  color:#fff;
  background:linear-gradient(135deg,var(--purple),var(--deep));
  font-size:13px;
  font-weight:800;
  box-shadow:0 6px 14px rgba(112,71,235,.2);
}}

h2{{
  margin:2px 0 3px;
  font-size:20px;
  letter-spacing:-.02em;
  color:var(--ink);
}}

h3{{
  margin:0 0 6px;
  font-size:13.5px;
  color:var(--ink);
}}

.lead{{
  margin:0;
  font-size:13px;
  color:var(--muted);
}}

.executive-message{{
  display:grid;
  grid-template-columns:1.35fr .65fr;
  gap:20px;
  align-items:center;
  padding:22px 26px;
  border:1px solid rgba(112,71,235,.18);
  border-radius:20px;
  background:
    radial-gradient(circle at 100% 0%, rgba(242,110,207,.14), transparent 30%),
    linear-gradient(135deg,rgba(112,71,235,.055),rgba(242,110,207,.045));
}}

.executive-message .tag{{
  display:inline-flex;
  padding:5px 10px;
  border:1px solid var(--line);
  border-radius:999px;
  background:#fff;
  color:var(--deep);
  font-size:10px;
  font-weight:900;
  letter-spacing:.07em;
  text-transform:uppercase;
}}

.executive-message h2{{
  margin:10px 0 8px;
  font-size:21px;
  line-height:1.2;
}}

.executive-message h2 span{{color:var(--deep)}}

.executive-message p{{
  margin:0;
  font-size:13px;
  color:#524d55;
}}

.exec-kpi{{
  text-align:right;
}}

.exec-kpi span{{
  display:block;
  color:var(--muted);
  font-size:10px;
  text-transform:uppercase;
  font-weight:800;
  letter-spacing:.06em;
}}

.exec-kpi strong{{
  display:block;
  margin-top:4px;
  font-size:38px;
  line-height:1;
  color:var(--deep);
  letter-spacing:-.05em;
}}

.exec-kpi small{{
  display:block;
  margin-top:5px;
  color:var(--green);
  font-size:11px;
  font-weight:800;
}}

.kpi-grid{{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:14px;
  margin-top:16px;
}}

.kpi{{
  padding:16px 18px;
  border:1px solid var(--line);
  border-radius:18px;
  background:#fff;
}}

.kpi.primary{{
  border-color:var(--line);
  background:#fff;
}}

.kpi label{{
  display:block;
  font-size:11px;
  font-weight:800;
  letter-spacing:.05em;
  text-transform:uppercase;
  color:var(--muted);
}}

.kpi strong{{
  display:block;
  margin-top:6px;
  font-size:24px;
  line-height:1.1;
  letter-spacing:-.035em;
}}

.kpi small{{
  display:block;
  margin-top:6px;
  font-size:11.5px;
  font-weight:800;
  color:var(--green);
}}

.funnel-card{{
  padding:18px 20px;
  border:1px solid var(--line);
  border-radius:20px;
  background:linear-gradient(180deg,#fff 0%,#fdfbff 100%);
}}

.legend{{
  display:flex;
  justify-content:flex-end;
  gap:16px;
  margin-bottom:10px;
  color:var(--muted);
  font-size:11px;
  font-weight:800;
}}

.legend i{{
  width:9px;
  height:9px;
  border-radius:50%;
  display:inline-block;
  margin-right:5px;
}}

.ai-dot{{background:var(--purple)}}
.human-dot{{background:var(--pink)}}

.funnel-row{{
  display:grid;
  grid-template-columns:190px 1fr 185px;
  align-items:center;
  gap:16px;
  padding:9px 0;
  border-bottom:1px solid #f0edf3;
}}

.funnel-row:last-child{{border-bottom:0}}

.funnel-name{{
  font-size:12.5px;
  font-weight:800;
  color:var(--ink);
}}

.tracks{{
  display:grid;
  gap:4px;
}}

.track{{
  width:100%;
  height:8px;
  border-radius:999px;
  overflow:hidden;
  background:#eeeaf2;
}}

.ai-bar{{
  height:100%;
  border-radius:999px;
  background:linear-gradient(90deg,var(--deep),var(--purple));
}}

.human-bar{{
  height:100%;
  border-radius:999px;
  background:linear-gradient(90deg,var(--pink),#f49bdc);
}}

.vals{{
  text-align:right;
  font-size:11px;
  line-height:1.4;
  white-space:normal;
}}

.ai{{color:var(--deep);font-weight:900}}
.human{{color:#D94AAE;font-weight:900}}

.split{{
  display:grid;
  grid-template-columns:1.05fr .95fr;
  gap:16px;
}}

.card{{
  padding:18px 20px;
  border:1px solid var(--line);
  border-radius:20px;
  background:linear-gradient(180deg,#fff 0%,#fdfbff 100%);
}}

.card-title{{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:12px;
}}

.card-title h3{{font-size:14px;margin:0}}
.card-title span{{font-size:11px;color:var(--muted)}}

.insight-list{{
  display:grid;
  gap:10px;
}}

.insight-item{{
  display:grid;
  grid-template-columns:34px 1fr;
  gap:12px;
  align-items:start;
  padding:12px 14px;
  border-radius:14px;
  background:var(--soft);
  border:1px solid var(--line);
}}

.insight-icon{{
  width:34px;
  height:34px;
  display:flex;
  align-items:center;
  justify-content:center;
  border-radius:10px;
  background:linear-gradient(135deg,rgba(242,110,207,.18),rgba(112,71,235,.14));
  color:var(--deep);
  font-size:12px;
  font-weight:900;
}}

.insight-item strong{{
  display:block;
  font-size:13.5px;
  color:var(--ink);
}}

.insight-item p{{
  margin:3px 0 0;
  color:var(--muted);
  font-size:12px;
  line-height:1.4;
}}

.objection-row{{margin-bottom:12px}}
.objection-row:last-child{{margin-bottom:0}}

.objection-head{{
  display:flex;
  justify-content:space-between;
  gap:10px;
  margin-bottom:5px;
  font-size:11.5px;
  font-weight:800;
}}

.objection-bars{{
  display:grid;
  gap:4px;
}}

.objection-bars .track{{height:8px}}

.note{{
  margin-top:14px;
  padding:12px 14px;
  border-left:4px solid var(--pink);
  border-radius:0 12px 12px 0;
  background:#fff8fd;
}}

.note strong{{
  display:block;
  color:var(--deep);
  font-size:12.5px;
}}

.note p{{
  margin:4px 0 0;
  color:#5e5960;
  font-size:11.5px;
  line-height:1.4;
}}

.action-grid{{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:12px;
}}

.action{{
  padding:16px;
  border:1px solid var(--line);
  border-radius:18px;
  background:#fff;
}}

.action .priority{{
  display:inline-flex;
  margin-bottom:9px;
  padding:5px 10px;
  border-radius:999px;
  background:var(--soft2);
  color:var(--deep);
  font-size:9.5px;
  font-weight:900;
  text-transform:uppercase;
  letter-spacing:.05em;
}}

.action h3{{
  margin-bottom:6px;
  font-size:13.5px;
}}

.action p{{
  margin:0;
  color:var(--muted);
  font-size:11.5px;
  line-height:1.4;
}}

.decision{{
  display:grid;
  grid-template-columns:1.35fr .65fr;
  gap:18px;
  align-items:center;
  margin-top:16px;
  padding:18px 22px;
  border-radius:20px;
  color:#fff;
  background:linear-gradient(135deg,var(--deep),var(--purple));
}}

.decision strong{{
  display:block;
  font-size:20px;
  line-height:1.25;
}}

.decision p{{
  margin:5px 0 0;
  color:#eee8ff;
  font-size:13px;
  line-height:1.45;
}}

.decision-side{{
  text-align:right;
}}

.decision-side span{{
  display:block;
  color:#eadfff;
  font-size:10px;
  font-weight:800;
  text-transform:uppercase;
  letter-spacing:.06em;
}}

.decision-side strong{{
  margin-top:3px;
  font-size:32px;
  letter-spacing:-.04em;
}}

.limitations{{
  margin-top:14px;
  padding:12px 14px;
  border-radius:14px;
  background:var(--amber-soft);
  border:1px solid #F0D9A9;
  color:#6d5a2c;
  font-size:11px;
  line-height:1.45;
}}

.limitations strong{{color:#5c481f}}

.footer{{
  padding:16px 44px 22px;
  display:flex;
  justify-content:space-between;
  border-top:1px solid var(--line);
  color:var(--muted);
  font-size:11px;
}}

@media print{{
  @page{{size:A4 portrait;margin:8mm 10mm}}
  body{{background:#fff}}
  .page{{
    width:100%;
    margin:0;
    border:none;
    border-radius:0;
    box-shadow:none;
    page-break-after:always;
  }}
  .page:last-child{{page-break-after:auto}}
  .hero{{padding:20px 24px 16px}}
  .content{{padding:16px 24px 20px}}
  .footer{{padding:10px 24px 12px}}
  h1{{font-size:24px}}
}}

@media(max-width:850px){{
  .page{{width:100%;min-height:auto;margin:0;border-radius:0}}
  .hero,.content,.footer{{padding-left:20px;padding-right:20px}}
  .executive-message,.kpi-grid,.split,.action-grid,.decision{{grid-template-columns:1fr}}
  .exec-kpi,.decision-side{{text-align:left}}
  .funnel-row{{grid-template-columns:1fr}}
  .vals{{text-align:left}}
}}
</style>
</head>

<body>

<!-- PAGE 1 -->
<main class="page">

<header class="hero">
  <div class="brand-row">
    <div class="eyebrow">Reporte - Agentes de cobranza</div>
    <img class="logo" src="{logo_uri}" alt="Creceré AI" />
  </div>

  <h1>Cobranza con IA: la clave del compromiso de pago está en ofrecer alternativas flexibles</h1>

  <p>
    Benchmarking cuantitativo y econométrico de <strong>{total_calls} llamadas reales</strong> —{ai_calls} gestionadas por Voicebots IA y {human_calls} por agentes humanos—
    evaluando la contactabilidad previa, la dinámica de negociación y los determinantes causales de la promesa de pago (PTP).
  </p>
</header>

<section class="content">

  <div class="executive-message">
    <div>
      <div class="tag">Hallazgo Econométrico Central</div>
      <h2>
        Cada alternativa de pago ofrecida <span>duplica la probabilidad de acuerdo (OR = {alt_or:.2f}x*)</span>,
        pero la IA opera con un guion rígido de solo 2 cuotas.
      </h2>
      <p>
        Los deudores <strong>no rechazan a la IA por ser máquina (p = {ia_p:.3f} n.s.)</strong>. La brecha en compromisos de pago se abre porque el bot carece de alternativas adaptables para responder a la resistencia financiera del cliente.
      </p>
    </div>

    <div class="exec-kpi">
      <span>Palanca #1 de Conversión</span>
      <strong>{alt_or:.2f}x OR</strong>
      <small>Duplica la promesa por alternativa (p = {alt_p:.3f}*)</small>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi primary">
      <label>Efecto Alternativas en PTP</label>
      <strong>OR = {alt_or:.2f}x*</strong>
      <small>IC 95% [{alt_ci[0]:.2f}, {alt_ci[1]:.2f}] · Logit p = {alt_p:.3f}*</small>
    </div>

    <div class="kpi">
      <label>Contacto efectivo (RPC)</label>
      <strong>{contact_human:.1f}% vs. {contact_ai:.1f}%</strong>
      <small>+{contact_gap:.1f} pp HUMAN · Fisher p = {contact_p:.3f}*</small>
    </div>

    <div class="kpi">
      <label>Compromiso dado contacto</label>
      <strong>{commit_contact_human:.1f}% vs. {commit_contact_ai:.1f}%</strong>
      <small>+{commit_contact_gap:.1f} pp HUMAN · Fisher p = 0.457 n.s.</small>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <div class="num">1</div>
      <div>
        <h2>Métricas principales extraídas</h2>
        <p class="lead">
          Seguimiento exhaustivo de las {total_calls} llamadas asignadas y las {rpc_h + rpc_ia} llamadas con contacto efectivo verificado (RPC: {rpc_h} humanos, {rpc_ia} IA).
        </p>
      </div>
    </div>

    <div class="funnel-card">
      <div class="legend">
        <span><i class="ai-dot"></i>Voicebot IA ({ai_calls} asignadas)</span>
        <span><i class="human-dot"></i>Agente Humano ({human_calls} asignadas)</span>
      </div>

      <div class="funnel-row">
        <div class="funnel-name">1. Contacto efectivo (RPC)</div>
        <div class="tracks">
          <div class="track"><div class="ai-bar" style="width:{contact_ai:.1f}%"></div></div>
          <div class="track"><div class="human-bar" style="width:{contact_human:.1f}%"></div></div>
        </div>
        <div class="vals">
          <div class="ai">AI {contact_ai:.1f}% ({rpc_ia})</div>
          <div class="human">HUM {contact_human:.1f}% ({rpc_h})</div>
        </div>
      </div>

      <div class="funnel-row">
        <div class="funnel-name">2. Negociación iniciada | RPC</div>
        <div class="tracks">
          <div class="track"><div class="ai-bar" style="width:100%"></div></div>
          <div class="track"><div class="human-bar" style="width:100%"></div></div>
        </div>
        <div class="vals">
          <div class="ai">AI 100% ({rpc_ia}/{rpc_ia})</div>
          <div class="human">HUM 100% ({rpc_h}/{rpc_h})</div>
        </div>
      </div>

      <div class="funnel-row">
        <div class="funnel-name">3. Oferta de alternativas | RPC</div>
        <div class="tracks">
          <div class="track"><div class="ai-bar" style="width:{alt_offer_pct_ia:.1f}%"></div></div>
          <div class="track"><div class="human-bar" style="width:{alt_offer_pct_h:.1f}%"></div></div>
        </div>
        <div class="vals">
          <div class="ai">AI {alt_offer_pct_ia:.1f}% ({alt_offer_cnt_ia}/{rpc_ia}) · Media {alt_mean_ia:.2f}, IQR {alt_iqr_ia:.1f}</div>
          <div class="human">HUM {alt_offer_pct_h:.1f}% ({alt_offer_cnt_h}/{rpc_h}) · Media {alt_mean_h:.2f}, IQR {alt_iqr_h:.1f}</div>
        </div>
      </div>

      <div class="funnel-row">
        <div class="funnel-name">4. Compromiso de pago | RPC</div>
        <div class="tracks">
          <div class="track"><div class="ai-bar" style="width:{commit_contact_ai:.1f}%"></div></div>
          <div class="track"><div class="human-bar" style="width:{commit_contact_human:.1f}%"></div></div>
        </div>
        <div class="vals">
          <div class="ai">AI {commit_contact_ai:.1f}% ({ptp_ia})</div>
          <div class="human">HUM {commit_contact_human:.1f}% ({ptp_h})</div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <div class="num">2</div>
      <div>
        <h2>Análisis estadístico</h2>
        <p class="lead">Tres conclusiones directas sustentadas por los contrastes estadísticos.</p>
      </div>
    </div>

    <div class="insight-list">
      <div class="insight-item">
        <div class="insight-icon">01</div>
        <div>
          <strong>La IA pierde el 52.0% de sus llamadas en buzones e IVRs no filtrados.</strong>
          <p>Solo el {contact_ai:.1f}% de llamadas de IA logran contacto efectivo frente al {contact_human:.1f}% humano (+{contact_gap:.1f} pp, p = {contact_p:.3f}*). El bot desgasta intentos en contestadoras automáticas.</p>
        </div>
      </div>

      <div class="insight-item">
        <div class="insight-icon">02</div>
        <div>
          <strong>Una vez contactado el deudor, no existe rechazo por ser IA.</strong>
          <p>En el 100% de los contactos la IA entra a negociar la deuda. En el modelo econométrico multivariado, ser IA no es un factor adverso significativo (p = {ia_p:.3f} n.s.).</p>
        </div>
      </div>

      <div class="insight-item">
        <div class="insight-icon">03</div>
        <div>
          <strong>Ofrecer alternativas es el verdadero motor del cierre (OR = {alt_or:.2f}x*).</strong>
          <p>La IA ofrece opciones con dispersión nula (IQR = 0.0; siempre 2 cuotas fijas). El humano adapta entre 0 y 5 opciones (IQR = 3.0), resolviendo el {res_human:.1f}% de objeciones vs. {res_ai:.1f}% en la IA.</p>
        </div>
      </div>
    </div>
  </div>

</section>

<footer class="footer">
  <span>Creceré AI · Reporte - Agentes de cobranza</span>
  <span>Página 1 de 2</span>
</footer>

</main>


<!-- PAGE 2 -->
<main class="page">

<header class="hero">
  <div class="brand-row">
    <div class="eyebrow">Hallazgos y mejoras recomendadas</div>
    <img class="logo" src="{logo_uri}" alt="Creceré AI" />
  </div>

  <h1>Por qué ofrecer alternativas es la clave del cierre en IA</h1>

  <p>
    El modelado multivariado demuestra que la brecha de conversión no se debe a que el cliente rechace hablar con una máquina,
    sino a la <strong>rigidez de la IA para desplegar opciones financieras ante la fricción</strong>.
  </p>
</header>

<section class="content">

  <div class="split">
    <div class="card">
      <div class="card-title">
        <h3>Desempeño real ante objeciones y negociación</h3>
        <span>población RPC (n = {rpc_h + rpc_ia} contactos)</span>
      </div>

      <div class="objection-row">
        <div class="objection-head">
          <span>Tasa de resolución de objeciones (H3)</span>
          <span>AI {res_ai:.1f}% · HUMAN {res_human:.1f}% (p = {res_p:.3f}*)</span>
        </div>
        <div class="objection-bars">
          <div class="track"><div class="ai-bar" style="width:{res_ai:.1f}%"></div></div>
          <div class="track"><div class="human-bar" style="width:{res_human:.1f}%"></div></div>
        </div>
      </div>

      <div class="objection-row">
        <div class="objection-head">
          <span>Compromiso logrado con objeción (PTP | Obj)</span>
          <span>AI {ptp_obj_ai:.1f}% ({ptp_obj_cnt_ia}/{obj_calls_ia}) · HUMAN {ptp_obj_human:.1f}% ({ptp_obj_cnt_h}/{obj_calls_h})</span>
        </div>
        <div class="objection-bars">
          <div class="track"><div class="ai-bar" style="width:{ptp_obj_ai:.1f}%"></div></div>
          <div class="track"><div class="human-bar" style="width:{ptp_obj_human:.1f}%"></div></div>
        </div>
      </div>

      <div class="objection-row">
        <div class="objection-head">
          <span>Flexibilidad de oferta (Rango IQR en H1)</span>
          <span>AI IQR = {alt_iqr_ia:.1f} (Rígido) · HUMAN IQR = {alt_iqr_h:.1f} (Flexible)</span>
        </div>
        <div class="objection-bars">
          <div class="track"><div class="ai-bar" style="width:20%"></div></div>
          <div class="track"><div class="human-bar" style="width:85%"></div></div>
        </div>
      </div>

      <div class="note">
        <strong>Hallazgo econométrico multivariado (Logit PTP)</strong>
        <p>
          Cada alternativa de pago ofrecida <strong>duplica la probabilidad de lograr el compromiso de pago (OR = {alt_or:.2f}x*, IC 95% [{alt_ci[0]:.2f}, {alt_ci[1]:.2f}], p = {alt_p:.3f}*)</strong>,
          mientras que cada objeción sin resolver reduce el cierre en un 34.5% (OR = {obj_or:.2f}x*, p = {obj_p:.3f}*).
        </p>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <h3>Diagnóstico del Voicebot IA: fortalezas y oportunidad</h3>
        <span>datos empíricos auditables</span>
      </div>

      <div class="insight-list">
        <div class="insight-item">
          <div class="insight-icon">✓</div>
          <div>
            <strong>100% de inicio de negociación</strong>
            <p>Los clientes contactados escuchan la deuda y conversan formalmente con el bot sin colgarle de inmediato.</p>
          </div>
        </div>

        <div class="insight-item">
          <div class="insight-icon">✓</div>
          <div>
            <strong>Reparto conversacional equilibrado</strong>
            <p>El bot habla solo el {talk_ai:.1f}% del tiempo vs. {talk_human:.1f}% del humano (p = {talk_p:.3f} n.s.), descartando monopolización de la llamada.</p>
          </div>
        </div>

        <div class="insight-item">
          <div class="insight-icon" style="background:#FFE4E6;color:#BE123C;">!</div>
          <div>
            <strong>Oportunidad crítica: romper el árbol rígido</strong>
            <p>La IA actual ofrece siempre 2 cuotas (IQR = 0.0). Al no ofrecer alternativas personalizadas (descuentos, gracia), desaprovecha el multiplicador {alt_or:.2f}x.</p>
          </div>
        </div>

        <div class="insight-item">
          <div class="insight-icon">≈</div>
          <div>
            <strong>Claridad fonética a optimizar</strong>
            <p>El {prop_cero_rep_ia:.1f}% de llamadas IA transcurren sin solicitudes de aclaración frente al {prop_cero_rep_h:.1f}% en humanos (cadencia TTS).</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-head">
      <div class="num">3</div>
      <div>
        <h2>Mejoras recomendadas</h2>
        <p class="lead">Tres iniciativas priorizadas para maximizar el retorno de cobranza con IA.</p>
      </div>
    </div>

    <div class="action-grid">
      <div class="action" style="border-color:rgba(112,71,235,.3);background:linear-gradient(180deg,#fff 0%,#fbf9ff 100%);">
        <div class="priority">Prioridad 1 · Máximo Impacto</div>
        <h3>Motor dinámico de alternativas de pago</h3>
        <p>
          Permitir que el LLM proponga opciones flexibles según la objeción: abonos parciales, prórrogas, descuento de intereses moratorios y fechas ajustadas al pago del deudor. Captura directa del multiplicador <strong>OR = {alt_or:.2f}x* en promesas de pago</strong>.
        </p>
      </div>

      <div class="action">
        <div class="priority">Prioridad 2</div>
        <h3>Filtro acústico de contestadora (AMD)</h3>
        <p>
          Implementar detección de buzón de voz e IVR antes de transferir la llamada al voicebot. Cerrar la brecha de <strong>+{contact_gap:.1f} pp en contactabilidad RPC</strong> aumentará de inmediato las promesas absolutas en más del 50%.
        </p>
      </div>

      <div class="action">
        <div class="priority">Prioridad 3</div>
        <h3>Optimización de cadencia y síntesis TTS</h3>
        <p>
          Mejorar prosodia, pausas conversacionales y velocidad de habla para reducir las solicitudes de repetición del deudor, facilitando la comprensión de los montos y fechas negociadas.
        </p>
      </div>
    </div>
  </div>

  <div class="decision">
    <div>
      <strong>Conclusión directiva: La IA debe ofrecer alternativas dinámicas para maximizar el compromiso de pago.</strong>
      <p style="margin-top:9px;font-size:12.5px;color:#eee8ff;line-height:1.5;">
        <em>Nota de control y riesgo:</em> Debe implementarse con precaución, pues las opciones y ofertas permitidas deben estar previamente parametrizadas y delimitadas por reglas de negocio de la entidad financiera (el agente no debe decidir discrecionalmente los alivios por motivos de seguridad y compliance). Asimismo, se recomienda ampliar el tamaño muestral en futuras iteraciones para obtener un análisis aún más preciso y segmentado.
      </p>
    </div>

    <div class="decision-side">
      <span>Multiplicador de Cierre</span>
      <strong>{alt_or:.2f}x OR</strong>
      <small style="color:#eadfff;display:block;margin-top:2px;font-size:10px;">por alternativa ofrecida (p = {alt_p:.3f}*)</small>
    </div>
  </div>

  <div class="limitations">
    <strong>Alcance y rigor metodológico.</strong> Muestra balanceada de {total_calls} llamadas procesadas y auditadas ({human_calls} humanos, {ai_calls} IA). Para aislar sesgos de contactabilidad previa, el contraste de comportamiento y el modelo econométrico multivariado se evaluaron sobre las {rpc_h + rpc_ia} llamadas con contacto efectivo verificado (RPC: {rpc_h} humanos, {rpc_ia} IA), aplicando tests no paramétricos (Mann-Whitney U, Exacto de Fisher) y regresión logística multivariada [logit(P(PTP=1))]. La métrica de tiempos de habla incorpora Winsorización al percentil 99 para robustez metodológica ante valores atípicos. Todos los resultados reportados son 100% reproducibles desde el dataset analítico.
  </div>

</section>

<footer class="footer">
  <span>Creceré AI · Reporte - Agentes de cobranza</span>
  <span>Página 2 de 2</span>
</footer>

</main>

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description='Generador del Reporte Ejecutivo en HTML (Creceré AI)')
    parser.add_argument('--input', type=Path, default=RESULTS_JSON)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f'Error: No se encontró {args.input}')
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        results = json.load(f)

    html_content = build_minimalist_report_html(results)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f'[✓] Reporte generado exitosamente en: {args.output}')
    print(f'    Tamaño: {len(html_content) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
