"""Estilo «obsidiana» de JaaZ sobre Streamlit: tarjetas de indicadores, encabezados, tablero de
correlación y algunos ajustes a los componentes de Streamlit (botón principal en píldora blanca,
títulos de peso 400, metadatos en monoespaciada mayúscula). Los colores salen de .streamlit/config.toml.
"""
import html

import streamlit as st

PALETA = ["#c98500", "#3987e5", "#d95926", "#199e70"]  # validada para daltonismo sobre #101010

CSS = """
<style>
:root {
  --ink: #f3f3f3; --ink-2: #c1c1c1; --muted: #9c9c9c; --faint: #6f6f6f; --line: #212121; --line-2: #2e2e2e;
  --line-3: #474747; --surface-3: #1a1a1a; --gold: #b3a78f; --pulse: #98ff38;
  --s1: #c98500; --s2: #3987e5; --s3: #d95926; --s4: #199e70;
  --ok: #7fd9a8; --warn: #e3b85c; --danger: #f07a7a; --info: #9ab8e6;
  --mono: "JetBrains Mono", "Cascadia Mono", ui-monospace, Consolas, monospace;
}
/* títulos livianos, como la versión de escritorio */
h1, h2, h3, h4 { font-weight: 400 !important; letter-spacing: -.015em; }
[data-testid="stSidebarNav"] span { font-size: 14px; }
[data-testid="stSidebarNavSeparator"] { border-color: var(--line); }
/* botón principal: píldora blanca en mayúscula */
[data-testid="stBaseButton-primary"] { background: var(--ink) !important; color: #101010 !important; border: 1px solid var(--ink) !important;
  border-radius: 9999px !important; text-transform: uppercase; letter-spacing: .05em; font-size: 13px !important; }
[data-testid="stBaseButton-primary"] p { color: #101010 !important; }
[data-testid="stBaseButton-primary"]:hover { background: #e2e2e2 !important; border-color: #e2e2e2 !important; }

/* etiquetas elegidas en los selectores múltiples: gris con texto claro (no el color primario blanco) */
[data-baseweb="tag"] { background: #262626 !important; border: 1px solid #3a3a3a !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: #f3f3f3 !important; fill: #f3f3f3 !important; }

/* piezas propias */
.jz-eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.jz-titulo { font-size: 44px; font-weight: 400; letter-spacing: -.02em; line-height: 1.05; color: var(--ink); margin: 0 0 12px; }
.jz-sub { font-size: 16px; color: var(--muted); line-height: 1.55; max-width: 780px; margin-bottom: 8px; }
.jz-meta { font-family: var(--mono); font-size: 10.5px; letter-spacing: .03em; text-transform: uppercase; color: var(--muted); }
.jz-hero { padding: 24px 0 8px; }
.jz-hero .jz-display { font-size: clamp(38px, 5vw, 68px); font-weight: 400; letter-spacing: -.025em; line-height: 1.02; color: var(--ink); max-width: 16ch; }
.jz-hero .jz-display em { font-style: normal; color: var(--muted); }
.jz-hero .jz-lead { font-size: 18px; line-height: 1.5; color: var(--muted); max-width: 680px; margin-top: 22px; }
.jz-pastilla-vivo { display: inline-flex; align-items: center; gap: 10px; font-family: var(--mono); font-size: 11px; text-transform: uppercase;
  letter-spacing: .03em; color: var(--muted); border: 1px solid var(--line); background: var(--surface-3); border-radius: 4px; padding: 6px 12px; margin-bottom: 26px; }
.jz-pastilla-vivo i { width: 6px; height: 6px; border-radius: 50%; background: var(--pulse); display: inline-block; }
.jz-pasos { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.jz-paso { padding: 24px 22px 26px; border-right: 1px solid var(--line); }
.jz-paso:last-child { border-right: 0; }
.jz-paso .n { font-family: var(--mono); font-size: 11px; color: var(--gold); margin-bottom: 30px; }
.jz-paso b { display: block; font-weight: 400; font-size: 20px; color: var(--ink); letter-spacing: -.01em; }
.jz-paso p { font-size: 13.5px; color: var(--muted); line-height: 1.55; margin-top: 8px; }

/* indicadores */
.jz-kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line);
  border-radius: 8px; overflow: hidden; margin: 6px 0 14px; }
.jz-kpi { background: #101010; padding: 18px 20px; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.jz-kpi.destacado { grid-column: span 2; }
.jz-kpi .et { font-family: var(--mono); font-size: 10.5px; letter-spacing: .03em; text-transform: uppercase; color: var(--muted);
  display: flex; justify-content: space-between; gap: 8px; align-items: center; }
.jz-kpi .val { font-size: 32px; letter-spacing: -.02em; line-height: 1.05; color: var(--ink); margin-top: 4px; }
.jz-kpi.destacado .val { font-size: 58px; }
.jz-kpi .ee { font-size: 15px; color: var(--muted); letter-spacing: 0; margin-left: 4px; }
.jz-kpi .ic { font-family: var(--mono); font-size: 11px; color: var(--faint); }
.jz-kpi .desc { font-size: 13px; color: var(--muted); line-height: 1.5; margin-top: 2px; }
.jz-medidor { position: relative; height: 1px; background: var(--line-2); margin: 12px 0 4px; }
.jz-medidor i { position: absolute; top: -1px; height: 3px; background: var(--ink); }
.jz-medidor span { position: absolute; top: -3px; height: 7px; border-left: 1px solid var(--line-3); border-right: 1px solid var(--line-3); }
.jz-escala { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10px; color: var(--faint); }
.jz-badge { display: inline-flex; align-items: center; gap: 6px; height: 21px; padding: 0 8px; border-radius: 4px; font-family: var(--mono);
  font-size: 10px; letter-spacing: .02em; text-transform: uppercase; white-space: nowrap; background: var(--surface-3); border: 1px solid var(--line); color: var(--muted); }
.jz-badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
.jz-badge.alta::before { background: var(--ok); } .jz-badge.media::before { background: var(--info); }
.jz-badge.baja::before { background: var(--warn); } .jz-badge.error::before { background: var(--danger); }

/* tablero de correlación */
.jz-panel { border: 1px solid var(--line); border-radius: 8px; padding: 20px 22px 22px; margin-bottom: 14px; }
.jz-panel h3 { font-size: 21px !important; margin: 6px 0 0 !important; padding: 0 !important; color: var(--ink); }
.jz-cab { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 14px; }
.jz-hero-cuerpo { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px 32px; flex-wrap: wrap; }
.jz-valor-grande { font-size: 72px; letter-spacing: -.02em; line-height: .95; color: var(--ink); }
.jz-valor-grande .ee { font-size: 22px; color: var(--muted); letter-spacing: 0; margin-left: 6px; }
.jz-pill { display: inline-block; margin-top: 12px; font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .03em;
  padding: 3px 9px; border-radius: 9999px; border: 1px solid var(--line-2); color: var(--muted); }
.jz-pill.si { border-color: var(--ink); color: var(--ink); }
.jz-datos { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line);
  border-radius: 6px; overflow: hidden; flex: 1 1 380px; max-width: 640px; padding: 0 !important; margin: 0; }
.jz-datos dd { margin-left: 0 !important; }
.jz-datos div { background: #101010; padding: 11px 15px 13px; }
.jz-datos dt { font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .03em; color: var(--faint); }
.jz-datos dd { margin: 5px 0 0; font-size: 22px; letter-spacing: -.01em; color: var(--ink); font-variant-numeric: tabular-nums; }
.jz-datos.chico dd { font-size: 14px; }
.jz-metricas { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 20px; }
.jz-metrica .val { font-size: 34px; letter-spacing: -.02em; line-height: 1; color: var(--ink); margin: 8px 0 6px; }
.jz-metrica .val .ee { font-size: 13px; color: var(--muted); margin-left: 4px; letter-spacing: 0; }
.jz-metrica .ic { font-family: var(--mono); font-size: 11px; color: var(--faint); }
.jz-metrica .nota { font-size: 12.5px; color: var(--muted); line-height: 1.5; }
.jz-barra { position: relative; height: 6px; border-radius: 3px; background: var(--surface-3); margin-top: 10px; }
.jz-barra i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--ink); }
.jz-barra span { position: absolute; top: -4px; bottom: -4px; border-left: 1px solid var(--line-3); border-right: 1px solid var(--line-3); }
.jz-bloque { border-top: 1px solid var(--line); margin-top: 16px; padding-top: 14px; }
.jz-particion { display: flex; gap: 2px; height: 14px; border-radius: 3px; overflow: hidden; margin: 10px 0 8px; }
.jz-leyenda { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 12px; color: var(--muted); }
.jz-leyenda span { display: inline-flex; align-items: center; gap: 7px; }
.jz-leyenda i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.jz-formula { display: grid; grid-template-columns: minmax(160px, 240px) 1fr; gap: 14px; padding: 9px 0; border-bottom: 1px solid var(--line); font-size: 13px; }
.jz-formula b { font-weight: 400; color: var(--muted); }
.jz-formula code { color: var(--ink); background: transparent; white-space: pre-wrap; }
.jz-pie { display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap; padding-top: 22px; margin-top: 30px; border-top: 1px solid var(--line);
  font-family: var(--mono); font-size: 11px; color: var(--faint); text-transform: uppercase; letter-spacing: .02em; }
.jz-pie .marca { text-transform: none; color: var(--ink-2); }
</style>
"""


def aplicar():
    st.html(CSS)


def esc(texto):
    return html.escape("" if texto is None else str(texto))


def encabezado(eyebrow, titulo, sub=None):
    st.html(f'<div class="jz-eyebrow">{esc(eyebrow)}</div><div class="jz-titulo">{esc(titulo)}</div>'
            + (f'<div class="jz-sub">{esc(sub)}</div>' if sub else ""))
