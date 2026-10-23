"""
1. Limpa os mapas HTML (remove títulos, caixas flutuantes e controles para print limpo no pôster).
2. Gera gráficos otimizados especificamente para a coluna direita do Pôster ConICC UNICAMP,
   com fontes grandes (legíveis a 1,5m de distância) e unificando Energia + Impacto Financeiro
   para abrir espaço para o Mapa e para a Conclusão.
"""

import io
import re
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def salvar_png_seguro(fig, nome_arq):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    data = buf.getvalue()
    artifact_dir = r"C:\Users\Vianini\.gemini\antigravity\brain\86262fa0-0141-495b-bd58-230a38514cd5"
    if os.path.exists(artifact_dir):
        with open(os.path.join(artifact_dir, nome_arq), "wb") as f:
            f.write(data)
    try:
        if os.path.exists(nome_arq):
            os.remove(nome_arq)
        with open(nome_arq, "wb") as f:
            f.write(data)
    except OSError:
        shutil.copyfile(os.path.join(artifact_dir, nome_arq), nome_arq)

# =============================================================================
# PARTE 1: LIMPAR OS MAPAS HTML (SEM ERRO 403 / SEM "API KEY REQUIRED" E SEM CLUSTERS NO ZOOM)
# =============================================================================
for arq_html in ["metodologia1_cabo30m.html", "metodologia1_cabo45m.html"]:
    if os.path.exists(arq_html):
        with open(arq_html, "r", encoding="utf-8") as f:
            html_orig = f.read()

        # 1. No zoom distante (14-16), manter agrupado para não poluir; no zoom de quarteirão (>=17),
        #    desagrupar 100% das bolinhas verdes ("2", "3") automaticamente!
        html_base = re.sub(
            r"L\.markerClusterGroup\(\s*\{[^}]*\}\s*\)",
            'L.markerClusterGroup({"disableClusteringAtZoom": 17, "maxClusterRadius": 30})',
            html_orig,
        )

        # 2. Permitir zoom próximo no Satélite Esri (até nível 22) sem sumir a imagem
        html_base = html_base.replace(
            '"maxZoom": 18,\n  "maxNativeZoom": 18,',
            '"maxZoom": 22,\n  "maxNativeZoom": 18,',
        )

        css_limpo = """
        <style>
            .leaflet-container { background: #f3f4f6 !important; }
            h3 { display: none !important; }
            .leaflet-control-layers { display: none !important; }
            .leaflet-control-zoom { display: none !important; }
            .leaflet-control-attribution { display: none !important; }
            .awesome-marker, .awesome-marker-shadow { display: none !important; }
            div[style*="position:fixed"] { display: none !important; }
        </style>
        """

        # VERSÃO 1: FUNDO CLARO LIMPO (igual ao seu 1º print dos quarteirões, mas SEM "API KEY REQUIRED" e sem bolinhas "2"/"3")
        html_claro = re.sub(
            r'var tile_layer_[0-9a-f]+ = L\.tileLayer\(.*?tile_layer_[0-9a-f]+\.addTo\(map_[0-9a-f]+\);',
            "",
            html_base,
            flags=re.DOTALL,
        )
        html_claro = html_claro.replace("</head>", css_limpo + "\n</head>")
        arq_saida_claro = arq_html.replace(".html", "_apenas_mapa.html")
        with open(arq_saida_claro, "w", encoding="utf-8") as f:
            f.write(html_claro)
        print(f"[OK] Mapa Fundo Claro Limpo salvo em: {arq_saida_claro}")

        # VERSÃO 2: FUNDO SATÉLITE PURO (remove a 2ª camada CartoDB/OSM que dava 403 Access Blocked por cima do satélite)
        html_sat = re.sub(
            r'var tile_layer_[0-9a-f]+ = L\.tileLayer\(\s*"https://\{s\}\.basemaps\.cartocdn\.com/light_all/.*?tile_layer_[0-9a-f]+\.addTo\(map_[0-9a-f]+\);',
            "",
            html_base,
            flags=re.DOTALL,
        )
        html_sat = html_sat.replace("</head>", css_limpo + "\n</head>")
        arq_saida_sat = arq_html.replace(".html", "_satelite_limpo.html")
        with open(arq_saida_sat, "w", encoding="utf-8") as f:
            f.write(html_sat)
        print(f"[OK] Mapa Fundo Satélite Limpo salvo em: {arq_saida_sat}")


# =============================================================================
# PARTE 2: GRÁFICOS COM FONTES GRANDES PARA O PÔSTER (COLUNA DE RESULTADOS)
# =============================================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Segoe UI"]

N_SUSPEITAS_30M = 1447
N_SUSPEITAS_45M = 776
TARIFA_KWH = 0.95
PNT_REGIAO_KWH = 1_127_700.0

perfis_labels = [
    "P1 Verão\n(460 kWh)",
    "P1 Inverno\n(190 kWh)",
    "P2 Equip.\n(105,8 kWh)",
    "P3 Sudeste\n(174,1 kWh)",
    "P4 Baixa\nRenda\n(153,7 kWh)",
    "P5 Mediana\nLocal\n(168,0 kWh)",
]

consumo_unitario = np.array([460.0, 190.0, 105.77, round(2088.7 / 12, 2), round(1843.9 / 12, 2), 168.0])

kwh_30m = N_SUSPEITAS_30M * consumo_unitario
kwh_45m = N_SUSPEITAS_45M * consumo_unitario

rs_mes_30m = kwh_30m * TARIFA_KWH
rs_mes_45m = kwh_45m * TARIFA_KWH

rs_ano_30m = rs_mes_30m * 12
rs_ano_45m = rs_mes_45m * 12

pct_pnt_30m = (kwh_30m / PNT_REGIAO_KWH) * 100.0


def fmt_mil_kwh(v):
    return f"{v/1000:,.0f} mil kWh".replace(",", ".")


def fmt_rs_mil(v):
    return f"R$ {v/1000:,.0f} mil".replace(",", ".")


def fmt_rs_mi(v):
    if v >= 1_000_000:
        return f"R$ {v/1_000_000:,.2f} mi".replace(".", ",")
    return f"R$ {v/1000:,.0f} mil".replace(",", ".")


# -----------------------------------------------------------------------------
# OPÇÃO 1: GRÁFICO UNIFICADO DUPLO EIXO (Energia Mensal + Valor Mensal e Anual)
# Substitui os 3 gráficos repetidos por 1 gráfico grande e ultra legível!
# -----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9.5, 5.4), dpi=300)

x = np.arange(len(perfis_labels))
largura = 0.58

base_45m = kwh_45m / 1000.0
delta_30m = (kwh_30m - kwh_45m) / 1000.0
topo_30m = kwh_30m / 1000.0

cor_45m = "#1b6ca8"
cor_delta = "#5aa897"

ax1.bar(
    x,
    base_45m,
    largura,
    label="Patamar Cabo 45 m (776 suspeitas)",
    color=cor_45m,
    edgecolor="white",
    linewidth=1.2,
    zorder=3,
)
ax1.bar(
    x,
    delta_30m,
    largura,
    bottom=base_45m,
    label="Acréscimo Cabo 30 m (+671 → 1.447 suspeitas)",
    color=cor_delta,
    edgecolor="white",
    linewidth=1.2,
    hatch="//",
    zorder=3,
)

ax1.set_title(
    "Energia Perdida e Impacto Financeiro por Perfil (Cabo 30 m vs. 45 m)",
    fontsize=15.5,
    fontweight="bold",
    color="#133b5c",
    pad=16,
)
ax1.set_ylabel("Energia Perdida (mil kWh / mês)", fontsize=13, fontweight="bold", color="#1b6ca8")
ax1.set_xticks(x)
ax1.set_xticklabels(perfis_labels, fontsize=11.5, fontweight="bold", color="#1a1a1a")
ax1.tick_params(axis="y", labelsize=11.5, colors="#1b6ca8")
ax1.set_ylim(0, max(topo_30m) * 1.28)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)} mil"))
ax1.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc", zorder=0)

# Segundo Eixo Y (Direita): Perda Financeira Anual (R$ milhões / ano)
ax2 = ax1.twinx()
max_ano_mi = (max(topo_30m) * 1.28 * 1000 * TARIFA_KWH * 12) / 1_000_000.0
ax2.set_ylim(0, max_ano_mi)
ax2.set_ylabel("Perda Financeira Anual (R$ milhões / ano)", fontsize=13, fontweight="bold", color="#8b0000", labelpad=10)
ax2.tick_params(axis="y", labelsize=11.5, colors="#8b0000")
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"R$ {v:,.1f} mi".replace(".", ",")))
ax2.spines["top"].set_visible(False)
ax1.spines["top"].set_visible(False)

# Rótulos grandes e legíveis para o pôster (3 linhas curtas, sem barra dupla de '$' na mesma linha)
for i in range(len(x)):
    lbl_30 = (
        f"30m: {kwh_30m[i]/1000:,.0f} mil kWh\n"
        f"{fmt_rs_mil(rs_mes_30m[i])}/mês\n"
        f"({fmt_rs_mi(rs_ano_30m[i])}/ano)"
    ).replace(",", ".")
    lbl_30 = re.sub(r"R\$ (\d+)\.(\d+) mi", r"R$ \1,\2 mi", lbl_30)

    ax1.annotate(
        lbl_30,
        xy=(x[i], topo_30m[i]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#0f3057",
        linespacing=1.08,
    )

    # Base interna (45m)
    lbl_45 = f"45m:\n{kwh_45m[i]/1000:,.0f} mil kWh\n{fmt_rs_mi(rs_ano_45m[i])}".replace(",", ".")
    lbl_45 = re.sub(r"R\$ (\d+)\.(\d+) mi", r"R$ \1,\2 mi", lbl_45)
    ax1.text(
        x[i],
        base_45m[i] * 0.48,
        lbl_45,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="white",
        linespacing=1.06,
        zorder=4,
    )

ax1.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=11,
    bbox_to_anchor=(0.98, 0.98),
)

plt.tight_layout()
salvar_png_seguro(fig, "grafico_poster_unificado.png")
plt.close(fig)
print("[OK] Gráfico unificado para pôster salvo: grafico_poster_unificado.png")


# -----------------------------------------------------------------------------
# OPÇÃO 2: DOIS GRÁFICOS COMPACTOS COM FONTES GIGANTES (Energia + Financeiro)
# -----------------------------------------------------------------------------
def criar_grafico_poster_compacto(valores_30, valores_45, titulo, ylabel, arq, fn_lbl_30, fn_lbl_45, escala, fmt_y):
    fig, ax = plt.subplots(figsize=(9.2, 4.3), dpi=300)
    x = np.arange(len(perfis_labels))
    largura = 0.56

    b45 = valores_45 / escala
    d30 = (valores_30 - valores_45) / escala
    t30 = valores_30 / escala

    ax.bar(x, b45, largura, label="Cabo 45 m (776 suspeitas)", color="#1b6ca8", edgecolor="white", zorder=3)
    ax.bar(
        x,
        d30,
        largura,
        bottom=b45,
        label="Acréscimo 30 m (+671 → 1.447 suspeitas)",
        color="#5aa897",
        edgecolor="white",
        hatch="//",
        zorder=3,
    )

    ax.set_title(titulo, fontsize=15, fontweight="bold", color="#133b5c", pad=12)
    ax.set_ylabel(ylabel, fontsize=12.5, fontweight="bold", color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(perfis_labels, fontsize=11.5, fontweight="bold", color="#1a1a1a")
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_ylim(0, max(t30) * 1.25)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(len(x)):
        ax.annotate(
            fn_lbl_30(i),
            xy=(x[i], t30[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#0f3057",
            linespacing=1.08,
        )
        ax.text(
            x[i],
            b45[i] * 0.48,
            fn_lbl_45(i),
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            linespacing=1.05,
            zorder=4,
        )

    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=11)
    plt.tight_layout()
    salvar_png_seguro(fig, arq)
    plt.close(fig)
    print(f"[OK] Gráfico compacto salvo: {arq}")


criar_grafico_poster_compacto(
    kwh_30m,
    kwh_45m,
    titulo="1. Energia Perdida Mensal e Participação na PNT (P1 a P5)",
    ylabel="mil kWh / mês",
    arq="grafico_poster_1_energia_grande.png",
    fn_lbl_30=lambda i: f"30m: {kwh_30m[i]/1000:,.0f} mil kWh\n({pct_pnt_30m[i]:.1f}% da PNT)".replace(".", ","),
    fn_lbl_45=lambda i: f"45m:\n{kwh_45m[i]/1000:,.0f} mil",
    escala=1000.0,
    fmt_y=lambda v, p: f"{int(v)} mil",
)

# 2. Gráfico Financeiro Juntando MENSAL + ANUAL (Duplo Eixo Y)
fig2, ax_mes = plt.subplots(figsize=(9.5, 4.4), dpi=300)
x = np.arange(len(perfis_labels))
largura = 0.56

b45_mes = rs_mes_45m / 1000.0
d30_mes = (rs_mes_30m - rs_mes_45m) / 1000.0
t30_mes = rs_mes_30m / 1000.0

ax_mes.bar(x, b45_mes, largura, label="Cabo 45 m (776 suspeitas)", color="#1b6ca8", edgecolor="white", zorder=3)
ax_mes.bar(
    x,
    d30_mes,
    largura,
    bottom=b45_mes,
    label="Acréscimo 30 m (+671 → 1.447 suspeitas)",
    color="#5aa897",
    edgecolor="white",
    hatch="//",
    zorder=3,
)

ax_mes.set_title(
    "2. Impacto Financeiro Mensal e Anual por Perfil (R$ 0,95/kWh)",
    fontsize=15,
    fontweight="bold",
    color="#133b5c",
    pad=12,
)
ax_mes.set_ylabel("Perda Mensal (R$ mil / mês)", fontsize=12, fontweight="bold", color="#1b6ca8")
ax_mes.set_xticks(x)
ax_mes.set_xticklabels(perfis_labels, fontsize=11.5, fontweight="bold", color="#1a1a1a")
ax_mes.tick_params(axis="y", labelsize=11, colors="#1b6ca8")
ax_mes.set_ylim(0, max(t30_mes) * 1.26)
ax_mes.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"R$ {int(v)} mil"))
ax_mes.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc", zorder=0)

# Eixo Direito: Perda Anual (R$ milhões / ano)
ax_ano = ax_mes.twinx()
ax_ano.set_ylim(0, (max(t30_mes) * 1.26 * 12) / 1000.0)
ax_ano.set_ylabel("Perda Anual (R$ milhões / ano)", fontsize=12, fontweight="bold", color="#0f3057", labelpad=10)
ax_ano.tick_params(axis="y", labelsize=11, colors="#0f3057")
ax_ano.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"R$ {v:,.1f} mi".replace(".", ",")))

ax_mes.spines["top"].set_visible(False)
ax_ano.spines["top"].set_visible(False)

for i in range(len(x)):
    lbl_topo = f"30m: {fmt_rs_mil(rs_mes_30m[i])}\n({fmt_rs_mi(rs_ano_30m[i])}/ano)"
    ax_mes.annotate(
        lbl_topo,
        xy=(x[i], t30_mes[i]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.6,
        fontweight="bold",
        color="#0f3057",
        linespacing=1.08,
    )
    lbl_base = f"45m:\n{fmt_rs_mil(rs_mes_45m[i])}\n({fmt_rs_mi(rs_ano_45m[i])})"
    ax_mes.text(
        x[i],
        b45_mes[i] * 0.48,
        lbl_base,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="white",
        linespacing=1.04,
        zorder=4,
    )

ax_mes.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10.5)
plt.tight_layout()
salvar_png_seguro(fig2, "grafico_poster_2_financeiro_grande.png")
plt.close(fig2)
print("[OK] Gráfico financeiro (Mensal + Anual juntos) salvo: grafico_poster_2_financeiro_grande.png")
