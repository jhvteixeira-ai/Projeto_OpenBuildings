"""
Geração de Gráficos para o Banner de Iniciação Científica (300 DPI)
Sem nome do bairro (confidencialidade) — Formato de Barras Empilhadas Incrementais (45m + Δ30m)
e Gráfico Proporcional em relação à PNT Total da Região (1.127,70 MWh/mês).
"""

import io
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import shutil
import os


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

# Configuração estética para pôster científico
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Segoe UI"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

# Parâmetros da simulação
N_SUSPEITAS_30M = 1447
N_SUSPEITAS_45M = 776
TARIFA_KWH = 0.95
PNT_REGIAO_KWH = 1_127_700.0  # 1.127,70 MWh/mês

# Perfis renumerados P1 a P5
perfis_labels = [
    "P1 Verão\n(460 kWh)",
    "P1 Inverno\n(190 kWh)",
    "P2 Equip.\n(105,8 kWh)",
    "P3 Sudeste\n(174,1 kWh)",
    "P4 Baixa Renda\n(153,7 kWh)",
    "P5 Mediana Local\n(168,0 kWh)",
]

consumo_unitario = np.array([460.0, 190.0, 105.77, round(2088.7 / 12, 2), round(1843.9 / 12, 2), 168.0])

# Grandezas
kwh_30m = N_SUSPEITAS_30M * consumo_unitario
kwh_45m = N_SUSPEITAS_45M * consumo_unitario

rs_mes_30m = kwh_30m * TARIFA_KWH
rs_mes_45m = kwh_45m * TARIFA_KWH

rs_ano_30m = rs_mes_30m * 12
rs_ano_45m = rs_mes_45m * 12

pct_pnt_30m = (kwh_30m / PNT_REGIAO_KWH) * 100.0
pct_pnt_45m = (kwh_45m / PNT_REGIAO_KWH) * 100.0


def fmt_br_int(val):
    return f"{int(round(val)):,}".replace(",", ".")


def fmt_br_mil(val, prefixo="R$ "):
    num = f"{val/1000:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{prefixo}{num} mil"


def fmt_br_anual(val, prefixo="R$ "):
    if val >= 1_000_000:
        num = f"{val/1_000_000:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{prefixo}{num} mi"
    num = f"{val/1_000:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{prefixo}{num} mil"


def criar_grafico_empilhado_incremental(
    valores_30m,
    valores_45m,
    titulo,
    subtitulo,
    ylabel,
    nome_arquivo,
    formatador_valor,
    escala_eixo=1.0,
    formato_eixo="mil",
    mostrar_pct_pnt=False,
):
    """
    Gráfico de Barras Empilhadas Incrementais (Nested Stacked Bar Chart):
    - Base sólida: Cenário de 45 m (776 edificações que permanecem suspeitas até 45 m)
    - Topo empilhado: Acréscimo do cenário de 30 m (+671 edificações -> totalizando 1.447 no topo)
    Evita dupla contagem (não soma 30m + 45m) e mostra ambos os cenários em uma única barra por perfil!
    """
    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=300)

    x = np.arange(len(perfis_labels))
    largura = 0.54

    base_45m = valores_45m / escala_eixo
    delta_30m = (valores_30m - valores_45m) / escala_eixo
    topo_30m = valores_30m / escala_eixo

    cor_45m = "#1b6ca8"    # Azul profundo (Núcleo 45m)
    cor_delta = "#68b0ab"  # Verde-água (Faixa adicional até 30m)

    # Barra base (45m)
    ax.bar(
        x,
        base_45m,
        largura,
        label=f"Patamar Cabo 45 m ({N_SUSPEITAS_45M} suspeitas)",
        color=cor_45m,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    # Barra incremental empilhada (diferença de 45m até 30m = Total 30m)
    ax.bar(
        x,
        delta_30m,
        largura,
        bottom=base_45m,
        label=f"Acréscimo no Cabo 30 m (+{N_SUSPEITAS_30M - N_SUSPEITAS_45M} -> Total: {N_SUSPEITAS_30M} suspeitas)",
        color=cor_delta,
        edgecolor="white",
        linewidth=1.0,
        hatch="//",
        alpha=0.92,
        zorder=3,
    )

    # Títulos sem o nome do bairro
    fig.suptitle(titulo, fontsize=14, fontweight="bold", color="#1a1a1a", y=0.97)
    ax.set_title(subtitulo, fontsize=10.5, fontweight="normal", color="#555555", pad=12)

    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", color="#333333", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(perfis_labels, fontsize=10, fontweight="bold", color="#222222")

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(topo_30m) * 1.22)

    # Formatação do eixo Y
    if formato_eixo == "mil":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,} mil".replace(",", ".")))
    elif formato_eixo == "rs_mil":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"R$ {int(v):,} mil".replace(",", ".")))
    elif formato_eixo == "rs_milhao":
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, p: f"R$ {v:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", "."))
        )

    # Rótulos no topo (Total 30m) e no centro da base (Total 45m)
    for i in range(len(x)):
        # Topo (30m)
        txt_30 = f"30m: {formatador_valor(valores_30m[i])}"
        if mostrar_pct_pnt:
            txt_30 += f"\n({pct_pnt_30m[i]:.1f}% da PNT)".replace(".", ",")

        ax.annotate(
            txt_30,
            xy=(x[i], topo_30m[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#113f67",
            linespacing=1.1,
        )

        # Interna na base (45m)
        txt_45 = f"45m:\n{formatador_valor(valores_45m[i])}"
        ax.text(
            x[i],
            base_45m[i] * 0.48,
            txt_45,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="white",
            linespacing=1.05,
            zorder=4,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        fontsize=9.5,
        bbox_to_anchor=(0.98, 0.97),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    salvar_png_seguro(fig, nome_arquivo)
    plt.close(fig)
    print(f"[OK] Gráfico empilhado salvo: {nome_arquivo}")


def criar_grafico_proporcional_pnt(nome_arquivo):
    """
    Gráfico de Barras Empilhadas Proporcionais (100% = PNT Total da Região: 1.127,7 MWh/mês)
    Mostra a fatia explicada pelas Edificações Sem Medidor (M1) vs. Demais PNT (Fraudes em UCs com medidor, etc.)
    """
    fig, ax = plt.subplots(figsize=(10.2, 5.5), dpi=300)

    x = np.arange(len(perfis_labels))
    largura = 0.54

    fatia_45m = pct_pnt_45m
    fatia_delta_30m = pct_pnt_30m - pct_pnt_45m
    fatia_outras_pnt = 100.0 - pct_pnt_30m

    ax.bar(
        x,
        fatia_45m,
        largura,
        label="Sem Medidor — Patamar Cabo 45 m (776 edificações)",
        color="#1b6ca8",
        edgecolor="white",
        zorder=3,
    )
    ax.bar(
        x,
        fatia_delta_30m,
        largura,
        bottom=fatia_45m,
        label="Sem Medidor — Faixa Adicional até Cabo 30 m (+671 -> 1.447 edificações)",
        color="#68b0ab",
        edgecolor="white",
        hatch="//",
        zorder=3,
    )
    ax.bar(
        x,
        fatia_outras_pnt,
        largura,
        bottom=pct_pnt_30m,
        label="Demais PNT da Região (Fraudes/Desvios em UCs com Medidor, Comércios, etc.)",
        color="#e0e4e8",
        edgecolor="white",
        zorder=3,
    )

    fig.suptitle(
        "Participação Proporcional das Ligações Sem Medidor na PNT Total da Região (100% = 1.127,7 MWh/mês)",
        fontsize=12.5,
        fontweight="bold",
        color="#1a1a1a",
        y=0.97,
    )
    ax.set_title(
        "Decomposição das Perdas Não Técnicas por Perfil de Consumo (P1 a P5) — Área de Estudo (Raio 3 km)",
        fontsize=10,
        color="#555555",
        pad=12,
    )

    ax.set_ylabel("Proporção da PNT Total da Região (%)", fontsize=11, fontweight="bold", color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(perfis_labels, fontsize=10, fontweight="bold", color="#222222")
    ax.set_ylim(0, 114)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#bbbbbb", zorder=0)

    for i in range(len(x)):
        # Rótulo 45m
        ax.text(
            x[i],
            fatia_45m[i] * 0.5,
            f"{fatia_45m[i]:.1f}%".replace(".", ","),
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        # Rótulo Total 30m na divisória
        ax.text(
            x[i],
            pct_pnt_30m[i] + 2.2,
            f"Total 30m: {pct_pnt_30m[i]:.1f}%".replace(".", ","),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#113f67",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#1b6ca8", alpha=0.9, lw=0.7),
            zorder=5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    salvar_png_seguro(fig, nome_arquivo)
    plt.close(fig)
    print(f"[OK] Gráfico proporcional salvo: {nome_arquivo}")


# 1. Energia Mensal (kWh/mês) com % da PNT
criar_grafico_empilhado_incremental(
    valores_30m=kwh_30m,
    valores_45m=kwh_45m,
    titulo="Estimativa de Energia Perdida Mensal por Perfil de Consumo (P1 a P5)",
    subtitulo="Barras Empilhadas Incrementais: Patamar Cabo 45 m + Acréscimo até Cabo 30 m (Raio de 3 km)",
    ylabel="Energia Perdida (mil kWh / mês)",
    nome_arquivo="grafico_empilhado_1_kwh_mensal.png",
    formatador_valor=lambda v: f"{fmt_br_int(v)} kWh",
    escala_eixo=1_000.0,
    formato_eixo="mil",
    mostrar_pct_pnt=True,
)

# 2. Financeiro Mensal (R$/mês)
criar_grafico_empilhado_incremental(
    valores_30m=rs_mes_30m,
    valores_45m=rs_mes_45m,
    titulo="Impacto Financeiro Mensal das Perdas Não Técnicas por Perfil (P1 a P5)",
    subtitulo="Tarifa Residencial B1 Convencional (R$ 0,95/kWh) — Patamar Cabo 45 m + Acréscimo Cabo 30 m",
    ylabel="Perda Financeira Mensal (R$ mil / mês)",
    nome_arquivo="grafico_empilhado_2_reais_mensal.png",
    formatador_valor=lambda v: fmt_br_mil(v, prefixo="R$ "),
    escala_eixo=1_000.0,
    formato_eixo="rs_mil",
    mostrar_pct_pnt=False,
)

# 3. Financeiro Anual (R$/ano)
criar_grafico_empilhado_incremental(
    valores_30m=rs_ano_30m,
    valores_45m=rs_ano_45m,
    titulo="Impacto Financeiro Anual Estimado por Perfil de Consumo (P1 a P5)",
    subtitulo="Projeção Anualizada (12 meses, R$ 0,95/kWh) — Patamar Cabo 45 m + Acréscimo Cabo 30 m",
    ylabel="Perda Financeira Anual (R$ milhões / ano)",
    nome_arquivo="grafico_empilhado_3_reais_anual.png",
    formatador_valor=lambda v: fmt_br_anual(v, prefixo="R$ "),
    escala_eixo=1_000_000.0,
    formato_eixo="rs_milhao",
    mostrar_pct_pnt=False,
)

# 4. Proporcional 100% da PNT da Região
criar_grafico_proporcional_pnt("grafico_empilhado_4_proporcional_pnt.png")
print("[OK] Todos os gráficos empilhados foram atualizados sem faixa amarela!")
