"""
PROJETO DE INICIAÇÃO CIENTÍFICA (FEEC / UNICAMP)
Estimativa de Consumo de Unidades Consumidoras Conectadas Ilegalmente
ao Sistema de Distribuição de Energia Elétrica

Script: metodologia1_comparativa.py
Autor: João Henrique Vianini Teixeira
Orientadora: Profa. Dra. Fernanda Caseão Trindade Arioli

Descrição:
  Implementação da Metodologia 1 com simulação comparativa entre os dois
  comprimentos de cabo de serviço padrão (30 metros e 45 metros).
  - Remoção total de estimativas por densidade de potência (1,5 W/m²).
  - Classificação morfológica de galpões/comércios (área >= 400 m² e retangularidade >= 0.85)
    para conferência com a rede de média tensão (Grupo A), sem atribuir perdas fictícias.
  - Detecção de anexos/edículas no mesmo lote (distância <= 6 m e área <= 20 m²),
    evitando falsos positivos de furto conforme observado na literatura (Silva, 2025; ANEEL 1.000/2021).
  - Aplicação dos perfis P2 a P6 devidamente fundamentados:
      * P2: Sazonalidade Verão (460 kWh) e Inverno (190 kWh) [Dissertação]
      * P3: Perfil próprio de equipamentos essenciais (105,77 kWh) [Modelagem bottom-up / PBE INMETRO]
      * P4: Tabela de consumo - Residencial Sudeste (174,06 kWh) [Artigo / EPE / PROCEL]
      * P5: Tabela de consumo - Baixa Renda Sudeste (153,66 kWh) [Artigo / EPE / PROCEL]
      * P6: Mediana empírica local dos consumidores regulares (calculada da base de UCs: ~168 kWh)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, box, MultiPoint
from shapely.ops import unary_union
import folium
from folium.plugins import MarkerCluster
import gc
import numpy as np
import math

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    TEM_SCIPY = True
except ImportError:
    TEM_SCIPY = False

# =============================================================================
# CONFIGURAÇÕES DA PESQUISA
# =============================================================================
ARQ_OPENBUILDINGS   = "Arquivos/OpBuildingCPS.gz"
ARQ_UCS             = "Arquivos/lista_ucs_fernanda_joao.xlsx"
ARQ_SAIDA_EXCEL     = "m1_comparativo_30m_vs_45m.xlsx"
ARQ_MAPA_30M        = "metodologia1_cabo30m.html"
ARQ_MAPA_45M        = "metodologia1_cabo45m.html"

RAIO_M              = 3_000
LON_CENTRO          = -47.1205
LAT_CENTRO          = -23.0384
CONFIANCA_MIN       = 0.75       # Limiar definido pelo autor
AREA_MIN_M2         = 50.0       # Limiar definido pelo autor
BUFFER_MESCLA_M     = 0.3        # Buffer de fusão geométrica
CHUNKSIZE           = 200_000

# Comprimentos de cabo testados (30 m = aéreo padrão CPFL; 45 m = subterrâneo/estendido)
CABOS_SIMULACAO     = [30.0, 45.0]

# --- Parâmetros Morfológicos para Galpão / Comércio (Não Residencial) --------
# Fontes: Barnsley & Barr (1996); Barr & Barnsley (2004); Biljecki et al. (2017)
AREA_MIN_GALPAO     = 400.0      # m²
RETANG_MIN_GALPAO   = 0.85       # Retangularidade (área / retângulo mínimo rotacionado)

# --- Detecção de Anexos / Edículas (mesmo lote) ------------------------------
# Fontes: Silva (2025, p. 31); Res. Normativa ANEEL nº 1.000/2021 (ponto de entrega único)
DIST_MAX_ANEXO_M    = 6.0        # Distância polígono a polígono até casa com medidor
AREA_MAX_ANEXO_M2   = 20.0       # Área máxima para ser considerado anexo

# --- Classes de UCs Residenciais da Distribuidora ----------------------------
CLASSES_RES         = ["RE1", "RE2"]

# --- Tarifa e Perfis de Consumo (kWh/mês) ------------------------------------
PRECO_KWH           = 0.95       # R$/kWh (Tarifa média B1 - CPFL Paulista / Campinas)

CONSUMO_ARTIGO_VERAO    = 460.0                  # P2: Verão (Dissertação de referência)
CONSUMO_ARTIGO_INVERNO  = 190.0                  # P2: Inverno (Dissertação de referência)
CONSUMO_PROPRIO         = 105.77                 # P3: Modelagem própria por equipamentos
TABELA2_RES_SUDESTE_ANO       = 2088.7          # P4: Artigo / EPE / PROCEL (kWh/UH.ano)
TABELA2_BAIXA_RENDA_SUDESTE_ANO = 1843.9        # P5: Artigo / EPE / PROCEL (kWh/UH.ano)
MEDIANA_PADRAO_BAIRRO   = 167.0                  # P6: Mediana residencial padrão

TIPO_COM   = "COM MEDIDOR"
TIPO_RES   = "SUSPEITA RESIDENCIAL"
TIPO_ANEXO = "ANEXO / EDICULA (COM MEDIDOR)"
TIPO_NAO_R = "NAO RESIDENCIAL (GALPAO/COMERCIO)"

# =============================================================================
# FUNÇÕES DE APOIO MORFOLÓGICO
# =============================================================================
def retangularidade(geoms):
    """
    Calcula a Razão de Retangularidade: Área do polígono / Área do retângulo mínimo orientado.
    Valores próximos a 1.0 indicam construções regulares/ortogonais (galpões, galpões logísticos).
    """
    arr   = np.asarray(list(geoms), dtype=object)
    a_pol = shapely.area(arr)
    a_ret = shapely.area(shapely.minimum_rotated_rectangle(arr))
    return np.where(a_ret > 0, a_pol / np.where(a_ret > 0, a_ret, 1.0), 0.0)

def classificar_inicial(df):
    """Classifica previamente se a edificação sem medidor é galpão/comércio ou residencial."""
    galpao = (df["area_casa_m2"] >= AREA_MIN_GALPAO) & (df["retangularidade"] >= RETANG_MIN_GALPAO)
    df["tipo"] = np.where(df["qtd_ucs"] > 0, TIPO_COM,
                 np.where(galpao, TIPO_NAO_R, TIPO_RES))
    return df

# =============================================================================
# ETAPA 1: PROCESSAMENTO DO OPENBUILDINGS (Executado apenas 1 vez)
# =============================================================================
print("=" * 80)
print("INICIANDO PROCESSAMENTO DA BASE GEOESPACIAL")
print("=" * 80)
print("\n1. Processando OpenBuildings...")

centro_shapely = Point(LON_CENTRO, LAT_CENTRO)
limite_estudo = (
    gpd.GeoDataFrame(geometry=[centro_shapely], crs="EPSG:4326")
    .to_crs(epsg=31983).buffer(RAIO_M).to_crs(epsg=4326)
)
lon_min, lat_min, lon_max, lat_max = limite_estudo.total_bounds

DTYPE = {"confidence":"float32","latitude":"float32","longitude":"float32","geometry":"object"}
geometrias = []

for i, chunk in enumerate(
    pd.read_csv(ARQ_OPENBUILDINGS, usecols=list(DTYPE.keys()), dtype=DTYPE, chunksize=CHUNKSIZE)
):
    mask = (
        (chunk["confidence"] >= CONFIANCA_MIN) &
        (chunk["longitude"]  >= lon_min) & (chunk["longitude"] <= lon_max) &
        (chunk["latitude"]   >= lat_min) & (chunk["latitude"]  <= lat_max)
    )
    chunk = chunk.loc[mask, "geometry"]
    if not chunk.empty:
        geometrias.extend(gpd.GeoSeries.from_wkt(chunk).tolist())
    del chunk; gc.collect()

print(f"   {len(geometrias):,} polígonos brutos coletados.")

gdf_bruto = gpd.GeoDataFrame(geometry=geometrias, crs="EPSG:4326")
del geometrias; gc.collect()

gdf_bruto = gpd.clip(gdf_bruto, limite_estudo).to_crs(epsg=31983).reset_index(drop=True)
gdf_bruto["area_m2"] = gdf_bruto.geometry.area

antes = len(gdf_bruto)
gdf_filtrado = gdf_bruto[gdf_bruto["area_m2"] >= AREA_MIN_M2].copy().reset_index(drop=True)
print(f"   {antes-len(gdf_filtrado):,} descartados (< {AREA_MIN_M2}m²) | {len(gdf_filtrado):,} mantidos.")
del gdf_bruto; gc.collect()

# =============================================================================
# ETAPA 2: MESCLAGEM POR BUFFER (0.3m) E IDENTIFICAÇÃO DE GRUPOS
# =============================================================================
print(f"\n2. Mesclando polígonos contíguos (buffer={BUFFER_MESCLA_M}m)...")

gdf_m = gdf_filtrado.copy()
gdf_m["geom_inflada"] = gdf_m.geometry.buffer(BUFFER_MESCLA_M)

unido = unary_union(gdf_m["geom_inflada"].values)
grupos = (
    gpd.GeoDataFrame(geometry=[unido], crs="EPSG:31983")
    .explode(index_parts=False).reset_index(drop=True)
)
grupos["grupo_id"] = range(len(grupos))

# Ponto representativo para indexação estrita dentro do polígono
gdf_pontos = gpd.GeoDataFrame(
    {"idx_orig": gdf_m.index},
    geometry=gdf_m.geometry.representative_point().values,
    crs="EPSG:31983"
)
join = gpd.sjoin(gdf_pontos, grupos, how="left", predicate="within")
join = join.drop_duplicates(subset=["idx_orig"], keep="first")
gdf_m["grupo_id"] = gdf_m.index.map(join.set_index("idx_orig")["grupo_id"])

print("   Calculando centroides reais e polígonos por grupo...")
edificacoes_base = []

for grupo_id, grupo in gdf_m.groupby("grupo_id"):
    _c = grupo.geometry.centroid
    pontos = list(zip(_c.x.tolist(), _c.y.tolist()))

    # Círculo envolvente pelo Shapely nativo
    if len(pontos) == 1:
        cx, cy = pontos[0]; raio = 0.0
    else:
        mp = MultiPoint(pontos)
        raio = float(shapely.minimum_bounding_radius(mp))
        centro_circ = shapely.minimum_bounding_circle(mp).centroid
        if raio == 0.0 or centro_circ.is_empty:
            cx, cy = pontos[0]; raio = 0.0
        else:
            cx, cy = centro_circ.x, centro_circ.y

    poligono_grupo = unary_union(grupo.geometry.values)

    edificacoes_base.append({
        "grupo_id":      grupo_id,
        "centroide":     Point(cx, cy),
        "poligono":      poligono_grupo,      
        "area_casa_m2":  grupo["area_m2"].sum(),
        "n_edificacoes": len(grupo),
        "raio_grupo_m":  round(raio, 2),
    })

gdf_edificacoes_base = gpd.GeoDataFrame(edificacoes_base, geometry="centroide", crs="EPSG:31983")
gdf_edificacoes_base["id_casa"] = range(1, len(gdf_edificacoes_base)+1)

cent_wgs = gdf_edificacoes_base.to_crs(epsg=4326)
gdf_edificacoes_base["lat_centroide"] = cent_wgs.geometry.y.round(6).values
gdf_edificacoes_base["lon_centroide"] = cent_wgs.geometry.x.round(6).values
gdf_edificacoes_base["retangularidade"] = retangularidade(gdf_edificacoes_base["poligono"]).round(3)

print(f"   {len(gdf_edificacoes_base):,} edificações resultantes após mesclagem.")

# Polígonos originais preservados para desmesclagem seletiva de unidades sem medidor
gdf_poligonos_originais = gdf_m[["grupo_id", "area_m2", "geometry"]].dropna(subset=["grupo_id"]).copy()
del gdf_m, gdf_filtrado, grupos, join; gc.collect()

# =============================================================================
# ETAPA 3: PROCESSAMENTO DA BASE DE UCs (CADASTRO COMERCIAL)
# =============================================================================
print("\n3. Processando cadastro de UCs...")

df_ucs_raw = pd.read_excel(ARQ_UCS)
df_ucs_raw = df_ucs_raw.dropna(subset=["Coordenada"]).copy()
df_ucs_raw["geometry"] = gpd.GeoSeries.from_wkt(df_ucs_raw["Coordenada"])
gdf_ucs_todas = gpd.GeoDataFrame(df_ucs_raw, geometry="geometry", crs="EPSG:4326").to_crs(epsg=31983)

centro_metros = gpd.GeoSeries([centro_shapely], crs="EPSG:4326").to_crs(epsg=31983).iloc[0]
gdf_ucs_todas["dist_centro"] = gdf_ucs_todas.geometry.distance(centro_metros)

gdf_ucs = gdf_ucs_todas[gdf_ucs_todas["dist_centro"] <= RAIO_M].copy().reset_index(drop=True)
print(f"   {len(gdf_ucs):,} UCs no raio de {RAIO_M/1000:.0f}km.")

# Cálculo dinâmico do consumo mediano residencial local (P6)
mask_ucs_res = gdf_ucs["Classe"].astype(str).str.upper().str.strip().isin(CLASSES_RES)
gdf_ucs_res = gdf_ucs[mask_ucs_res] if mask_ucs_res.any() else gdf_ucs
cons_res_pos = gdf_ucs_res.loc[gdf_ucs_res["Consumo Mensal"] > 0, "Consumo Mensal"]

if not cons_res_pos.empty:
    MEDIANA_RESIDENCIAL_LOCAL = float(cons_res_pos.median())
    MEDIA_RESIDENCIAL_LOCAL   = float(cons_res_pos.mean())
else:
    MEDIANA_RESIDENCIAL_LOCAL = MEDIANA_PADRAO_BAIRRO
    MEDIA_RESIDENCIAL_LOCAL   = MEDIANA_PADRAO_BAIRRO

print(f"   UCs Residenciais na área de estudo: {len(gdf_ucs_res):,}")
print(f"   -> Consumo Residencial Positivo Local: Mediana = {MEDIANA_RESIDENCIAL_LOCAL:.1f} kWh/mes | "
      f"Media = {MEDIA_RESIDENCIAL_LOCAL:.1f} kWh/mes")

# Dicionário de Perfis de Consumo (P1 REMOVIDO; apenas P2 a P6)
PERFIS = {
    "p2_verao":   {"nome": "P2 Sazonal Verão (Dissertação)",         "kwh": CONSUMO_ARTIGO_VERAO,   "origem": "Dissertação de referência"},
    "p2_inverno": {"nome": "P2 Sazonal Inverno (Dissertação)",       "kwh": CONSUMO_ARTIGO_INVERNO, "origem": "Dissertação de referência"},
    "p3_proprio": {"nome": "P3 Perfil Próprio (Equipamentos)",       "kwh": CONSUMO_PROPRIO,        "origem": "Modelagem própria bottom-up / PBE INMETRO"},
    "p4_tab_res": {"nome": "P4 Tabela de Consumo - Res. Sudeste",    "kwh": round(TABELA2_RES_SUDESTE_ANO / 12, 2), "origem": "Artigo / EPE / PROCEL"},
    "p5_tab_br":  {"nome": "P5 Tabela de Consumo - Baixa Renda",     "kwh": round(TABELA2_BAIXA_RENDA_SUDESTE_ANO / 12, 2), "origem": "Artigo / EPE / PROCEL"},
    "p6_mediana": {"nome": "P6 Mediana Residencial Local das UCs",   "kwh": round(MEDIANA_RESIDENCIAL_LOCAL, 1), "origem": "Cadastro local de UCs (clientes regulares > 0)"},
}
PERFIL_PRINCIPAL = "p6_mediana"

# =============================================================================
# FUNÇÃO DE ASSOCIAÇÃO COMBINATÓRIA E RESGATE DE ÓRFÃS
# =============================================================================
def executar_associacao(gdf_edif, gdf_ucs_input, cabo_max):
    """
    Executa o casamento de medidores às edificações respeitando a restrição física
    do comprimento máximo de cabo (cabo_max).
    Utiliza casamento bipartido de peso mínimo (Branch and Bound / Hungarian relaxado)
    para maximizar edificações atendidas e minimizar distância total de cabos.
    """
    gdf_edif_pts = gpd.GeoDataFrame(
        gdf_edif[["id_casa"]].reset_index(drop=True),
        geometry=list(gdf_edif["centroide"]), crs="EPSG:31983"
    )
    ucs_pts = gdf_ucs_input[["Código ID", "geometry"]].reset_index(drop=True)
    ucs_buf = ucs_pts.copy()
    ucs_buf["geometry"] = ucs_buf.geometry.buffer(cabo_max)

    # 1. Pares possíveis dentro do cabo máximo
    pares = gpd.sjoin(gdf_edif_pts, ucs_buf, how="inner", predicate="within")
    pares = pares.rename(columns={"index_right": "idx_uc"})
    pares["distancia"] = pares.geometry.distance(
        ucs_pts.geometry.iloc[pares["idx_uc"].values], align=False
    )
    pares = pares[pares["distancia"] <= cabo_max]
    pares = pares[["id_casa", "Código ID", "distancia"]].drop_duplicates(subset=["id_casa", "Código ID"])

    if pares.empty:
        cruz_inicial = pd.DataFrame(columns=["Código ID", "id_casa", "distancia"])
    elif TEM_SCIPY:
        # Casamento Bipartido Ótimo (1 UC primária por casa para evitar acúmulo desbalanceado)
        casas = pares["id_casa"].unique()
        ucs   = pares["Código ID"].unique()
        n, m  = len(casas), len(ucs)
        i_casa = pd.Index(casas).get_indexer(pares["id_casa"])
        i_uc   = pd.Index(ucs).get_indexer(pares["Código ID"])

        # Linhas: casas; Colunas: UCs reais + UCs virtuais penalizadas
        linhas  = np.concatenate([i_casa, np.arange(n)])
        colunas = np.concatenate([i_uc, m + np.arange(n)])
        pesos   = np.concatenate([pares["distancia"].to_numpy() + 0.01, np.full(n, 10_000.0)])

        grafo = csr_matrix((pesos, (linhas, colunas)), shape=(n, m + n))
        lin, col = min_weight_full_bipartite_matching(grafo)
        ok = col < m
        atrib_fase_a = dict(zip(ucs[col[ok]], casas[lin[ok]]))

        fase_a = (pd.DataFrame({"Código ID": list(atrib_fase_a.keys()), "id_casa": list(atrib_fase_a.values())})
                  .merge(pares, on=["Código ID", "id_casa"], how="left"))

        # Fase B: UCs remanescentes associadas à edificação mais próxima
        resto_ucs = pares[~pares["Código ID"].isin(atrib_fase_a.keys())]
        fase_b = resto_ucs.sort_values("distancia").drop_duplicates(subset=["Código ID"], keep="first")

        cruz_inicial = pd.concat([fase_a, fase_b], ignore_index=True)
    else:
        # Fallback guloso por escassez
        opcoes = pares.groupby("id_casa")["Código ID"].count().rename("n_opc")
        pares_ord = pares.join(opcoes, on="id_casa").sort_values(["n_opc", "distancia"])
        ucs_usadas = set()
        resultado = []
        for uc, casa, dist in zip(pares_ord["Código ID"], pares_ord["id_casa"], pares_ord["distancia"]):
            if uc not in ucs_usadas:
                ucs_usadas.add(uc)
                resultado.append({"Código ID": uc, "id_casa": casa, "distancia": dist})
        cruz_inicial = pd.DataFrame(resultado)

    # 2. Resgate de UCs órfãs pelo polígono real da edificação
    orfas = gdf_ucs_input[~gdf_ucs_input["Código ID"].isin(cruz_inicial["Código ID"])].copy()
    if not orfas.empty:
        gdf_edif_pol = gpd.GeoDataFrame(
            gdf_edif[["id_casa"]].reset_index(drop=True),
            geometry=list(gdf_edif["poligono"]), crs="EPSG:31983"
        )
        resgate = gpd.sjoin_nearest(
            orfas, gdf_edif_pol, how="inner", max_distance=cabo_max, distance_col="distancia"
        ).sort_values("distancia").drop_duplicates(subset=["Código ID"], keep="first")

        resgate_df = resgate[["Código ID", "id_casa", "distancia"]].copy()
        cruz_final = pd.concat([cruz_inicial, resgate_df], ignore_index=True)
    else:
        cruz_final = cruz_inicial

    return cruz_final.drop_duplicates(subset=["Código ID"], keep="first")

# =============================================================================
# PIPELINE COMPLETO DE SIMULAÇÃO (Executado para cabo=30m e cabo=45m)
# =============================================================================
def simular_cenario(cabo_max):
    print("\n" + "=" * 80)
    print(f"EXECUTANDO SIMULAÇÃO: CABO MÁXIMO = {cabo_max:.0f} METROS")
    print("=" * 80)

    # Associação
    cruzamento = executar_associacao(gdf_edificacoes_base, gdf_ucs, cabo_max)
    print(f"   UCs associadas: {len(cruzamento):,} | Órfãs: {len(gdf_ucs) - len(cruzamento):,}")

    # Consolidação inicial
    cruz_completo = gdf_ucs.merge(cruzamento, on="Código ID", how="inner")
    agrupado = cruz_completo.groupby("id_casa").agg(
        qtd_ucs       = ("Código ID",      "count"),
        consumo_total = ("Consumo Mensal", "sum"),
        classes_ucs   = ("Classe", lambda x: ", ".join(x.astype(str).unique()))
    ).reset_index()

    gdf_f = gdf_edificacoes_base.merge(agrupado, on="id_casa", how="left")
    gdf_f["qtd_ucs"]       = gdf_f["qtd_ucs"].fillna(0).astype(int)
    gdf_f["consumo_total"] = gdf_f["consumo_total"].fillna(0.0)
    gdf_f["classes_ucs"]   = gdf_f["classes_ucs"].fillna("SEM MEDIDOR")
    gdf_f = classificar_inicial(gdf_f)

    # Desfaz a mesclagem apenas de grupos residenciais sem medidor com mais de 1 polígono
    mask_desfazer = (gdf_f["tipo"] == TIPO_RES) & (gdf_f["n_edificacoes"] > 1)
    gdf_desfazer = gdf_f[mask_desfazer]
    gdf_resto    = gdf_f[~mask_desfazer].copy()

    orig = gdf_poligonos_originais[
        gdf_poligonos_originais["grupo_id"].isin(gdf_desfazer["grupo_id"])
    ].reset_index(drop=True)

    if not orig.empty:
        proximo_id = int(gdf_f["id_casa"].max()) + 1
        gdf_novos = gpd.GeoDataFrame({
            "id_casa":       range(proximo_id, proximo_id + len(orig)),
            "grupo_id":      orig["grupo_id"].values,
            "centroide":     list(orig.geometry.representative_point()),
            "poligono":      list(orig.geometry),
            "area_casa_m2":  orig["area_m2"].values,
            "n_edificacoes": 1,
            "raio_grupo_m":  0.0,
            "qtd_ucs":       0,
            "consumo_total": 0.0,
            "classes_ucs":   "SEM MEDIDOR",
        }, geometry="centroide", crs="EPSG:31983")

        cent_wgs = gdf_novos.to_crs(epsg=4326)
        gdf_novos["lat_centroide"] = cent_wgs.geometry.y.round(6).values
        gdf_novos["lon_centroide"] = cent_wgs.geometry.x.round(6).values
        gdf_novos["retangularidade"] = retangularidade(gdf_novos["poligono"]).round(3)
        gdf_novos = classificar_inicial(gdf_novos)

        gdf_f = gpd.GeoDataFrame(
            pd.concat([gdf_resto, gdf_novos], ignore_index=True),
            geometry="centroide", crs="EPSG:31983"
        )
        print(f"   Desfeita mesclagem de {len(gdf_desfazer):,} grupos -> {len(gdf_novos):,} polígonos originais recuperados.")
    else:
        gdf_f = gdf_resto.copy()

    # Identificação de Anexos / Edículas no mesmo lote
    gdf_com = gdf_f[gdf_f["qtd_ucs"] > 0]
    gdf_com_pol = gpd.GeoDataFrame(
        gdf_com[["id_casa", "area_casa_m2"]].copy(),
        geometry=list(gdf_com["poligono"]), crs="EPSG:31983"
    )
    gdf_sus = gdf_f[gdf_f["tipo"] == TIPO_RES]
    gdf_sus_pol = gpd.GeoDataFrame(
        gdf_sus[["id_casa", "area_casa_m2"]].copy(),
        geometry=list(gdf_sus["poligono"]), crs="EPSG:31983"
    )

    gdf_f["id_casa_principal"] = np.nan
    gdf_f["dist_medidor_m"]    = np.nan

    if not gdf_sus_pol.empty and not gdf_com_pol.empty:
        j_anexos = gpd.sjoin_nearest(
            gdf_sus_pol, gdf_com_pol, how="inner",
            max_distance=DIST_MAX_ANEXO_M, distance_col="dist_m",
            lsuffix="sus", rsuffix="com"
        ).drop_duplicates(subset=["id_casa_sus"])

        filtro_anexo = (j_anexos["dist_m"] <= DIST_MAX_ANEXO_M) & (j_anexos["area_casa_m2_sus"] <= AREA_MAX_ANEXO_M2)
        anexos_match = j_anexos[filtro_anexo]

        id_map   = dict(zip(anexos_match["id_casa_sus"], anexos_match["id_casa_com"]))
        dist_map = dict(zip(anexos_match["id_casa_sus"], anexos_match["dist_m"]))

        mask_anexos = gdf_f["id_casa"].isin(anexos_match["id_casa_sus"])
        gdf_f.loc[mask_anexos, "tipo"] = TIPO_ANEXO
        gdf_f.loc[mask_anexos, "id_casa_principal"] = gdf_f.loc[mask_anexos, "id_casa"].map(id_map)
        gdf_f.loc[mask_anexos, "dist_medidor_m"]    = gdf_f.loc[mask_anexos, "id_casa"].map(dist_map).round(2)

        print(f"   Anexos/Edículas identificados (dist <= {DIST_MAX_ANEXO_M:.0f}m, área <= {AREA_MAX_ANEXO_M2:.0f}m²): "
              f"{mask_anexos.sum():,} ({mask_anexos.sum()/len(gdf_sus)*100:.1f}% das suspeitas brutas)")

    # Aplicação dos Perfis de Consumo (P2 a P6) - P1 REMOVIDO
    mask_res     = gdf_f["tipo"] == TIPO_RES
    mask_nao_res = gdf_f["tipo"] == TIPO_NAO_R
    mask_anexo   = gdf_f["tipo"] == TIPO_ANEXO

    for chave, info in PERFIS.items():
        col_kwh, col_rs = f"est_{chave}", f"valor_{chave}"
        gdf_f[col_kwh] = 0.0
        gdf_f[col_rs]  = 0.0
        # Consumo residencial aplicado exclusivamente às residências suspeitas
        gdf_f.loc[mask_res, col_kwh] = info["kwh"]
        gdf_f.loc[mask_res, col_rs]  = info["kwh"] * PRECO_KWH

    gdf_res     = gdf_f[mask_res].copy()
    gdf_anexo   = gdf_f[mask_anexo].copy()
    gdf_nao_res = gdf_f[mask_nao_res].copy()
    ucs_orfas   = gdf_ucs[~gdf_ucs["Código ID"].isin(cruzamento["Código ID"])].copy()

    # Tabela Resumo de Perdas Financeiras e Energéticas
    linhas_resumo = []
    for chave, info in PERFIS.items():
        kwh_mes = gdf_res[f"est_{chave}"].sum()
        rs_mes  = gdf_res[f"valor_{chave}"].sum()
        linhas_resumo.append({
            "Perfil": info["nome"],
            "Fundamentação / Origem": info["origem"],
            "kWh/mês por Edificação": info["kwh"],
            "Edificações Suspeitas": len(gdf_res),
            "Perda Total (kWh/mês)": round(kwh_mes, 0),
            "Perda Total (R$/mês)": round(rs_mes, 2),
            "Perda Anual (kWh/ano)": round(kwh_mes * 12, 0),
            "Perda Anual (R$/ano)": round(rs_mes * 12, 2),
        })

    df_resumo = pd.DataFrame(linhas_resumo)

    print(f"\n   RESULTADO DA SIMULAÇÃO (CABO = {cabo_max:.0f}m):")
    print(f"   🟢 Com Medidor                : {(gdf_f['qtd_ucs'] > 0).sum():,}")
    print(f"   ⚪ Anexos/Edículas (mesmo lote): {len(gdf_anexo):,}")
    print(f"   🟣 Galpões/Comércios (Grupo A): {len(gdf_nao_res):,}  (sem perdas fictícias)")
    print(f"   🔴 Suspeitas Residenciais     : {len(gdf_res):,}")
    print(f"   ⚫ UCs Órfãs                  : {len(ucs_orfas):,}")

    for _, r in df_resumo.iterrows():
        marca = "  <-- CENÁRIO CENTRAL" if r["Perfil"] == PERFIS[PERFIL_PRINCIPAL]["nome"] else ""
        print(f"     ↳ {r['Perfil']:<38}: {r['Perda Total (kWh/mês)']:>12,.0f} kWh/mês | "
              f"R$ {r['Perda Total (R$/mês)']:>12,.2f}{marca}")

    return {
        "cabo_max":    cabo_max,
        "gdf_final":   gdf_f,
        "gdf_res":     gdf_res,
        "gdf_anexo":   gdf_anexo,
        "gdf_nao_res": gdf_nao_res,
        "ucs_orfas":   ucs_orfas,
        "df_resumo":   df_resumo,
        "cruzamento":  cruzamento,
    }

# =============================================================================
# EXECUÇÃO DAS DUAS SIMULAÇÕES
# =============================================================================
res_30m = simular_cenario(30.0)
res_45m = simular_cenario(45.0)

# =============================================================================
# ETAPA 4: TABELA COMPARATIVA EXECUTIVA (30m vs 45m)
# =============================================================================
print("\n" + "=" * 80)
print("COMPARATIVO CONSOLIDADO: CABO 30m vs CABO 45m")
print("=" * 80)

comp_edif = pd.DataFrame({
    "Indicador": [
        "Total de Edificações Mapeadas",
        "Edificações com Medidor Ativo",
        "Anexos / Edículas no mesmo lote (isentos de furto)",
        "Galpões / Comércios (Verificação Grupo A)",
        "Edificações Suspeitas Residenciais",
        "UCs Cadastradas Associadas",
        "UCs Órfãs (sem edificação próxima)",
        "Perda P6 Mediana Local (kWh/mês)",
        "Perda P6 Mediana Local (R$/mês)",
        "Perda P6 Mediana Local (R$/ano)",
        "Perda P2 Verão (R$/mês)",
        "Perda P2 Inverno (R$/mês)",
        "Perda P3 Próprio (R$/mês)",
    ],
    "Cenário 30 metros": [
        f"{len(res_30m['gdf_final']):,}",
        f"{(res_30m['gdf_final']['qtd_ucs'] > 0).sum():,}",
        f"{len(res_30m['gdf_anexo']):,}",
        f"{len(res_30m['gdf_nao_res']):,}",
        f"{len(res_30m['gdf_res']):,}",
        f"{len(res_30m['cruzamento']):,}",
        f"{len(res_30m['ucs_orfas']):,}",
        f"{res_30m['df_resumo'].loc[res_30m['df_resumo']['Perfil'] == PERFIS['p6_mediana']['nome'], 'Perda Total (kWh/mês)'].values[0]:,.0f} kWh",
        f"R$ {res_30m['df_resumo'].loc[res_30m['df_resumo']['Perfil'] == PERFIS['p6_mediana']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
        f"R$ {res_30m['df_resumo'].loc[res_30m['df_resumo']['Perfil'] == PERFIS['p6_mediana']['nome'], 'Perda Anual (R$/ano)'].values[0]:,.2f}",
        f"R$ {res_30m['df_resumo'].loc[res_30m['df_resumo']['Perfil'] == PERFIS['p2_verao']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
        f"R$ {res_30m['df_resumo'].loc[res_30m['df_resumo']['Perfil'] == PERFIS['p2_inverno']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
        f"R$ {res_30m['df_resumo'].loc[res_30m['df_resumo']['Perfil'] == PERFIS['p3_proprio']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
    ],
    "Cenário 45 metros": [
        f"{len(res_45m['gdf_final']):,}",
        f"{(res_45m['gdf_final']['qtd_ucs'] > 0).sum():,}",
        f"{len(res_45m['gdf_anexo']):,}",
        f"{len(res_45m['gdf_nao_res']):,}",
        f"{len(res_45m['gdf_res']):,}",
        f"{len(res_45m['cruzamento']):,}",
        f"{len(res_45m['ucs_orfas']):,}",
        f"{res_45m['df_resumo'].loc[res_45m['df_resumo']['Perfil'] == PERFIS['p6_mediana']['nome'], 'Perda Total (kWh/mês)'].values[0]:,.0f} kWh",
        f"R$ {res_45m['df_resumo'].loc[res_45m['df_resumo']['Perfil'] == PERFIS['p6_mediana']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
        f"R$ {res_45m['df_resumo'].loc[res_45m['df_resumo']['Perfil'] == PERFIS['p6_mediana']['nome'], 'Perda Anual (R$/ano)'].values[0]:,.2f}",
        f"R$ {res_45m['df_resumo'].loc[res_45m['df_resumo']['Perfil'] == PERFIS['p2_verao']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
        f"R$ {res_45m['df_resumo'].loc[res_45m['df_resumo']['Perfil'] == PERFIS['p2_inverno']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
        f"R$ {res_45m['df_resumo'].loc[res_45m['df_resumo']['Perfil'] == PERFIS['p3_proprio']['nome'], 'Perda Total (R$/mês)'].values[0]:,.2f}",
    ]
})

print(comp_edif.to_string(index=False))

# =============================================================================
# ETAPA 5: EXPORTAÇÃO EXCEL COM TODAS AS ABAS
# =============================================================================
print("\nExportando planilha comparativa com múltiplas abas...")

colunas_base = [
    "id_casa", "tipo", "area_casa_m2", "retangularidade", "n_edificacoes",
    "raio_grupo_m", "lat_centroide", "lon_centroide"
]
colunas_perfis = [f"est_{k}" for k in PERFIS] + [f"valor_{k}" for k in PERFIS]
colunas_uc = [c for c in df_ucs_raw.columns if c != "geometry"]

def salvar_planilha(caminho):
    with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
        # 1. Comparação executiva
        comp_edif.to_excel(writer, sheet_name="comparacao_30m_vs_45m", index=False)

        # 2. Resumos de perdas
        res_30m["df_resumo"].to_excel(writer, sheet_name="resumo_perdas_30m", index=False)
        res_45m["df_resumo"].to_excel(writer, sheet_name="resumo_perdas_45m", index=False)

        # 3. Suspeitas residenciais
        res_30m["gdf_res"][colunas_base + colunas_perfis].to_excel(writer, sheet_name="suspeitas_30m", index=False)
        res_45m["gdf_res"][colunas_base + colunas_perfis].to_excel(writer, sheet_name="suspeitas_45m", index=False)

        # 4. Anexos / Edículas
        res_30m["gdf_anexo"][colunas_base + ["id_casa_principal", "dist_medidor_m"]].to_excel(writer, sheet_name="anexos_30m", index=False)
        res_45m["gdf_anexo"][colunas_base + ["id_casa_principal", "dist_medidor_m"]].to_excel(writer, sheet_name="anexos_45m", index=False)

        # 5. Galpões / Comércios (Grupo A) - Idênticos em ambos os cenários
        res_30m["gdf_nao_res"][colunas_base].to_excel(writer, sheet_name="galpoes_grupo_a", index=False)

        # 6. UCs Órfãs
        res_30m["ucs_orfas"].to_crs(epsg=4326)[colunas_uc].to_excel(writer, sheet_name="ucs_orfas_30m", index=False)
        res_45m["ucs_orfas"].to_crs(epsg=4326)[colunas_uc].to_excel(writer, sheet_name="ucs_orfas_45m", index=False)

try:
    salvar_planilha(ARQ_SAIDA_EXCEL)
    print(f"   [OK] Arquivo Excel gerado com sucesso: {ARQ_SAIDA_EXCEL}")
except PermissionError:
    arq_alt = ARQ_SAIDA_EXCEL.replace(".xlsx", "_atualizado.xlsx")
    salvar_planilha(arq_alt)
    print(f"   [AVISO] '{ARQ_SAIDA_EXCEL}' estava aberto no Excel. Salvo em: {arq_alt}")

# =============================================================================
# ETAPA 6: GERAÇÃO DOS MAPAS INTERATIVOS FOLIUM (30m e 45m)
# =============================================================================
def gerar_mapa_html(res_sim, caminho_arquivo):
    cabo = res_sim["cabo_max"]
    gdf_f = res_sim["gdf_final"]
    gdf_r = res_sim["gdf_res"]
    gdf_a = res_sim["gdf_anexo"]
    gdf_g = res_sim["gdf_nao_res"]
    gdf_o = res_sim["ucs_orfas"]

    print(f"   Gerando mapa interativo para cabo={cabo:.0f}m ({caminho_arquivo})...")

    gdf_pol = gdf_f.copy().set_geometry("poligono").set_crs(epsg=31983).to_crs(epsg=4326)
    gdf_pts = gdf_f.copy().set_geometry("centroide").set_crs(epsg=31983).to_crs(epsg=4326)

    def cor_marcador(row):
        if row["tipo"] == TIPO_NAO_R: return "purple"
        if row["tipo"] == TIPO_ANEXO: return "gray"
        if row["qtd_ucs"] == 0:       return "red"
        if row["qtd_ucs"] == 1:       return "green"
        return "blue"

    mapa = folium.Map(location=[LAT_CENTRO, LON_CENTRO], zoom_start=14, tiles=None)

    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satélite", overlay=False, control=True
    ).add_to(mapa)
    folium.TileLayer("CartoDB positron", name="Ruas", overlay=False, control=True).add_to(mapa)

    # Camada 1: Polígonos
    grupo_pol = folium.FeatureGroup(name="Polígonos das Edificações", show=True)
    gdf_pol_mapa = gdf_pol.drop(columns=["centroide"])
    gdf_pol_mapa["poligono"] = gdf_pol_mapa["poligono"].simplify(0.00001, preserve_topology=True)

    folium.GeoJson(
        gdf_pol_mapa,
        style_function=lambda _: {
            "fillColor": "#555555", "color": "#222222", "weight": 1.0, "fillOpacity": 0.15
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["id_casa", "tipo", "area_casa_m2", "n_edificacoes", "qtd_ucs"],
            aliases=["ID:", "Classificação:", "Área (m²):", "Polígonos agrupados:", "Medidores:"],
            localize=True, sticky=True
        )
    ).add_to(grupo_pol)
    grupo_pol.add_to(mapa)

    # Camada 2: Marcadores Clusterizados
    grupo_cluster = folium.FeatureGroup(name=f"Edificações (Cabo máx: {cabo:.0f}m)", show=True)
    cluster = MarkerCluster().add_to(grupo_cluster)

    for _, row in gdf_pts.iterrows():
        if row["tipo"] == TIPO_NAO_R:
            texto = (
                f"<b style='color:purple;'>Não Residencial (Galpão/Comércio)</b><br>"
                f"<i>Área: {row['area_casa_m2']:.0f} m² | Retangularidade: {row['retangularidade']:.2f}</i><br>"
                f"<span style='color:#4a148c;'>Verificar cadastro em Média Tensão (Grupo A)</span>"
            )
        elif row["tipo"] == TIPO_ANEXO:
            id_p = int(row['id_casa_principal']) if pd.notna(row['id_casa_principal']) else "-"
            d_p  = f"{row['dist_medidor_m']:.1f}m" if pd.notna(row['dist_medidor_m']) else "-"
            texto = (
                f"<b style='color:#555;'>Anexo / Edícula no mesmo lote</b><br>"
                f"<i>Atendido pela casa com medidor ID {id_p} (a {d_p})</i><br>"
                f"<span style='color:#2e7d32;'>Não contabilizado como furto</span>"
            )
        elif row["qtd_ucs"] == 0:
            linhas = "".join([
                f" ↳ {info['nome']}: <b>{row[f'est_{k}']:.0f} kWh</b> "
                f"(<span style='color:#cc0000;'>R$ {row[f'valor_{k}']:.2f}</span>)<br>"
                for k, info in PERFIS.items()
            ])
            texto = f"<b style='color:red;'>Estimativa de Furto/Mês:</b><br>{linhas}"
        else:
            texto = f"<b style='color:green;'>Consumo Medido (Real):</b> {row['consumo_total']:.0f} kWh"

        folium.CircleMarker(
            location=[row["lat_centroide"], row["lon_centroide"]],
            radius=6,
            color=cor_marcador(row),
            fill=True,
            fill_color=cor_marcador(row),
            fill_opacity=0.85,
            tooltip=(
                f"<b>ID:</b> {row['id_casa']}<br>"
                f"<b>Tipo:</b> {row['tipo']}<br>"
                f"<b>Área:</b> {row['area_casa_m2']:.0f} m²<br>"
                f"<b>Medidores:</b> {row['qtd_ucs']}<br>"
                f"<hr style='margin:4px 0;'>"
                f"{texto}<br>"
                f"<hr style='margin:4px 0;'>"
                f"<b>Classes UCs:</b> {row['classes_ucs']}"
            )
        ).add_to(cluster)
    grupo_cluster.add_to(mapa)

    # Camada 3: UCs Órfãs
    grupo_orfas = folium.FeatureGroup(name="UCs Órfãs", show=False)
    for _, row in gdf_o.to_crs(epsg=4326).iterrows():
        folium.CircleMarker(
            location=[row["geometry"].y, row["geometry"].x],
            radius=3, color="black", fill=True, fill_opacity=1.0,
            tooltip=f"Órfã: {row['Código ID']} | {row['Classe']} ({row['Consumo Mensal']} kWh)"
        ).add_to(grupo_orfas)
    grupo_orfas.add_to(mapa)

    # Raio de Estudo
    folium.Circle(
        location=[LAT_CENTRO, LON_CENTRO], radius=RAIO_M,
        color="#0055ff", weight=2.5, fill=False, dash_array="8",
        tooltip=f"Área de estudo: {RAIO_M/1000:.0f} km de raio"
    ).add_to(mapa)

    # Marcador Central
    folium.Marker(
        location=[LAT_CENTRO, LON_CENTRO],
        tooltip="Centro de pesquisa",
        popup=folium.Popup(f"""
            <div style='text-align:center; width:220px; font-family:Arial;'>
                <b>Campo Belo — M1 (Cabo: {cabo:.0f}m)</b><br>
                Raio: {RAIO_M/1000:.0f} km<br><br>
                Total edificações: {len(gdf_f):,}<br>
                🟢 Com medidor: {(gdf_f['qtd_ucs']>0).sum():,}<br>
                ⚪ Anexos / Edículas: {len(gdf_a):,}<br>
                🔴 Suspeitas residenciais: {len(gdf_r):,}<br>
                🟣 Galpões / Comércios: {len(gdf_g):,}<br><br>
                UCs no raio: {len(gdf_ucs):,}<br>
                UCs órfãs: {len(gdf_o):,}
            </div>""", max_width=320),
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(mapa)

    # Legenda Fixa
    legenda_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:white; padding:15px; border-radius:8px;
         border:2px solid #ccc; font-family:Arial; font-size:13px; max-height:460px; overflow-y:auto;">
      <b>Metodologia 1 (Múltiplas UCs)</b><br>
      <i style="font-size:11px;">Cabo de serviço máximo: <b>{cabo:.0f} metros</b></i><br>
      <i style="font-size:11px;">conf≥{CONFIANCA_MIN} | área≥{AREA_MIN_M2:.0f}m²</i><br><br>
      <span style="background:red;padding:2px 10px;border-radius:50%;">&nbsp;</span>
      &nbsp;<b>Suspeita residencial ({len(gdf_r):,})</b><br><br>
      <span style="background:gray;padding:2px 10px;border-radius:50%;">&nbsp;</span>
      &nbsp;Anexo / Edícula ({len(gdf_a):,})<br><br>
      <span style="background:purple;padding:2px 10px;border-radius:50%;">&nbsp;</span>
      &nbsp;Galpão / Comércio ({len(gdf_g):,})<br><br>
      <span style="background:green;padding:2px 10px;border-radius:50%;">&nbsp;</span>
      &nbsp;1 medidor ({(gdf_f['qtd_ucs']==1).sum():,})<br><br>
      <span style="background:blue;padding:2px 10px;border-radius:50%;">&nbsp;</span>
      &nbsp;2+ medidores ({(gdf_f['qtd_ucs']>1).sum():,})<br><br>
      <span style="background:black;padding:2px 10px;border-radius:50%;">&nbsp;</span>
      &nbsp;UC órfã ({len(gdf_o):,})<br><br>
      <hr style="margin:6px 0;">
      <i>Total: {len(gdf_f):,} edificações</i>
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(legenda_html))

    titulo_html = f'''
    <h3 align="center" style="font-size:16px; font-family:Arial;">
        <b>M1 | Cabo Máximo: {cabo:.0f}m | Total: {len(gdf_f):,} |
        Com Medidor: {(gdf_f["qtd_ucs"]>0).sum():,} |
        Anexos: {len(gdf_a):,} |
        Suspeitas Residenciais: {len(gdf_r):,}</b>
    </h3>'''
    mapa.get_root().html.add_child(folium.Element(titulo_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(mapa)
    mapa.save(caminho_arquivo)
    print(f"   [OK] Mapa salvo: {caminho_arquivo}")

gerar_mapa_html(res_30m, ARQ_MAPA_30M)
gerar_mapa_html(res_45m, ARQ_MAPA_45M)

print("\n" + "=" * 80)
print("PROCESSAMENTO FINALIZADO COM SUCESSO!")
print(f"  -> Planilha: {ARQ_SAIDA_EXCEL}")
print(f"  -> Mapa 30m: {ARQ_MAPA_30M}")
print(f"  -> Mapa 45m: {ARQ_MAPA_45M}")
print("=" * 80)
