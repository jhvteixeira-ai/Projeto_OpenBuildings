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

try:  # casamento ótimo (recomendado): python -m pip install scipy
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    TEM_SCIPY = True
except ImportError:
    TEM_SCIPY = False

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
ARQ_OPENBUILDINGS = "Arquivos/OpBuildingCPS.gz"
ARQ_UCS           = "Arquivos/lista_ucs_fernanda_joao.xlsx"
ARQ_SAIDA_MAPA    = "metodologia1_ic.html"
ARQ_SEM_MEDIDOR   = "m1_edificacoes_sem_medidor.xlsx"
ARQ_ORFAS         = "m1_ucs_sem_edificacao.xlsx"

RAIO_M          = 3_000
LON_CENTRO      = -47.1205
LAT_CENTRO      = -23.0384
CONFIANCA_MIN   = 0.75
AREA_MIN_M2     = 50.0
BUFFER_MESCLA_M = 0.3
CABO_MAX_M      = 30.0    
CHUNKSIZE       = 200_000

# --- Associação UC -> edificação (Passo 4) ------------------------------------
CABO_FASE_A_M       = 20.0     # alcance do casamento ótimo (menor que CABO_MAX_M evita
                               # que UCs de prédios "cubram" casas distantes)
PENALIDADE_SEM_UC   = 10_000.0 # custo de deixar uma edificação sem UC no casamento
CABOS_SENSIBILIDADE = [20, 30, 45, 60]   # só imprime a sensibilidade
CENARIO_PRINCIPAL   = "minimo"  # "minimo" (casamento) ou "maximo" (UC mais próxima):
                                # define qual cenário gera planilhas e mapa

# --- Parâmetros de Galpão / Comércio (Não Residencial) ------------------------
AREA_MIN_GALPAO   = 400.0   # m² (acima disso e retangular = galpão/comércio)
RETANG_MIN_GALPAO = 0.85    # índice de retangularidade (1.0 = retângulo perfeito)

# --- Detecção de Anexos / Edículas (polígonos dependentes no mesmo lote) -------
DIST_MAX_ANEXO_M  = 6.0     # distância máx. (m) polígono-a-polígono até casa com medidor
AREA_MAX_ANEXO_M2 = 90.0    # área máx. (m²) para ser considerado anexo/edícula

# --- Classes de UCs consideradas residenciais ---------------------------------
CLASSES_RES = ["RE1", "RE2"]

# --- Preço e Perfis de Consumo (kWh/mês por edificação sem medidor) -----------
PRECO_KWH = 0.95  # R$/kWh (CPFL Paulista - Campinas)

KWH_POR_M2_MES          = (1.5 * 24 * 30) / 1000  # P1: Científico (1,08 kWh/m².mês)
CONSUMO_ARTIGO_VERAO    = 460.0                  # P2: Artigo (região parecida - verão)
CONSUMO_ARTIGO_INVERNO  = 190.0                  # P2: Artigo (inverno)
CONSUMO_PROPRIO         = 105.77                 # P3: Perfil próprio (por equipamentos)
TABELA2_RES_SUDESTE_ANO       = 2088.7          # P4: Tabela 2, kWh/UH.ano
TABELA2_BAIXA_RENDA_SUDESTE_ANO = 1843.9        # P5: Tabela 2, kWh/UH.ano
MEDIANA_BAIRRO_SEM_ZERO = 167.0                 # P6: Mediana UCs do bairro, sem zeros

PERFIS = {
    "p1_cientifico": {"nome": "P1 Científico (1,5 W/m²)",          "tipo": "area", "kwh": None},
    "p2_verao":      {"nome": "P2 Artigo (Verão)",                 "tipo": "fixo", "kwh": CONSUMO_ARTIGO_VERAO},
    "p2_inverno":    {"nome": "P2 Artigo (Inverno)",               "tipo": "fixo", "kwh": CONSUMO_ARTIGO_INVERNO},
    "p3_proprio":    {"nome": "P3 Perfil próprio (equipamentos)",  "tipo": "fixo", "kwh": CONSUMO_PROPRIO},
    "p4_tab_res":    {"nome": "P4 Tabela 2 - Residencial Sudeste", "tipo": "fixo", "kwh": round(TABELA2_RES_SUDESTE_ANO / 12, 2)},
    "p5_tab_br":     {"nome": "P5 Tabela 2 - Baixa renda Sudeste", "tipo": "fixo", "kwh": round(TABELA2_BAIXA_RENDA_SUDESTE_ANO / 12, 2)},
    "p6_mediana":    {"nome": "P6 Mediana UCs do bairro (s/ zeros)","tipo": "fixo", "kwh": MEDIANA_BAIRRO_SEM_ZERO},
}
PERFIL_PRINCIPAL = "p6_mediana"  # cenário central sugerido (dado local)

TIPO_COM   = "COM MEDIDOR"
TIPO_RES   = "SUSPEITA RESIDENCIAL"
TIPO_ANEXO = "ANEXO / EDICULA (COM MEDIDOR)"
TIPO_NAO_R = "NAO RESIDENCIAL"

# =============================================================================
# PASSO 1: PROCESSAR OPENBUILDINGS
# =============================================================================
print("1. Processando OpenBuildings...")

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

gdf = gpd.GeoDataFrame(geometry=geometrias, crs="EPSG:4326")
del geometrias; gc.collect()

gdf = gpd.clip(gdf, limite_estudo).to_crs(epsg=31983).reset_index(drop=True)
gdf["area_m2"] = gdf.geometry.area

antes = len(gdf)
gdf = gdf[gdf["area_m2"] >= AREA_MIN_M2].copy().reset_index(drop=True)
print(f"   {antes-len(gdf):,} descartados (< {AREA_MIN_M2}m²) | {len(gdf):,} mantidos.")

# =============================================================================
# PASSO 2: MESCLAGEM POR INTERSEÇÃO REAL + SALVA POLÍGONO DO GRUPO
# =============================================================================
print(f"\n2. Mesclando polígonos que se tocam (buffer={BUFFER_MESCLA_M}m)...")

gdf_m = gdf.copy()
gdf_m["geom_inflada"] = gdf_m.geometry.buffer(BUFFER_MESCLA_M)

unido = unary_union(gdf_m["geom_inflada"].values)
grupos = (
    gpd.GeoDataFrame(geometry=[unido], crs="EPSG:31983")
    .explode(index_parts=False).reset_index(drop=True)
)
grupos["grupo_id"] = range(len(grupos))

# representative_point() garante um ponto DENTRO do polígono
gdf_pontos = gpd.GeoDataFrame(
    {"idx_orig": gdf_m.index},
    geometry=gdf_m.geometry.representative_point().values,
    crs="EPSG:31983"
)
join = gpd.sjoin(gdf_pontos, grupos, how="left", predicate="within")
join = join.drop_duplicates(subset=["idx_orig"], keep="first")
gdf_m["grupo_id"] = gdf_m.index.map(join.set_index("idx_orig")["grupo_id"])

sem_grupo = gdf_m["grupo_id"].isna().sum()
if sem_grupo:
    print(f"   AVISO: {sem_grupo:,} polígonos ficaram sem grupo.")

print("   Calculando centroides e polígonos por grupo...")
edificacoes = []

for grupo_id, grupo in gdf_m.groupby("grupo_id"):
    _c = grupo.geometry.centroid
    pontos = list(zip(_c.x.tolist(), _c.y.tolist()))

    # Menor círculo envolvente nativo do Shapely 2.x (sem limite de recursão)
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

    edificacoes.append({
        "grupo_id":      grupo_id,
        "centroide":     Point(cx, cy),
        "poligono":      poligono_grupo,      
        "area_casa_m2":  grupo["area_m2"].sum(),
        "n_edificacoes": len(grupo),
        "raio_grupo_m":  round(raio, 2),
    })

gdf_edificacoes = gpd.GeoDataFrame(edificacoes, geometry="centroide", crs="EPSG:31983")
gdf_edificacoes["id_casa"] = range(1, len(gdf_edificacoes)+1)

cent_wgs = gdf_edificacoes.to_crs(epsg=4326)
gdf_edificacoes["lat_centroide"] = cent_wgs.geometry.y.round(6).values
gdf_edificacoes["lon_centroide"] = cent_wgs.geometry.x.round(6).values

print(f"   {len(gdf_edificacoes):,} edificações após mesclagem.")

# Guarda os polígonos ORIGINAIS do OpenBuildings de cada grupo
gdf_poligonos_orig = gdf_m[["grupo_id", "area_m2", "geometry"]].dropna(subset=["grupo_id"]).copy()

del gdf, gdf_m; gc.collect()

# =============================================================================
# PASSO 3: PROCESSAR UCs
# =============================================================================
print("\n3. Processando UCs...")

df_ucs = pd.read_excel(ARQ_UCS)
df_ucs = df_ucs.dropna(subset=["Coordenada"]).copy()
df_ucs["geometry"] = gpd.GeoSeries.from_wkt(df_ucs["Coordenada"])
gdf_ucs_todas = gpd.GeoDataFrame(df_ucs, geometry="geometry", crs="EPSG:4326").to_crs(epsg=31983)
centro_metros = gpd.GeoSeries([centro_shapely], crs="EPSG:4326").to_crs(epsg=31983).iloc[0]
gdf_ucs_todas["dist_centro"] = gdf_ucs_todas.geometry.distance(centro_metros)

gdf_ucs = gdf_ucs_todas[gdf_ucs_todas["dist_centro"] <= RAIO_M].copy()
print(f"   {len(gdf_ucs):,} UCs no raio de {RAIO_M/1000:.0f}km.")

# Checagem das UCs residenciais
mask_ucs_res = gdf_ucs["Classe"].astype(str).str.upper().str.strip().isin(CLASSES_RES)
gdf_ucs_res = gdf_ucs[mask_ucs_res] if mask_ucs_res.any() else gdf_ucs
cons_res_pos = gdf_ucs_res.loc[gdf_ucs_res["Consumo Mensal"] > 0, "Consumo Mensal"]
mediana_local = float(cons_res_pos.median()) if not cons_res_pos.empty else MEDIANA_BAIRRO_SEM_ZERO
print(f"   UCs Residenciais: {len(gdf_ucs_res):,} | Mediana s/ zero local: {mediana_local:.1f} kWh/mês "
      f"(P6 configurado em {MEDIANA_BAIRRO_SEM_ZERO:.1f} kWh)")

# =============================================================================
# PASSO 4: ASSOCIAÇÃO UC -> EDIFICAÇÃO  (dois cenários para formar uma FAIXA)
#   "minimo": casamento ótimo — cada edificação recebe no máx. 1 UC (Fase A,
#             alcance CABO_FASE_A_M), UCs extras vão para a mais próxima (Fase B).
#             Tende a SUBESTIMAR as suspeitas (UCs excedentes cobrem casas vizinhas).
#   "maximo": cada UC vai para a edificação mais próxima. Tende a SUPERESTIMAR.
# =============================================================================
print(f"\n4. Associando UCs às edificações (cabo máx. {CABO_MAX_M:.0f} m; "
      f"alcance da Fase A {CABO_FASE_A_M:.0f} m)...")
if not TEM_SCIPY:
    print("   AVISO: scipy não encontrado; Fase A usará um guloso (python -m pip install scipy).")

def retangularidade(geoms):
    """Área do polígono / área do menor retângulo girado que o envolve."""
    arr   = np.asarray(list(geoms), dtype=object)
    a_pol = shapely.area(arr)
    a_ret = shapely.area(shapely.minimum_rotated_rectangle(arr))
    return np.where(a_ret > 0, a_pol / np.where(a_ret > 0, a_ret, 1.0), 0.0)

def classificar_inicial(df):
    galpao = (df["area_casa_m2"] >= AREA_MIN_GALPAO) & (df["retangularidade"] >= RETANG_MIN_GALPAO)
    df["tipo"] = np.where(df["qtd_ucs"] > 0, TIPO_COM,
                 np.where(galpao, TIPO_NAO_R, TIPO_RES))
    return df

gdf_edificacoes["retangularidade"] = retangularidade(gdf_edificacoes["poligono"]).round(3)

gdf_edif_poligono = gpd.GeoDataFrame(
    gdf_edificacoes[["id_casa"]].reset_index(drop=True),
    geometry=list(gdf_edificacoes["poligono"]), crs="EPSG:31983"
)
gdf_edif_pontos = gpd.GeoDataFrame(
    gdf_edificacoes[["id_casa"]].reset_index(drop=True),
    geometry=list(gdf_edificacoes["centroide"]), crs="EPSG:31983"
)

def pares_candidatos(cabo_max):
    """Todos os pares (edificação, UC) a até cabo_max metros (centroide da edificação)."""
    ucs_pts = gdf_ucs[["Código ID", "geometry"]].reset_index(drop=True)
    ucs_buf = ucs_pts.copy()
    ucs_buf["geometry"] = ucs_buf.geometry.buffer(cabo_max)
    pares = gpd.sjoin(gdf_edif_pontos, ucs_buf, how="inner", predicate="within")
    pares = pares.rename(columns={"index_right": "idx_uc"})
    pares["distancia"] = pares.geometry.distance(
        ucs_pts.geometry.iloc[pares["idx_uc"].values], align=False
    )
    pares = pares[pares["distancia"] <= cabo_max]
    return (pares[["id_casa", "Código ID", "distancia"]]
            .drop_duplicates(subset=["id_casa", "Código ID"]).reset_index(drop=True))

def fase_a_casamento(pares):
    """Cada edificação recebe no máx. 1 UC; maximiza as edificações atendidas e,
    entre as soluções, minimiza a distância total. Retorna {Código ID: id_casa}."""
    casas = pares["id_casa"].unique()
    ucs   = pares["Código ID"].unique()
    n, m  = len(casas), len(ucs)

    if TEM_SCIPY:
        i_casa = pd.Index(casas).get_indexer(pares["id_casa"])
        i_uc   = pd.Index(ucs).get_indexer(pares["Código ID"])
        linhas  = np.concatenate([i_casa, np.arange(n)])        # + 1 UC fictícia por edificação
        colunas = np.concatenate([i_uc, m + np.arange(n)])
        pesos   = np.concatenate([pares["distancia"].to_numpy() + 0.01,
                                  np.full(n, PENALIDADE_SEM_UC)])
        grafo = csr_matrix((pesos, (linhas, colunas)), shape=(n, m + n))
        lin, col = min_weight_full_bipartite_matching(grafo)
        ok = col < m
        return dict(zip(ucs[col[ok]], casas[lin[ok]]))

    # Alternativa sem scipy: guloso por escassez, 1 UC por edificação
    pares = pares.copy()
    pares["n_opc"] = pares.groupby("id_casa")["Código ID"].transform("nunique")
    ordem = pares.sort_values(["n_opc", "id_casa", "distancia"])
    usadas, casas_ok, atrib = set(), set(), {}
    for uc, casa in zip(ordem["Código ID"], ordem["id_casa"]):
        if casa in casas_ok or uc in usadas:
            continue
        atrib[uc] = casa; usadas.add(uc); casas_ok.add(casa)
    return atrib

def associar_minimo(cabo_max):
    pares = pares_candidatos(cabo_max)
    if pares.empty:
        return pd.DataFrame(columns=["Código ID", "id_casa", "distancia", "fase"])
    cabo_a = min(CABO_FASE_A_M, cabo_max)
    atrib  = fase_a_casamento(pares[pares["distancia"] <= cabo_a])

    fase_a = (pd.DataFrame({"Código ID": list(atrib.keys()), "id_casa": list(atrib.values())})
              .merge(pares, on=["Código ID", "id_casa"], how="left"))
    fase_a["fase"] = "A"

    resto = pares[~pares["Código ID"].isin(atrib.keys())]
    fase_b = (resto.sort_values("distancia")
              .drop_duplicates(subset=["Código ID"], keep="first").copy())
    fase_b["fase"] = "B"
    return pd.concat([fase_a, fase_b], ignore_index=True)

def associar_maximo(cabo_max):
    j = gpd.sjoin_nearest(
        gdf_ucs[["Código ID", "geometry"]], gdf_edif_pontos,
        how="inner", max_distance=cabo_max, distance_col="distancia"
    ).drop_duplicates(subset=["Código ID"])
    out = j[["Código ID", "id_casa", "distancia"]].copy()
    out["fase"] = "N"
    return out

def rodar_associacao(modo, cabo_max, verbose=True):
    cruz = associar_minimo(cabo_max) if modo == "minimo" else associar_maximo(cabo_max)

    # Resgate das órfãs pelo polígono real da edificação
    orfas = gdf_ucs[~gdf_ucs["Código ID"].isin(cruz["Código ID"])]
    n_resg = 0
    if not orfas.empty:
        resgate = gpd.sjoin_nearest(
            orfas, gdf_edif_poligono,
            how="inner", max_distance=cabo_max, distance_col="distancia"
        )
        resgate = (resgate.sort_values("distancia")
                   .drop_duplicates(subset=["Código ID"], keep="first"))
        resg = resgate[["Código ID", "id_casa", "distancia"]].copy()
        resg["fase"] = "R"
        n_resg = len(resg)
        cruz = pd.concat([cruz, resg], ignore_index=True)
    cruz = cruz.drop_duplicates(subset=["Código ID"], keep="first")

    if verbose:
        fases = cruz["fase"].value_counts().to_dict()
        print(f"   [{modo}] UCs associadas: {len(cruz):,} (órfãs: {len(gdf_ucs) - len(cruz):,}) "
              f"| por fase: {fases} | distância mediana {cruz['distancia'].median():.1f} m")
    return cruz

ASSOC = {
    "minimo": rodar_associacao("minimo", CABO_MAX_M),
    "maximo": rodar_associacao("maximo", CABO_MAX_M),
}

print("   Sensibilidade ao cabo máximo (edificações sem nenhuma UC, antes da classificação):")
n_total_edif = len(gdf_edificacoes)
for cabo in CABOS_SENSIBILIDADE:
    if cabo == CABO_MAX_M:
        c_min, c_max = ASSOC["minimo"], ASSOC["maximo"]
    else:
        c_min = rodar_associacao("minimo", cabo, verbose=False)
        c_max = rodar_associacao("maximo", cabo, verbose=False)
    s_min = n_total_edif - c_min["id_casa"].nunique()
    s_max = n_total_edif - c_max["id_casa"].nunique()
    print(f"     cabo {cabo:>3.0f} m: mínimo {s_min:,} ({s_min/n_total_edif*100:.1f}%) | "
          f"máximo {s_max:,} ({s_max/n_total_edif*100:.1f}%)"
          + ("  <- atual" if cabo == CABO_MAX_M else ""))

# =============================================================================
# PASSO 5: CONSOLIDAR + DESFAZER MESCLAGEM + CLASSIFICAR (GALPÕES E ANEXOS)
# =============================================================================
def consolidar_e_classificar(cruz, nome):
    print(f"\n5. [{nome}] Consolidando e classificando...")
    cruzamento = gdf_ucs.merge(cruz, on="Código ID", how="inner")

    agrupado = cruzamento.groupby("id_casa").agg(
        qtd_ucs       = ("Código ID",      "count"),
        consumo_total = ("Consumo Mensal", "sum"),
        classes_ucs   = ("Classe", lambda x: ", ".join(x.astype(str).unique()))
    ).reset_index()

    gdf_f = gdf_edificacoes.merge(agrupado, on="id_casa", how="left")
    gdf_f["qtd_ucs"]       = gdf_f["qtd_ucs"].fillna(0).astype(int)
    gdf_f["consumo_total"] = gdf_f["consumo_total"].fillna(0)
    gdf_f["classes_ucs"]   = gdf_f["classes_ucs"].fillna("SEM MEDIDOR")
    gdf_f = classificar_inicial(gdf_f)

    # Desfaz a mesclagem dos grupos residenciais sem medidor com mais de 1 polígono
    mask_desfazer = (gdf_f["tipo"] == TIPO_RES) & (gdf_f["n_edificacoes"] > 1)
    gdf_desfazer = gdf_f[mask_desfazer]
    gdf_resto    = gdf_f[~mask_desfazer].copy()
    print(f"   Grupos sem medidor: galpão/comércio {(gdf_resto['tipo'] == TIPO_NAO_R).sum():,} | "
          f"mesclagem desfeita {len(gdf_desfazer):,}")

    orig = gdf_poligonos_orig[
        gdf_poligonos_orig["grupo_id"].isin(gdf_desfazer["grupo_id"])
    ].reset_index(drop=True)

    if not orig.empty:
        proximo_id = int(gdf_f["id_casa"].max()) + 1
        gdf_novos = gpd.GeoDataFrame(
            {
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
            },
            geometry="centroide", crs="EPSG:31983",
        )
        cent_wgs = gdf_novos.to_crs(epsg=4326)
        gdf_novos["lat_centroide"] = cent_wgs.geometry.y.round(6).values
        gdf_novos["lon_centroide"] = cent_wgs.geometry.x.round(6).values
        gdf_novos["retangularidade"] = retangularidade(gdf_novos["poligono"]).round(3)
        gdf_novos = classificar_inicial(gdf_novos)
        print(f"   Polígonos originais recuperados: {len(gdf_novos):,} "
              f"(galpão/comércio isolado: {(gdf_novos['tipo'] == TIPO_NAO_R).sum():,})")
        gdf_f = gpd.GeoDataFrame(
            pd.concat([gdf_resto, gdf_novos], ignore_index=True),
            geometry="centroide", crs="EPSG:31983"
        )
    else:
        gdf_f = gdf_resto.copy()

    # Anexos / edículas no mesmo lote
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

        filtro_anexo = (j_anexos["dist_m"] <= DIST_MAX_ANEXO_M) & \
                       (j_anexos["area_casa_m2_sus"] <= AREA_MAX_ANEXO_M2)
        anexos_match = j_anexos[filtro_anexo]

        id_para_anexo   = dict(zip(anexos_match["id_casa_sus"], anexos_match["id_casa_com"]))
        dist_para_anexo = dict(zip(anexos_match["id_casa_sus"], anexos_match["dist_m"]))

        mask_anexos_final = gdf_f["id_casa"].isin(anexos_match["id_casa_sus"])
        gdf_f.loc[mask_anexos_final, "tipo"] = TIPO_ANEXO
        gdf_f.loc[mask_anexos_final, "id_casa_principal"] = gdf_f.loc[mask_anexos_final, "id_casa"].map(id_para_anexo)
        gdf_f.loc[mask_anexos_final, "dist_medidor_m"] = gdf_f.loc[mask_anexos_final, "id_casa"].map(dist_para_anexo).round(2)

        print(f"   Anexos/edículas (<= {DIST_MAX_ANEXO_M:.0f} m, <= {AREA_MAX_ANEXO_M2:.0f} m²): "
              f"{mask_anexos_final.sum():,} ({mask_anexos_final.sum()/max(len(gdf_sus), 1)*100:.1f}% das suspeitas)")
    print(f"   Suspeitas residenciais restantes: {(gdf_f['tipo'] == TIPO_RES).sum():,}")
    return gdf_f

# =============================================================================
# PASSO 5.6: APLICAÇÃO DOS PERFIS DE CONSUMO
# =============================================================================
def aplicar_perfis(gdf_f):
    mask_res     = gdf_f["tipo"] == TIPO_RES
    mask_nao_res = gdf_f["tipo"] == TIPO_NAO_R
    mask_anexo   = gdf_f["tipo"] == TIPO_ANEXO

    for chave, info in PERFIS.items():
        col_kwh, col_rs = f"est_{chave}", f"valor_{chave}"
        gdf_f[col_kwh] = 0.0
        gdf_f[col_rs]  = 0.0
        if info["tipo"] == "area":
            # P1: baseado na área (residencial suspeita e galpões)
            gdf_f.loc[mask_res, col_kwh]     = gdf_f.loc[mask_res, "area_casa_m2"] * KWH_POR_M2_MES
            gdf_f.loc[mask_nao_res, col_kwh] = gdf_f.loc[mask_nao_res, "area_casa_m2"] * KWH_POR_M2_MES
        else:
            # P2 a P6: consumo domiciliar por edificação (SÓ residências suspeitas)
            gdf_f.loc[mask_res, col_kwh] = info["kwh"]
        gdf_f.loc[mask_res | mask_nao_res, col_rs] = gdf_f.loc[mask_res | mask_nao_res, col_kwh] * PRECO_KWH

    g_res, g_anexo, g_nao_res = (gdf_f[mask_res].copy(), gdf_f[mask_anexo].copy(),
                                 gdf_f[mask_nao_res].copy())
    linhas = []
    for chave, info in PERFIS.items():
        kwh_mes = g_res[f"est_{chave}"].sum()
        rs_mes  = g_res[f"valor_{chave}"].sum()
        kwh_ind = info["kwh"] if info["kwh"] is not None else (kwh_mes / len(g_res) if len(g_res) else 0.0)
        linhas.append({
            "Perfil": info["nome"],
            "kWh/mês por edificação": round(kwh_ind, 2),
            "Edificações suspeitas": len(g_res),
            "Perda kWh/mês": round(kwh_mes, 0),
            "Perda R$/mês": round(rs_mes, 2),
            "Perda kWh/ano (x12)": round(kwh_mes * 12, 0),
            "Perda R$/ano (x12)": round(rs_mes * 12, 2),
        })
    return g_res, g_anexo, g_nao_res, pd.DataFrame(linhas)

# =============================================================================
# DIAGNÓSTICO (para avaliar se o resultado é plausível)
# =============================================================================
def diagnostico(gdf_f, cruz, nome):
    res = gdf_f[gdf_f["tipo"] == TIPO_RES]
    com = gdf_f["qtd_ucs"] > 0
    print(f"\n   [{nome}] Diagnóstico")
    print(f"     UCs associadas: {len(cruz):,} de {len(gdf_ucs):,} | edificações com medidor: {int(com.sum()):,} "
          f"(média {gdf_f.loc[com, 'qtd_ucs'].mean():.2f} UCs; com 2+ UCs: {int((gdf_f['qtd_ucs'] > 1).sum()):,})")
    print(f"     Suspeitas residenciais: {len(res):,} ({len(res)/len(gdf_f)*100:.1f}% das edificações) "
          f"| referência da dissertação: ~0,4% dos medidores")
    faixas = pd.cut(res["area_casa_m2"], [0, 60, 80, 150, 1e9],
                    labels=["<60 m²", "60-80 m²", "80-150 m²", ">150 m²"])
    print("     Suspeitas por área: " +
          " | ".join(f"{k}: {v:,}" for k, v in faixas.value_counts().sort_index().items()))
    d_centro = gdf_f.geometry.distance(centro_metros)
    for a, b in [(0, 1000), (1000, 2000), (2000, RAIO_M + 1)]:
        anel = (d_centro >= a) & (d_centro < b)
        n_t, n_s = int(anel.sum()), int((anel & (gdf_f["tipo"] == TIPO_RES)).sum())
        n_u = int(((gdf_ucs["dist_centro"] >= a) & (gdf_ucs["dist_centro"] < b)).sum())
        print(f"     Anel {a/1000:.0f}-{min(b, RAIO_M)/1000:.0f} km: {n_t:,} edificações | "
              f"{n_u:,} UCs | {n_s:,} suspeitas ({n_s/max(n_t, 1)*100:.1f}%)")

RESULTADOS = {}
for cen, cruz in ASSOC.items():
    g = consolidar_e_classificar(cruz, cen)
    g_res, g_anexo, g_nao_res, df_res = aplicar_perfis(g)
    diagnostico(g, cruz, cen)
    RESULTADOS[cen] = {"gdf": g, "res": g_res, "anexo": g_anexo, "nao_res": g_nao_res,
                       "resumo": df_res, "cruz": cruz}

# ── Faixa de estimativa (mínimo x máximo) ────────────────────────────────────
r_min, r_max = RESULTADOS["minimo"]["resumo"], RESULTADOS["maximo"]["resumo"]
df_faixa = pd.DataFrame({
    "Perfil":                 r_min["Perfil"],
    "Suspeitas (mínimo)":     r_min["Edificações suspeitas"],
    "Suspeitas (máximo)":     r_max["Edificações suspeitas"],
    "Perda kWh/mês (mínimo)": r_min["Perda kWh/mês"],
    "Perda kWh/mês (máximo)": r_max["Perda kWh/mês"],
    "Perda R$/mês (mínimo)":  r_min["Perda R$/mês"],
    "Perda R$/mês (máximo)":  r_max["Perda R$/mês"],
    "Perda R$/ano (mínimo)":  r_min["Perda R$/ano (x12)"],
    "Perda R$/ano (máximo)":  r_max["Perda R$/ano (x12)"],
})
print("\n5.6. FAIXA DE ESTIMATIVA DE PERDA (mínimo = casamento ótimo | máximo = UC mais próxima):")
for _, r in df_faixa.iterrows():
    marca = "  <- principal" if r["Perfil"] == PERFIS[PERFIL_PRINCIPAL]["nome"] else ""
    print(f"   {r['Perfil']:<40}: {r['Perda kWh/mês (mínimo)']:>11,.0f} a {r['Perda kWh/mês (máximo)']:>11,.0f} kWh/mês"
          f" | R$ {r['Perda R$/mês (mínimo)']:>11,.0f} a {r['Perda R$/mês (máximo)']:>11,.0f}{marca}")
print(f"   Suspeitas residenciais: {len(RESULTADOS['minimo']['res']):,} (mínimo) a "
      f"{len(RESULTADOS['maximo']['res']):,} (máximo)")

# ── Cenário que alimenta as planilhas e o mapa ───────────────────────────────
_p = RESULTADOS[CENARIO_PRINCIPAL]
gdf_final, gdf_res, gdf_anexo, gdf_nao_res = _p["gdf"], _p["res"], _p["anexo"], _p["nao_res"]
df_resumo, cruzamento_ucs = _p["resumo"], _p["cruz"]
print(f"\n   Planilhas e mapa gerados com o cenário: {CENARIO_PRINCIPAL.upper()}")

print(f"\n   NÃO RESIDENCIAIS / GALPÕES ({len(gdf_nao_res):,} edificações — verificar Grupo A):")
print(f"   P1 Científico (1,5 W/m²)                : {gdf_nao_res['est_p1_cientifico'].sum():>12,.0f} kWh/mês | "
      f"R$ {gdf_nao_res['valor_p1_cientifico'].sum():>12,.2f}")
print(f"\n   ANEXOS / EDÍCULAS NO MESMO LOTE ({len(gdf_anexo):,} edificações):")
print(f"   Atendidos pela medição da edificação principal (não considerados furto).")

# =============================================================================
# PASSO 6: EXPORTAR PLANILHAS
# =============================================================================
print("\n6. Exportando planilhas...")
ucs_casadas_ids = cruzamento_ucs["Código ID"].unique()
gdf_ucs_orfas   = gdf_ucs[~gdf_ucs["Código ID"].isin(ucs_casadas_ids)]

colunas_uc = [c for c in df_ucs.columns if c != "geometry"]

colunas_base = [
    "id_casa", "tipo", "area_casa_m2", "retangularidade", "n_edificacoes",
    "raio_grupo_m", "lat_centroide", "lon_centroide"
]
colunas_est = []
for chave in PERFIS:
    colunas_est += [f"est_{chave}", f"valor_{chave}"]

with pd.ExcelWriter(ARQ_SEM_MEDIDOR, engine="openpyxl") as w:
    gdf_res[colunas_base + colunas_est].to_excel(w, sheet_name="suspeitas_residenciais", index=False)
    df_resumo.to_excel(w, sheet_name="resumo_perdas", index=False)
    gdf_anexo[colunas_base + ["id_casa_principal", "dist_medidor_m"]].to_excel(w, sheet_name="anexos_edículas", index=False)
    gdf_nao_res[colunas_base + ["est_p1_cientifico", "valor_p1_cientifico"]].to_excel(w, sheet_name="galpoes_nao_residenciais", index=False)
    df_faixa.to_excel(w, sheet_name="faixa_min_max", index=False)

print(f"   ✓ {ARQ_SEM_MEDIDOR} (abas: suspeitas_residenciais, resumo_perdas, anexos_edículas, galpoes_nao_residenciais, faixa_min_max)")

gdf_ucs_orfas.to_crs(epsg=4326)[colunas_uc].to_excel(ARQ_ORFAS, index=False)
print(f"   ✓ {ARQ_ORFAS} ({len(gdf_ucs_orfas):,} órfãs)")

# =============================================================================
# PASSO 7: MAPA
# =============================================================================
print("\n7. Gerando mapa...")

gdf_poligonos = gdf_final.copy()
gdf_poligonos = gdf_poligonos.set_geometry("poligono").set_crs(epsg=31983).to_crs(epsg=4326)

gdf_plot = gdf_final.copy()
gdf_plot = gdf_plot.set_geometry("centroide").set_crs(epsg=31983).to_crs(epsg=4326)

def cor_marcador(row):
    if row["tipo"] == TIPO_NAO_R:
        return "purple"
    if row["tipo"] == TIPO_ANEXO:
        return "gray"
    if row["qtd_ucs"] == 0:
        return "red"
    if row["qtd_ucs"] == 1:
        return "green"
    return "blue"

mapa = folium.Map(location=[LAT_CENTRO, LON_CENTRO], zoom_start=14, tiles=None)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satélite", overlay=False, control=True,
).add_to(mapa)
folium.TileLayer("CartoDB positron", name="Mapa de Ruas",
                 overlay=False, control=True).add_to(mapa)

# ── CAMADA 1: Polígonos reais ────────────────────────────────────────────────
grupo_poligonos = folium.FeatureGroup(name="Polígonos das Edificações", show=True)

gdf_para_mapa = gdf_poligonos.drop(columns=["centroide"])
gdf_para_mapa["poligono"] = gdf_para_mapa["poligono"].simplify(0.00001, preserve_topology=True) 

folium.GeoJson(
    gdf_para_mapa,
    style_function=lambda _: {
        "fillColor":   "#555555",
        "color":       "#222222",
        "weight":      1.0,
        "fillOpacity": 0.15,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["id_casa", "tipo", "area_casa_m2", "n_edificacoes",
                "lat_centroide", "lon_centroide", "qtd_ucs"],
        aliases=["ID:", "Classificação:", "Área (m²):", "Polígonos no grupo:",
                 "Lat centroide:", "Lon centroide:", "Medidores:"],
        localize=True,
        sticky=True,
    ),
).add_to(grupo_poligonos)
grupo_poligonos.add_to(mapa)

# ── CAMADA 2: Cluster de marcadores ─────────────────────────────────────────
grupo_cluster = folium.FeatureGroup(name="Edificações (cluster)", show=True)
cluster = MarkerCluster().add_to(grupo_cluster)

for _, row in gdf_plot.iterrows():
    if row["tipo"] == TIPO_NAO_R:
        texto_consumo = (
            f"<b style='color:purple;'>Possível galpão/comércio</b><br>"
            f"<i>Retangularidade: {row['retangularidade']:.2f} — verificar cliente Grupo A</i><br>"
            f" ↳ P1 (1,5 W/m²): <b>{row['est_p1_cientifico']:.1f} kWh</b> "
            f"(<span style='color:#cc0000;'>R$ {row['valor_p1_cientifico']:.2f}</span>)"
        )
    elif row["tipo"] == TIPO_ANEXO:
        dist_str = f"{row['dist_medidor_m']:.1f}m" if pd.notna(row['dist_medidor_m']) else "próximo"
        id_princ = int(row['id_casa_principal']) if pd.notna(row['id_casa_principal']) else "-"
        texto_consumo = (
            f"<b style='color:#555;'>Anexo / Edícula no mesmo lote</b><br>"
            f"<i>Atendido pela casa com medidor (ID {id_princ}, a {dist_str})</i><br>"
            f"<span style='color:#2e7d32;'>Não contabilizado como furto</span>"
        )
    elif row["qtd_ucs"] == 0:
        linhas_est = "".join([
            f" ↳ {info['nome']}: <b>{row[f'est_{k}']:.0f} kWh</b> "
            f"(<span style='color:#cc0000;'>R$ {row[f'valor_{k}']:.2f}</span>)<br>"
            for k, info in PERFIS.items()
        ])
        texto_consumo = f"<b style='color:red;'>Estimativa de Furto/Mês:</b><br>{linhas_est}"
    else:
        texto_consumo = f"<b style='color:green;'>Consumo Medido (Real):</b> {row['consumo_total']:.0f} kWh"

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
            f"{texto_consumo}<br>"
            f"<hr style='margin:4px 0;'>"
            f"<b>Perfil UCs:</b> {row['classes_ucs']}"
        ),
    ).add_to(cluster)
grupo_cluster.add_to(mapa)

# ── CAMADA 3: UCs órfãs ──────────────────────────────────────────────────────
grupo_orfas = folium.FeatureGroup(name="UCs Órfãs", show=False)
for _, row in gdf_ucs_orfas.to_crs(epsg=4326).iterrows():
    folium.CircleMarker(
        location=[row["geometry"].y, row["geometry"].x],
        radius=3, color="black", fill=True, fill_opacity=1.0,
        tooltip=f"Órfã: {row['Código ID']} | {row['Classe']}",
    ).add_to(grupo_orfas)
grupo_orfas.add_to(mapa)

# ── Círculo azul da área de estudo ───────────────────────────────────────────
folium.Circle(
    location=[LAT_CENTRO, LON_CENTRO],
    radius=RAIO_M,
    color="#0055ff",        
    weight=2.5,
    fill=False,
    dash_array="8",
    tooltip=f"Área de estudo: {RAIO_M/1000:.0f} km de raio",
).add_to(mapa)

# Marcador central
folium.Marker(
    location=[LAT_CENTRO, LON_CENTRO],
    tooltip="Centro de pesquisa",
    popup=folium.Popup(f"""
        <div style='text-align:center; width:220px; font-family:Arial;'>
            <b>Campo Belo — M1</b><br>
            Raio: {RAIO_M/1000:.0f} km<br><br>
            Total edificações: {len(gdf_final):,}<br>
            🟢 Com medidor: {(gdf_final['qtd_ucs']>0).sum():,}<br>
            ⚪ Anexos / Edículas: {len(gdf_anexo):,}<br>
            🔴 Suspeitas residenciais: {len(gdf_res):,}<br>
            🟣 Galpões / Comércios: {len(gdf_nao_res):,}<br><br>
            UCs no raio: {len(gdf_ucs):,}<br>
            UCs órfãs: {len(gdf_ucs_orfas):,}
        </div>""", max_width=320),
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(mapa)

# Legenda
legenda_html = f"""
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:15px; border-radius:8px;
     border:2px solid #ccc; font-family:Arial; font-size:13px; max-height:480px; overflow-y:auto;">
  <b>Metodologia 1 (Múltiplas UCs)</b><br>
  <i style="font-size:11px;">Cenário: {CENARIO_PRINCIPAL}</i><br>
  <i style="font-size:11px;">conf≥{CONFIANCA_MIN} | área≥{AREA_MIN_M2:.0f}m² | cabo≤{CABO_MAX_M:.0f}m</i><br><br>
  <span style="background:red;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;<b>Suspeita residencial ({len(gdf_res):,})</b><br><br>
  <span style="background:gray;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;Anexo / Edícula ({len(gdf_anexo):,})<br><br>
  <span style="background:purple;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;Galpão / Comércio ({len(gdf_nao_res):,})<br><br>
  <span style="background:green;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;1 medidor ({(gdf_final['qtd_ucs']==1).sum():,})<br><br>
  <span style="background:blue;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;2+ medidores ({(gdf_final['qtd_ucs']>1).sum():,})<br><br>
  <span style="background:black;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;UC órfã ({len(gdf_ucs_orfas):,})<br><br>
  <hr style="margin:6px 0;">
  <i>Total: {len(gdf_final):,} edificações</i>
</div>
"""
mapa.get_root().html.add_child(folium.Element(legenda_html))

# =============================================================================
# PASSO EXTRA: VERIFICAÇÃO MANUAL DE ANOMALIAS
# =============================================================================
print(" -> Adicionando camada de verificação manual (Vermelho/Laranja)...")

gdf_verificacao = gdf_para_mapa[
    (gdf_para_mapa["n_edificacoes"] > 20) | (gdf_para_mapa["area_casa_m2"] > 350.0)
].copy()

if not gdf_verificacao.empty:
    grupo_verificacao = folium.FeatureGroup(name="Verificação Manual (Anomalias)", show=True)

    def estilo_verificacao(feature):
        n_poli = feature["properties"]["n_edificacoes"]
        area = feature["properties"]["area_casa_m2"]

        if n_poli > 20:
            return {"fillColor": "#ff0000", "color": "#cc0000", "weight": 3.0, "fillOpacity": 0.75}
        elif area > 350.0:
            return {"fillColor": "#ff8800", "color": "#cc6600", "weight": 3.0, "fillOpacity": 0.75}
        return {}

    folium.GeoJson(
        gdf_verificacao,
        style_function=estilo_verificacao,
        tooltip=folium.GeoJsonTooltip(
            fields=["id_casa", "area_casa_m2", "n_edificacoes", "qtd_ucs"],
            aliases=["ID:", "Área (m²):", "Qtd. Polígonos fundidos:", "Medidores:"],
            localize=True,
            sticky=True,
        )
    ).add_to(grupo_verificacao)

    grupo_verificacao.add_to(mapa)

folium.LayerControl(position="topright", collapsed=False).add_to(mapa)

titulo_html = f'''
<h3 align="center" style="font-size:16px; font-family:Arial;">
    <b>M1 (Múltiplas UCs) | {len(gdf_final):,} edificações |
    Com medidor: {(gdf_final["qtd_ucs"]>0).sum():,} |
    Anexos/Edículas: {len(gdf_anexo):,} |
    Suspeitas Residenciais: {len(gdf_res):,} |
    Galpões: {len(gdf_nao_res):,}</b>
</h3>'''
mapa.get_root().html.add_child(folium.Element(titulo_html))

mapa.save(ARQ_SAIDA_MAPA)
print(f"\n✓ Concluído com sucesso:")
print(f"  → {ARQ_SAIDA_MAPA}")
print(f"  → {ARQ_SEM_MEDIDOR}")
print(f"  → {ARQ_ORFAS}")
