import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import unary_union
import folium
from folium.plugins import MarkerCluster
import gc
import math
import random
import numpy as np

try:  # casamento ótimo (recomendado): pip install scipy
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching
    TEM_SCIPY = True
except ImportError:
    TEM_SCIPY = False

random.seed(42)  # reprodutibilidade (Welzl embaralha os pontos)

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
CABO_MAX_M      = 30.0     # a dissertação usa 45 m; para mudar, altere só aqui
CHUNKSIZE       = 200_000

# --- Divisão de grupos grandes (Passo 3.5, ANTES da associação) --------------
LIMIAR_DIVISAO_M2  = 150.0  # grupos acima disso são fatiados
AREA_ALVO_FIXA     = None   # None = calibra pela mediana dos grupos com 1 UC
AREA_ALVO_MIN_M2   = 50.0   # piso para a área-alvo calibrada

# --- Anomalias: NÃO são divididas e NÃO entram na estimativa ------------------
MAX_POLIGONOS_GRUPO = 20     # muitos polígonos fundidos (encadeamento)
AREA_MAX_GRUPO_M2   = 350.0  # área grande demais (galpão, escola, igreja...)

# --- Associação UC -> edificação (Passo 4) ------------------------------------
PENALIDADE_SEM_UC   = 10_000.0        # custo de deixar uma edificação sem UC no casamento
CABOS_SENSIBILIDADE = [20, 30, 45, 60]  # só imprime o nº de edificações sem UC em cada cabo

# --- Anexos (OPCIONAL; None = desligado) --------------------------------------
# Suspeita pequena muito perto de edificação COM medidor = provável anexo/garagem
ANEXO_DIST_M      = None   # ex.: 8.0
ANEXO_AREA_MAX_M2 = 80.0

# --- Classes consideradas residenciais ----------------------------------------
CLASSES_RES = ["RE1", "RE2"]

# --- Preço e perfis de consumo (kWh/mês por edificação sem medidor) -----------
PRECO_KWH = 0.95  # R$/kWh (CPFL Paulista - Campinas)

CONSUMO_ARTIGO_VERAO   = 460.0                  # P2: artigo (região parecida)
CONSUMO_ARTIGO_INVERNO = 190.0                  # P2: artigo
CONSUMO_PROPRIO        = 105.77                 # P3: perfil por equipamentos
TABELA2_RES_SUDESTE_ANO       = 2088.7          # P4: Tabela 2, kWh/UH.ano
TABELA2_BAIXA_RENDA_SUDESTE_ANO = 1843.9        # P5: Tabela 2, kWh/UH.ano
MEDIANA_BAIRRO_SEM_ZERO = 167.0                 # P6: mediana das UCs do bairro, sem zeros
# MEDIANA_BAIRRO_COM_ZERO = 126.0               # alternativa (inclui UCs zeradas)

PERFIS = {
    "p2_verao":    {"nome": "P2 Artigo (Verão)",                   "kwh": CONSUMO_ARTIGO_VERAO},
    "p2_inverno":  {"nome": "P2 Artigo (Inverno)",                 "kwh": CONSUMO_ARTIGO_INVERNO},
    "p3_proprio":  {"nome": "P3 Perfil próprio (equipamentos)",    "kwh": CONSUMO_PROPRIO},
    "p4_tab_res":  {"nome": "P4 Tabela 2 - Residencial Sudeste",   "kwh": TABELA2_RES_SUDESTE_ANO / 12},
    "p5_tab_br":   {"nome": "P5 Tabela 2 - Baixa renda Sudeste",   "kwh": TABELA2_BAIXA_RENDA_SUDESTE_ANO / 12},
    "p6_mediana":  {"nome": "P6 Mediana UCs do bairro (s/ zeros)", "kwh": MEDIANA_BAIRRO_SEM_ZERO},
}
PERFIL_PRINCIPAL = "p6_mediana"  # cenário central sugerido (dado local)

# =============================================================================
# ALGORITMO DE WELZL (versão iterativa: sem limite de recursão do Python)
# =============================================================================
def _circulo_de_3(p1, p2, p3):
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
    if abs(d) < 1e-10: return None
    ux = ((ax**2+ay**2)*(by-cy)+(bx**2+by**2)*(cy-ay)+(cx**2+cy**2)*(ay-by))/d
    uy = ((ax**2+ay**2)*(cx-bx)+(bx**2+by**2)*(ax-cx)+(cx**2+cy**2)*(bx-ax))/d
    return (ux, uy, math.sqrt((ax-ux)**2+(ay-uy)**2))

def _circulo_de_2(p1, p2):
    cx=(p1[0]+p2[0])/2; cy=(p1[1]+p2[1])/2
    return (cx, cy, math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)/2)

def _ponto_no_circulo(c, p, eps=1e-7):
    return math.sqrt((p[0]-c[0])**2+(p[1]-c[1])**2) <= c[2]+eps

def _circulo_de_3_seguro(p, q, r):
    c = _circulo_de_3(p, q, r)
    if c is not None: return c
    # pontos colineares: usa o círculo dos dois mais distantes
    cands = [_circulo_de_2(p, q), _circulo_de_2(p, r), _circulo_de_2(q, r)]
    return max(cands, key=lambda c: c[2])

def menor_circulo_welzl(pontos):
    if not pontos: return None
    P = list(pontos); random.shuffle(P)
    c = (P[0][0], P[0][1], 0.0)
    for i in range(1, len(P)):
        if not _ponto_no_circulo(c, P[i]):
            c = (P[i][0], P[i][1], 0.0)
            for j in range(i):
                if not _ponto_no_circulo(c, P[j]):
                    c = _circulo_de_2(P[i], P[j])
                    for k in range(j):
                        if not _ponto_no_circulo(c, P[k]):
                            c = _circulo_de_3_seguro(P[i], P[j], P[k])
    return c

# =============================================================================
# DIVISÃO DE POLÍGONO EM N PARTES DE ÁREA PARECIDA
# =============================================================================
def _so_poligonos(g):
    if g is None or g.is_empty: return None
    if g.geom_type in ("Polygon", "MultiPolygon"): return g
    pol = [x for x in getattr(g, "geoms", []) if x.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(pol) if pol else None

def dividir_em_n(poligono, n):
    """Corta o polígono em n partes de área parecida (cortes retos, na maior dimensão)."""
    if n <= 1: return [poligono]
    minx, miny, maxx, maxy = poligono.bounds
    n1 = n // 2
    alvo = poligono.area * (n1 / n)
    horizontal = (maxx - minx) >= (maxy - miny)

    def metade(corte):
        if horizontal: return box(minx-1, miny-1, corte, maxy+1)
        return box(minx-1, miny-1, maxx+1, corte)

    lo, hi = (minx, maxx) if horizontal else (miny, maxy)
    for _ in range(25):  # bisseção para achar o corte que dá a área desejada
        meio = (lo + hi) / 2
        if poligono.intersection(metade(meio)).area < alvo: lo = meio
        else: hi = meio
    b1 = metade((lo + hi) / 2)
    p1 = _so_poligonos(poligono.intersection(b1))
    p2 = _so_poligonos(poligono.difference(b1))

    partes = []
    if p1 is not None: partes += dividir_em_n(p1, n1)
    if p2 is not None: partes += dividir_em_n(p2, n - n1)
    return [p for p in partes if p.area >= 1.0] or [poligono]

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
# PASSO 2: MESCLAGEM POR INTERSEÇÃO REAL
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

# representative_point() sempre cai DENTRO do polígono (o centroide pode cair fora
# em formas côncavas). O vínculo é feito por índice, sem depender da ordem das linhas.
pontos_rep = gpd.GeoDataFrame(
    {"idx_orig": gdf_m.index},
    geometry=gdf_m.geometry.representative_point(),
    crs="EPSG:31983"
)
join = gpd.sjoin(pontos_rep, grupos[["grupo_id", "geometry"]], how="left", predicate="intersects")
join = join.drop_duplicates(subset=["idx_orig"], keep="first")
gdf_m["grupo_id"] = gdf_m.index.map(join.set_index("idx_orig")["grupo_id"])

n_sem_grupo = int(gdf_m["grupo_id"].isna().sum())
if n_sem_grupo:
    print(f"   AVISO: {n_sem_grupo} polígonos sem grupo foram descartados.")
    gdf_m = gdf_m.dropna(subset=["grupo_id"])
gdf_m["grupo_id"] = gdf_m["grupo_id"].astype(int)

print("   Calculando centroides e polígonos por grupo...")
cent = gdf_m.geometry.centroid
gdf_m["cx"] = cent.x
gdf_m["cy"] = cent.y

grupos_info = []
for grupo_id, grupo in gdf_m.groupby("grupo_id"):
    pontos = list(zip(grupo["cx"].tolist(), grupo["cy"].tolist()))
    if len(pontos) == 1:
        cx, cy = pontos[0]; raio = 0.0
    else:
        cx, cy, raio = menor_circulo_welzl(pontos)

    grupos_info.append({
        "grupo_id":      grupo_id,
        "cx":            cx,
        "cy":            cy,
        "poligono":      unary_union(grupo.geometry.values),
        "area_casa_m2":  grupo["area_m2"].sum(),
        "n_edificacoes": len(grupo),
        "raio_grupo_m":  round(raio, 2),
    })

df_grupos = pd.DataFrame(grupos_info)
df_grupos["anomalia"] = (
    (df_grupos["n_edificacoes"] > MAX_POLIGONOS_GRUPO) |
    (df_grupos["area_casa_m2"] > AREA_MAX_GRUPO_M2)
)
print(f"   {len(df_grupos):,} grupos | {int(df_grupos['anomalia'].sum()):,} anomalias "
      f"(> {MAX_POLIGONOS_GRUPO} polígonos ou > {AREA_MAX_GRUPO_M2:.0f} m²).")

del gdf, gdf_m, pontos_rep, join, grupos, unido; gc.collect()

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

gdf_ucs = gdf_ucs_todas[gdf_ucs_todas["dist_centro"] <= RAIO_M].copy().reset_index(drop=True)
print(f"   {len(gdf_ucs):,} UCs no raio de {RAIO_M/1000:.0f}km.")

# UCs residenciais (RE1 + RE2) — usadas na calibração e na checagem do perfil P6
mask_res = gdf_ucs["Classe"].astype(str).str.upper().str.strip().isin(CLASSES_RES)
gdf_ucs_res = gdf_ucs[mask_res] if mask_res.any() else gdf_ucs
if not mask_res.any():
    print(f"   AVISO: nenhuma UC com classe em {CLASSES_RES}; usando todas as UCs.")

cons_res = gdf_ucs_res["Consumo Mensal"]
cons_pos = cons_res[cons_res > 0]
print(f"   Checagem do consumo das UCs residenciais ({len(cons_res):,} UCs):")
print(f"     média {cons_res.mean():.1f} | média s/0 {cons_pos.mean():.1f} | "
      f"mediana {cons_res.median():.1f} | mediana s/0 {cons_pos.median():.1f} kWh "
      f"| zeros: {(cons_res <= 0).mean()*100:.1f}%")
print(f"     (P6 está fixado em {MEDIANA_BAIRRO_SEM_ZERO:.0f} kWh)")

# =============================================================================
# PASSO 3.5: DIVISÃO DE GRUPOS GRANDES (ANTES DA ASSOCIAÇÃO)
# =============================================================================
print("\n3.5. Dividindo grupos grandes antes de associar as UCs...")

def calibrar_area_alvo():
    """Mediana da área dos grupos pequenos que têm exatamente 1 UC residencial."""
    gdf_pol = gpd.GeoDataFrame(
        df_grupos[["grupo_id"]], geometry=list(df_grupos["poligono"]), crs="EPSG:31983"
    )
    j = gpd.sjoin_nearest(
        gdf_ucs_res[["Código ID", "geometry"]], gdf_pol,
        how="inner", max_distance=CABO_MAX_M
    ).drop_duplicates(subset=["Código ID"])
    ucs_por_grupo = j.groupby("grupo_id")["Código ID"].count().rename("n_ucs")
    ref = df_grupos.set_index("grupo_id").join(ucs_por_grupo, how="inner")
    ref = ref[(ref["n_ucs"] == 1) & (ref["area_casa_m2"] <= LIMIAR_DIVISAO_M2) & (~ref["anomalia"])]
    if len(ref) < 30:
        print(f"   Poucos grupos de referência ({len(ref)}); usando o piso {AREA_ALVO_MIN_M2:.0f} m².")
        return AREA_ALVO_MIN_M2
    mediana = float(ref["area_casa_m2"].median())
    print(f"   Mediana da área de grupos com 1 UC: {mediana:.1f} m² ({len(ref):,} grupos)")
    return max(AREA_ALVO_MIN_M2, mediana)

AREA_ALVO = AREA_ALVO_FIXA if AREA_ALVO_FIXA else calibrar_area_alvo()
print(f"   Área-alvo por edificação: {AREA_ALVO:.1f} m² | limiar de divisão: {LIMIAR_DIVISAO_M2:.0f} m²")

registros = []
n_divididos = 0
for g in df_grupos.itertuples(index=False):
    n_partes = 1
    if (not g.anomalia) and g.area_casa_m2 > LIMIAR_DIVISAO_M2:
        n_partes = max(1, min(50, round(g.area_casa_m2 / AREA_ALVO)))

    if n_partes >= 2:
        partes = dividir_em_n(g.poligono, n_partes)
        n_divididos += 1
        for parte in partes:
            pt = parte.centroid
            if not parte.contains(pt):
                pt = parte.representative_point()
            registros.append({
                "grupo_id": g.grupo_id, "centroide": pt, "poligono": parte,
                "area_casa_m2": parte.area, "n_edificacoes": g.n_edificacoes,
                "raio_grupo_m": 0.0, "n_partes": len(partes), "dividida": True,
                "anomalia": False,
            })
    else:
        registros.append({
            "grupo_id": g.grupo_id, "centroide": Point(g.cx, g.cy), "poligono": g.poligono,
            "area_casa_m2": g.area_casa_m2, "n_edificacoes": g.n_edificacoes,
            "raio_grupo_m": g.raio_grupo_m, "n_partes": 1, "dividida": False,
            "anomalia": bool(g.anomalia),
        })

gdf_edificacoes = gpd.GeoDataFrame(registros, geometry="centroide", crs="EPSG:31983")
gdf_edificacoes["id_casa"] = range(1, len(gdf_edificacoes) + 1)

cent_wgs = gdf_edificacoes.to_crs(epsg=4326)
gdf_edificacoes["lat_centroide"] = cent_wgs.geometry.y.round(6).values
gdf_edificacoes["lon_centroide"] = cent_wgs.geometry.x.round(6).values

print(f"   {n_divididos:,} grupos divididos | {len(gdf_edificacoes):,} edificações no total.")
del registros, df_grupos, cent_wgs; gc.collect()

# =============================================================================
# PASSO 4: ASSOCIAÇÃO UC -> EDIFICAÇÃO
#   Fase A: casamento ótimo — cada edificação recebe NO MÁXIMO 1 UC, maximizando
#           o nº de edificações atendidas (como o problema de associação da dissertação)
#   Fase B: UCs restantes (prédios, kitnets, várias casas no lote) vão para a
#           edificação mais próxima
#   Fase R: resgate das que sobraram pelo polígono real
# =============================================================================
print(f"\n4. Associando UCs às edificações (cabo máx. {CABO_MAX_M} m)...")
if not TEM_SCIPY:
    print("   AVISO: scipy não encontrado; usando guloso na Fase A (pip install scipy).")

gdf_edif_poligono = gpd.GeoDataFrame(
    gdf_edificacoes[["id_casa"]].reset_index(drop=True),
    geometry=list(gdf_edificacoes["poligono"]), crs="EPSG:31983"
)

def pares_candidatos(df_ucs, df_edif, cabo_max):
    """Todos os pares (edificação, UC) a até cabo_max metros (centroide da edificação)."""
    edif_pts = gpd.GeoDataFrame(
        df_edif[["id_casa"]].reset_index(drop=True),
        geometry=list(df_edif["centroide"]), crs="EPSG:31983"
    )
    ucs_pts = df_ucs[["Código ID", "geometry"]].reset_index(drop=True)
    ucs_buf = ucs_pts.copy()
    ucs_buf["geometry"] = ucs_buf.geometry.buffer(cabo_max)

    pares = gpd.sjoin(edif_pts, ucs_buf, how="inner", predicate="within")
    pares = pares.rename(columns={"index_right": "idx_uc"})
    pares["distancia"] = pares.geometry.distance(
        ucs_pts.geometry.iloc[pares["idx_uc"].values], align=False
    )
    pares = pares[pares["distancia"] <= cabo_max]
    return (pares[["id_casa", "Código ID", "distancia"]]
            .drop_duplicates(subset=["id_casa", "Código ID"]).reset_index(drop=True))

def fase_a_casamento(pares):
    """Cada edificação recebe no máx. 1 UC; maximiza edificações atendidas e,
    entre as soluções, minimiza a distância total. Retorna {Código ID: id_casa}."""
    casas = pares["id_casa"].unique()
    ucs   = pares["Código ID"].unique()
    n, m  = len(casas), len(ucs)

    if TEM_SCIPY:
        i_casa = pd.Index(casas).get_indexer(pares["id_casa"])
        i_uc   = pd.Index(ucs).get_indexer(pares["Código ID"])
        # colunas: m UCs reais + n UCs fictícias (uma por edificação, custo alto)
        linhas = np.concatenate([i_casa, np.arange(n)])
        colunas = np.concatenate([i_uc, m + np.arange(n)])
        pesos = np.concatenate([pares["distancia"].to_numpy() + 0.01,
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

def associar_ucs(df_ucs, df_edif, cabo_max, verbose=True):
    pares = pares_candidatos(df_ucs, df_edif, cabo_max)
    if pares.empty:
        return pd.DataFrame(columns=["Código ID", "id_casa", "distancia", "fase"])

    atrib = fase_a_casamento(pares)
    fase_a = (pd.DataFrame({"Código ID": list(atrib.keys()), "id_casa": list(atrib.values())})
              .merge(pares, on=["Código ID", "id_casa"], how="left"))
    fase_a["fase"] = "A"

    resto = pares[~pares["Código ID"].isin(atrib.keys())]
    fase_b = (resto.sort_values("distancia")
              .drop_duplicates(subset=["Código ID"], keep="first").copy())
    fase_b["fase"] = "B"

    if verbose:
        print(f"   Fase A (1 UC por edificação) : {len(fase_a):,} edificações atendidas")
        print(f"   Fase B (UCs extras)          : {len(fase_b):,} UCs")
    return pd.concat([fase_a, fase_b], ignore_index=True)

def rodar_associacao(cabo_max, verbose=True):
    cruz = associar_ucs(gdf_ucs, gdf_edificacoes, cabo_max, verbose)

    orfas = gdf_ucs[~gdf_ucs["Código ID"].isin(cruz["Código ID"])]
    if not orfas.empty:
        resgate = gpd.sjoin_nearest(
            orfas, gdf_edif_poligono,
            how="inner", max_distance=cabo_max, distance_col="distancia"
        )
        resgate = (resgate.sort_values("distancia")
                   .drop_duplicates(subset=["Código ID"], keep="first"))
        resg = resgate[["Código ID", "id_casa", "distancia"]].copy()
        resg["fase"] = "R"
        cruz = pd.concat([cruz, resg], ignore_index=True)
        if verbose:
            print(f"   Fase R (resgate por polígono): {len(resg):,} UCs")
    return cruz.drop_duplicates(subset=["Código ID"], keep="first")

cruzamento_ucs = rodar_associacao(CABO_MAX_M)
print(f"   UCs associadas: {len(cruzamento_ucs):,} | órfãs: {len(gdf_ucs) - len(cruzamento_ucs):,}")
print(f"   Distância UC->edificação: mediana {cruzamento_ucs['distancia'].median():.1f} m | "
      f"média {cruzamento_ucs['distancia'].mean():.1f} m")

print("   Sensibilidade ao cabo máximo (edificações sem nenhuma UC):")
for cabo in CABOS_SENSIBILIDADE:
    cz = cruzamento_ucs if cabo == CABO_MAX_M else rodar_associacao(cabo, verbose=False)
    sem_uc = len(gdf_edificacoes) - cz["id_casa"].nunique()
    print(f"     cabo {cabo:>3.0f} m: {sem_uc:,} ({sem_uc/len(gdf_edificacoes)*100:.1f}%)"
          + ("  <- atual" if cabo == CABO_MAX_M else ""))

# =============================================================================
# PASSO 5: CONSOLIDAR EDIFICAÇÕES COM E SEM MEDIDOR
# =============================================================================
print("\n5. Consolidando resultado...")

cruzamento = gdf_ucs.merge(cruzamento_ucs, on="Código ID", how="inner")

agrupado = cruzamento.groupby("id_casa").agg(
    qtd_ucs       = ("Código ID",      "count"),
    consumo_total = ("Consumo Mensal", "sum"),
    classes_ucs   = ("Classe", lambda x: ", ".join(x.astype(str).unique()))
).reset_index()

gdf_final = gdf_edificacoes.merge(agrupado, on="id_casa", how="left")
gdf_final["qtd_ucs"]       = gdf_final["qtd_ucs"].fillna(0).astype(int)
gdf_final["consumo_total"] = gdf_final["consumo_total"].fillna(0)
gdf_final["classes_ucs"]   = gdf_final["classes_ucs"].fillna("SEM MEDIDOR")

# Distância de cada edificação sem medidor até a edificação COM medidor mais próxima
gdf_final["dist_vizinho_medidor"] = np.nan
_sem = gdf_final[gdf_final["qtd_ucs"] == 0]
_com = gdf_final[gdf_final["qtd_ucs"] > 0]
if len(_sem) and len(_com):
    _gs = gpd.GeoDataFrame({"id_casa": _sem["id_casa"].values},
                           geometry=list(_sem["poligono"]), crs="EPSG:31983")
    _gv = gpd.GeoDataFrame({"id_viz": _com["id_casa"].values},
                           geometry=list(_com["poligono"]), crs="EPSG:31983")
    _jn = (gpd.sjoin_nearest(_gs, _gv, how="left", max_distance=50, distance_col="d_viz")
           .drop_duplicates(subset=["id_casa"]))
    gdf_final["dist_vizinho_medidor"] = gdf_final["id_casa"].map(_jn.set_index("id_casa")["d_viz"])

# Anexos prováveis (só se ANEXO_DIST_M estiver definido)
if ANEXO_DIST_M is not None:
    gdf_final["anexo"] = (
        (gdf_final["qtd_ucs"] == 0) & (~gdf_final["anomalia"]) &
        (gdf_final["area_casa_m2"] <= ANEXO_AREA_MAX_M2) &
        (gdf_final["dist_vizinho_medidor"] <= ANEXO_DIST_M)
    )
else:
    gdf_final["anexo"] = False

mask_com_medidor = gdf_final["qtd_ucs"] > 0
mask_anomalia    = (gdf_final["qtd_ucs"] == 0) & (gdf_final["anomalia"])
mask_anexo       = (gdf_final["qtd_ucs"] == 0) & (~gdf_final["anomalia"]) & (gdf_final["anexo"])
mask_suspeita    = (gdf_final["qtd_ucs"] == 0) & (~gdf_final["anomalia"]) & (~gdf_final["anexo"])

print(f"   Com medidor: {int(mask_com_medidor.sum()):,} | "
      f"Suspeitas: {int(mask_suspeita.sum()):,} | "
      f"Anomalias sem medidor (fora da estimativa): {int(mask_anomalia.sum()):,} | "
      f"Anexos (fora da estimativa): {int(mask_anexo.sum()):,}")

# -----------------------------------------------------------------------------
# DIAGNÓSTICO DAS SUSPEITAS (para avaliar se o resultado é plausível)
# -----------------------------------------------------------------------------
print("\n5.5. Diagnóstico das suspeitas...")
susp = gdf_final[mask_suspeita]
n_com = int(mask_com_medidor.sum())
print(f"   UCs no raio: {len(gdf_ucs):,} | associadas: {len(cruzamento_ucs):,} | "
      f"edificações: {len(gdf_final):,}")
if n_com:
    print(f"   UCs por edificação com medidor: média {gdf_final.loc[mask_com_medidor, 'qtd_ucs'].mean():.2f} | "
          f"com 2+ UCs: {int((gdf_final['qtd_ucs'] > 1).sum()):,}")
print(f"   Suspeitas: {len(susp):,} ({len(susp)/len(gdf_final)*100:.1f}% das edificações) "
      f"| meta de comparação da dissertação: ~0,4% dos medidores")
print(f"   Suspeitas vindas de grupos divididos: {int(susp['dividida'].sum()):,}")

faixas = pd.cut(susp["area_casa_m2"], [0, 60, 80, 150, 1e9],
                labels=["<60 m²", "60-80 m²", "80-150 m²", ">150 m²"])
print("   Suspeitas por faixa de área: " +
      " | ".join(f"{k}: {v:,}" for k, v in faixas.value_counts().sort_index().items()))

d_centro = gdf_final.geometry.distance(centro_metros)
for a, b in [(0, 1000), (1000, 2000), (2000, RAIO_M + 1)]:
    em_anel = (d_centro >= a) & (d_centro < b)
    n_tot, n_s = int(em_anel.sum()), int((em_anel & mask_suspeita).sum())
    n_ucs_anel = int(((gdf_ucs["dist_centro"] >= a) & (gdf_ucs["dist_centro"] < b)).sum())
    print(f"   Anel {a/1000:.0f}-{min(b, RAIO_M)/1000:.0f} km: {n_tot:,} edificações | "
          f"{n_ucs_anel:,} UCs | {n_s:,} suspeitas ({n_s/max(n_tot,1)*100:.1f}%)")

dv = susp["dist_vizinho_medidor"]
print(f"   Suspeitas a <=5 m de edificação com medidor: {int((dv <= 5).sum()):,} | "
      f"a <=10 m: {int((dv <= 10).sum()):,} (prováveis anexos; ver ANEXO_DIST_M)")

# =============================================================================
# PASSO 5.6: APLICAÇÃO DOS PERFIS DE CONSUMO PARA AS SUSPEITAS
# =============================================================================
print("\n5.6. Aplicando os perfis de consumo às edificações suspeitas...")

for chave, info in PERFIS.items():
    col_kwh, col_rs = f"est_{chave}", f"valor_{chave}"
    gdf_final[col_kwh] = 0.0
    gdf_final.loc[mask_suspeita, col_kwh] = info["kwh"]  # valor fixo por edificação
    gdf_final[col_rs] = gdf_final[col_kwh] * PRECO_KWH

gdf_sem_medidor = gdf_final[mask_suspeita].copy()
gdf_anomalias   = gdf_final[mask_anomalia].copy()
gdf_anexos      = gdf_final[mask_anexo].copy()

linhas_resumo = []
for chave, info in PERFIS.items():
    kwh_mes = gdf_sem_medidor[f"est_{chave}"].sum()
    rs_mes  = gdf_sem_medidor[f"valor_{chave}"].sum()
    linhas_resumo.append({
        "Perfil": info["nome"],
        "kWh/mês por edificação": round(info["kwh"], 2),
        "Edificações suspeitas": len(gdf_sem_medidor),
        "Perda kWh/mês": round(kwh_mes, 0),
        "Perda R$/mês": round(rs_mes, 2),
        "Perda kWh/ano (x12)": round(kwh_mes * 12, 0),
        "Perda R$/ano (x12)": round(rs_mes * 12, 2),
    })
    marca = "  <- principal" if chave == PERFIL_PRINCIPAL else ""
    print(f"   {info['nome']:<40}: {kwh_mes:>12,.0f} kWh/mês | R$ {rs_mes:>12,.2f}{marca}")
df_resumo = pd.DataFrame(linhas_resumo)

# =============================================================================
# PASSO 6: EXPORTAR PLANILHAS
# =============================================================================
print("\n6. Exportando planilhas...")
ucs_casadas_ids = cruzamento_ucs["Código ID"].unique()
gdf_ucs_orfas   = gdf_ucs[~gdf_ucs["Código ID"].isin(ucs_casadas_ids)]

colunas_uc = [c for c in df_ucs.columns if c != "geometry"]

colunas_base = ["id_casa", "area_casa_m2", "n_edificacoes", "n_partes", "dividida",
                "raio_grupo_m", "lat_centroide", "lon_centroide", "dist_vizinho_medidor"]
colunas_est = []
for chave in PERFIS:
    colunas_est += [f"est_{chave}", f"valor_{chave}"]

with pd.ExcelWriter(ARQ_SEM_MEDIDOR, engine="openpyxl") as w:
    gdf_sem_medidor[colunas_base + colunas_est].to_excel(w, sheet_name="suspeitas", index=False)
    df_resumo.to_excel(w, sheet_name="resumo", index=False)
    gdf_anomalias[colunas_base].to_excel(w, sheet_name="anomalias_excluidas", index=False)
    if ANEXO_DIST_M is not None:
        gdf_anexos[colunas_base].to_excel(w, sheet_name="anexos_excluidos", index=False)
print(f"   ✓ {ARQ_SEM_MEDIDOR} ({len(gdf_sem_medidor):,} suspeitas | "
      f"{len(gdf_anomalias):,} anomalias excluídas)")

gdf_ucs_orfas.to_crs(epsg=4326)[colunas_uc].to_excel(ARQ_ORFAS, index=False)
print(f"   ✓ {ARQ_ORFAS} ({len(gdf_ucs_orfas):,} órfãs)")

# =============================================================================
# PASSO 7: MAPA
# =============================================================================
print("\n7. Gerando mapa...")

colunas_mapa = ["id_casa", "area_casa_m2", "n_edificacoes", "lat_centroide",
                "lon_centroide", "qtd_ucs", "anomalia", "poligono"]
gdf_poligonos = (
    gdf_final[colunas_mapa].copy()
    .set_geometry("poligono").set_crs(epsg=31983).to_crs(epsg=4326)
)
# Simplifica só para plotagem (evita travar o navegador)
gdf_poligonos["poligono"] = gdf_poligonos["poligono"].simplify(0.00001, preserve_topology=True)

gdf_plot = gdf_final.copy()
gdf_plot = gdf_plot.set_geometry("centroide").set_crs(epsg=31983).to_crs(epsg=4326)

def cor_marcador(row):
    if row["qtd_ucs"] == 0 and row["anexo"]:
        return "gray"
    if row["qtd_ucs"] > 0:
        return "green" if row["qtd_ucs"] == 1 else "blue"
    return "orange" if row["anomalia"] else "red"

mapa = folium.Map(location=[LAT_CENTRO, LON_CENTRO], zoom_start=14, tiles=None)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satélite", overlay=False, control=True,
).add_to(mapa)
folium.TileLayer("CartoDB positron", name="Mapa de Ruas",
                 overlay=False, control=True).add_to(mapa)

# ── CAMADA 1: Polígonos reais ────────────────────────────────────────────────
grupo_poligonos = folium.FeatureGroup(name="Polígonos das Edificações", show=True)
folium.GeoJson(
    gdf_poligonos,
    style_function=lambda _: {
        "fillColor": "#555555", "color": "#222222", "weight": 1.0, "fillOpacity": 0.15,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["id_casa", "area_casa_m2", "n_edificacoes",
                "lat_centroide", "lon_centroide", "qtd_ucs"],
        aliases=["ID:", "Área (m²):", "Polígonos no grupo:",
                 "Lat centroide:", "Lon centroide:", "Medidores:"],
        localize=True, sticky=True,
    ),
).add_to(grupo_poligonos)
grupo_poligonos.add_to(mapa)

# ── CAMADA 2: Cluster de marcadores ─────────────────────────────────────────
grupo_cluster = folium.FeatureGroup(name="Edificações (cluster)", show=True)
cluster = MarkerCluster().add_to(grupo_cluster)

for _, row in gdf_plot.iterrows():
    if row["qtd_ucs"] > 0:
        texto_consumo = f"<b style='color:green;'>Consumo Medido (Real):</b> {row['consumo_total']:.0f} kWh"
    elif row["anomalia"]:
        texto_consumo = "<b style='color:#cc6600;'>Anomalia (fora da estimativa)</b>"
    elif row["anexo"]:
        texto_consumo = "<b style='color:gray;'>Provável anexo (fora da estimativa)</b>"
    else:
        linhas = "".join(
            f" ↳ {info['nome']}: <b>{row[f'est_{k}']:.0f} kWh</b> "
            f"(<span style='color:#cc0000;'>R$ {row[f'valor_{k}']:.2f}</span>)<br>"
            for k, info in PERFIS.items()
        )
        texto_consumo = f"<b style='color:red;'>Estimativa de Furto/Mês:</b><br>{linhas}"

    cor = cor_marcador(row)
    folium.CircleMarker(
        location=[row["lat_centroide"], row["lon_centroide"]],
        radius=6, color=cor, fill=True, fill_color=cor, fill_opacity=0.85,
        tooltip=(
            f"<b>ID:</b> {row['id_casa']}<br>"
            f"<b>Área:</b> {row['area_casa_m2']:.0f} m²<br>"
            f"<b>Medidores:</b> {row['qtd_ucs']}<br>"
            f"<hr style='margin:4px 0;'>"
            f"{texto_consumo}<br>"
            f"<hr style='margin:4px 0;'>"
            f"<b>Perfil:</b> {row['classes_ucs']}"
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

# ── CAMADA 4: Anomalias para verificação manual ──────────────────────────────
gdf_verificacao = gdf_poligonos[gdf_poligonos["anomalia"]].copy()
if not gdf_verificacao.empty:
    grupo_verificacao = folium.FeatureGroup(name="Verificação Manual (Anomalias)", show=True)

    def estilo_verificacao(feature):
        if feature["properties"]["n_edificacoes"] > MAX_POLIGONOS_GRUPO:
            return {"fillColor": "#ff0000", "color": "#cc0000", "weight": 3.0, "fillOpacity": 0.75}
        return {"fillColor": "#ff8800", "color": "#cc6600", "weight": 3.0, "fillOpacity": 0.75}

    folium.GeoJson(
        gdf_verificacao,
        style_function=estilo_verificacao,
        tooltip=folium.GeoJsonTooltip(
            fields=["id_casa", "area_casa_m2", "n_edificacoes", "qtd_ucs"],
            aliases=["ID:", "Área (m²):", "Qtd. Polígonos fundidos:", "Medidores:"],
            localize=True, sticky=True,
        )
    ).add_to(grupo_verificacao)
    grupo_verificacao.add_to(mapa)

# ── Círculo da área de estudo ────────────────────────────────────────────────
folium.Circle(
    location=[LAT_CENTRO, LON_CENTRO], radius=RAIO_M,
    color="#0055ff", weight=2.5, fill=False, dash_array="8",
    tooltip=f"Área de estudo: {RAIO_M/1000:.0f} km de raio",
).add_to(mapa)

folium.Marker(
    location=[LAT_CENTRO, LON_CENTRO],
    tooltip="Centro de pesquisa",
    popup=folium.Popup(f"""
        <div style='text-align:center; width:200px; font-family:Arial;'>
            <b>Campo Belo — M1</b><br>
            Raio: {RAIO_M/1000:.0f} km<br><br>
            Total edificações: {len(gdf_final):,}<br>
            🟢 Com medidor: {int(mask_com_medidor.sum()):,}<br>
            🔴 Suspeitas: {len(gdf_sem_medidor):,}<br>
            🟠 Anomalias: {len(gdf_anomalias):,}<br><br>
            UCs no raio: {len(gdf_ucs):,}<br>
            UCs órfãs: {len(gdf_ucs_orfas):,}
        </div>""", max_width=300),
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(mapa)

legenda_html = f"""
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:15px; border-radius:8px;
     border:2px solid #ccc; font-family:Arial; font-size:13px;">
  <b>Metodologia 1 (Múltiplas UCs)</b><br>
  <i style="font-size:11px;">conf≥{CONFIANCA_MIN} | área≥{AREA_MIN_M2:.0f}m² | cabo≤{CABO_MAX_M:.0f}m</i><br><br>
  <span style="background:red;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;<b>Suspeita ({len(gdf_sem_medidor):,})</b><br><br>
  <span style="background:orange;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;Anomalia s/ medidor ({len(gdf_anomalias):,})<br><br>
  <span style="background:gray;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;Provável anexo ({len(gdf_anexos):,})<br><br>
  <span style="background:green;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;1 medidor ({int((gdf_final['qtd_ucs']==1).sum()):,})<br><br>
  <span style="background:blue;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;2+ medidores ({int((gdf_final['qtd_ucs']>1).sum()):,})<br><br>
  <span style="background:black;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;UC órfã ({len(gdf_ucs_orfas):,})<br><br>
  <hr style="margin:6px 0;">
  <i>Total: {len(gdf_final):,} edificações</i>
</div>
"""
mapa.get_root().html.add_child(folium.Element(legenda_html))

folium.LayerControl(position="topright", collapsed=False).add_to(mapa)

titulo_html = f'''
<h3 align="center" style="font-size:16px; font-family:Arial;">
    <b>M1 (Múltiplas UCs) | {len(gdf_final):,} edificações |
    Com medidor: {int(mask_com_medidor.sum()):,} |
    Suspeitas: {len(gdf_sem_medidor):,}</b>
</h3>'''
mapa.get_root().html.add_child(folium.Element(titulo_html))

mapa.save(ARQ_SAIDA_MAPA)
print(f"\n✓ Concluído:")
print(f"  → {ARQ_SAIDA_MAPA}")
print(f"  → {ARQ_SEM_MEDIDOR}")
print(f"  → {ARQ_ORFAS}")