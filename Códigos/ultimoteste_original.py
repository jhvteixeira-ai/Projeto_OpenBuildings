import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, MultiPoint
from shapely.ops import unary_union
import folium
from folium.plugins import MarkerCluster
import gc
import numpy as np
import math
import random

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

# PREÇO MÉDIO DO KWH EM R$ (CPFL Paulista - Campinas)
PRECO_KWH = 0.95 

# =============================================================================
# ALGORITMO DE WELZL (Cálculo do Raio compatível com qualquer versão do Shapely)
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

def _welzl(P, R):
    if len(P)==0 or len(R)==3:
        if   len(R)==0: return (0,0,0)
        elif len(R)==1: return (R[0][0],R[0][1],0)
        elif len(R)==2: return _circulo_de_2(R[0],R[1])
        else:           return _circulo_de_3(R[0],R[1],R[2])
    p=P[0]; resto=P[1:]
    d=_welzl(resto,R)
    if d is not None and _ponto_no_circulo(d,p): return d
    return _welzl(resto, R+[p])

def menor_circulo_welzl(pontos):
    if not pontos: return None
    P=list(pontos); random.shuffle(P)
    return _welzl(P,[])

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
        # Otimização: Leitura vetorizada de WKT
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

gdf_pontos = gpd.GeoDataFrame(
    {"idx_orig": gdf_m.index},
    geometry=gpd.points_from_xy(
        gdf_m.geometry.centroid.x, gdf_m.geometry.centroid.y
    ),
    crs="EPSG:31983"
)
join = gpd.sjoin(gdf_pontos, grupos, how="left", predicate="within")
join = join.drop_duplicates(subset=["idx_orig"], keep="first")
gdf_m["grupo_id"] = join.set_index("idx_orig")["grupo_id"].values

print("   Calculando centroides e polígonos por grupo...")
edificacoes = []

for grupo_id, grupo in gdf_m.groupby("grupo_id"):
    pontos = list(zip(
        grupo.geometry.centroid.x.tolist(),
        grupo.geometry.centroid.y.tolist()
    ))

    # Otimização: Uso do Welzl para evitar erro de versão no Shapely
    if len(pontos) == 1:
        cx, cy = pontos[0]; raio = 0.0
    else:
        res = menor_circulo_welzl(pontos)
        cx, cy, raio = res if res else (pontos[0][0], pontos[0][1], 0.0)

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

del gdf, gdf_m; gc.collect()

# =============================================================================
# PASSO 3: PROCESSAR UCs
# =============================================================================
print("\n3. Processando UCs...")

df_ucs = pd.read_excel(ARQ_UCS)
df_ucs = df_ucs.dropna(subset=["Coordenada"]).copy()
# Otimização na leitura
df_ucs["geometry"] = gpd.GeoSeries.from_wkt(df_ucs["Coordenada"])
gdf_ucs_todas = gpd.GeoDataFrame(df_ucs, geometry="geometry", crs="EPSG:4326").to_crs(epsg=31983)
centro_metros = gpd.GeoSeries([centro_shapely], crs="EPSG:4326").to_crs(epsg=31983).iloc[0]
gdf_ucs_todas["dist_centro"] = gdf_ucs_todas.geometry.distance(centro_metros)

gdf_ucs      = gdf_ucs_todas[gdf_ucs_todas["dist_centro"] <= RAIO_M].copy()
print(f"   {len(gdf_ucs):,} UCs no raio de {RAIO_M/1000:.0f}km.")

# =============================================================================
# PASSO 4: ASSOCIAÇÃO COMBINATÓRIA — GREEDY POR ESCASSEZ
# =============================================================================
print(f"\n4. Associando UCs — greedy por escassez (máx. {CABO_MAX_M}m)...")

def associar_ucs_greedy(df_ucs: pd.DataFrame, df_edif: pd.DataFrame, cabo_max: float) -> pd.DataFrame:
    gdf_edif_geo = (
        df_edif[["id_casa", "centroide"]]
        .rename(columns={"centroide": "geometry"})
        .set_geometry("geometry")
    )

    todos_pares = gpd.sjoin_nearest(
        df_ucs, 
        gdf_edif_geo, 
        how="inner", 
        max_distance=cabo_max, 
        distance_col="distancia"
    )
    
    todos_pares = todos_pares.drop_duplicates(subset=["Código ID", "id_casa"])

    opcoes_por_edif = todos_pares.groupby("id_casa")["Código ID"].count().rename("n_opcoes_edif")
    todos_pares = todos_pares.join(opcoes_por_edif, on="id_casa")
    todos_pares = todos_pares.sort_values(by=["n_opcoes_edif", "distancia"], ascending=[True, True])

    uc_ids = todos_pares["Código ID"].tolist()
    edif_ids = todos_pares["id_casa"].tolist()
    distancias = todos_pares["distancia"].tolist()
    
    ucs_ja_atribuidas = set() 
    resultado = []

    for uc_id, edif_id, dist in zip(uc_ids, edif_ids, distancias):
        if uc_id not in ucs_ja_atribuidas:
            ucs_ja_atribuidas.add(uc_id) 
            resultado.append({
                "Código ID": uc_id,
                "id_casa": edif_id,
                "distancia": dist
            })

    return pd.DataFrame(resultado)

cruzamento_greedy = associar_ucs_greedy(gdf_ucs, gdf_edificacoes, CABO_MAX_M)

print(f"   UCs associadas: {len(cruzamento_greedy):,} "
      f"({len(gdf_ucs) - len(cruzamento_greedy):,} órfãs)")

# =============================================================================
# PASSO 4.5: RESGATE DE UCs ÓRFÃS POR POLÍGONO REAL
# =============================================================================
print("\n4.5. Resgatando UCs órfãs pelo polígono real...")

orfas_greedy = gdf_ucs[~gdf_ucs["Código ID"].isin(cruzamento_greedy["Código ID"])].copy()

if not orfas_greedy.empty:
    gdf_edif_poligono = (
        gdf_edificacoes[["id_casa", "area_casa_m2", "poligono"]]
        .rename(columns={"poligono": "geometry"})
        .set_geometry("geometry")
        .set_crs(epsg=31983)
    )
    # Otimização: Remoção do "double buffer", usa-se max_distance do sjoin_nearest

    resgate = gpd.sjoin_nearest(
        orfas_greedy,
        gdf_edif_poligono,
        how="inner",
        max_distance=CABO_MAX_M,
        distance_col="distancia"
    )
    resgate = (
        resgate.sort_values("distancia")
        .drop_duplicates(subset=["Código ID"], keep="first")
    )

    resgate_df = resgate[["Código ID", "id_casa", "distancia"]].copy()
    print(f"   UCs resgatadas: {len(resgate_df):,}")

    cruzamento_greedy = pd.concat(
        [cruzamento_greedy, resgate_df], ignore_index=True
    ).drop_duplicates(subset=["Código ID"], keep="first")
else:
    print("   Nenhuma órfã para resgatar.")

cruzamento = gdf_ucs.merge(cruzamento_greedy, on="Código ID", how="inner")

agrupado = cruzamento.groupby("id_casa").agg(
    qtd_ucs       = ("Código ID",      "count"),
    consumo_total = ("Consumo Mensal", "sum"),
    classes_ucs   = ("Classe", lambda x: ", ".join(x.astype(str).unique()))
).reset_index()

gdf_final = gdf_edificacoes.merge(agrupado, on="id_casa", how="left")

gdf_final["qtd_ucs"]       = gdf_final["qtd_ucs"].fillna(0).astype(int)
gdf_final["consumo_total"] = gdf_final["consumo_total"].fillna(0)
gdf_final["classes_ucs"]   = gdf_final["classes_ucs"].fillna("SEM MEDIDOR")

# =============================================================================
# PASSO 5.5: DIVISÃO ORGÂNICA DE SUSPEITOS GRANDES
# =============================================================================
print("\n5.5. Dividindo TODOS os suspeitos grandes (fatiamento orgânico)...")

AREA_ALVO = 50.0        # Área alvo por sub-polígono 
LIMIAR_DIVISAO = 150.0  # Polígonos maiores que 150m2 começam a ser cortados

mask_suspeito_grande = (
    (gdf_final["qtd_ucs"] == 0) &
    (gdf_final["area_casa_m2"] > LIMIAR_DIVISAO)
)
gdf_suspeitos_grandes = gdf_final[mask_suspeito_grande].copy()
gdf_resto             = gdf_final[~mask_suspeito_grande].copy()

print(f"   Suspeitos que serão fatiados: {len(gdf_suspeitos_grandes):,}")

def dividir_poligono_recursivo(poligono, area_alvo):
    if poligono.area <= area_alvo * 1.5:
        return [poligono]

    minx, miny, maxx, maxy = poligono.bounds
    largura = maxx - minx
    altura = maxy - miny

    if largura >= altura:
        meio = minx + (largura / 2.0)
        box1 = box(minx, miny, meio, maxy)
        box2 = box(meio, miny, maxx, maxy)
    else:
        meio = miny + (altura / 2.0)
        box1 = box(minx, miny, maxx, meio)
        box2 = box(minx, meio, maxx, maxy)

    partes = []
    for b in [box1, box2]:
        intersecao = poligono.intersection(b)
        if not intersecao.is_empty:
            if intersecao.geom_type == 'MultiPolygon':
                for geom in intersecao.geoms:
                    if geom.area >= 20.0:
                        partes.extend(dividir_poligono_recursivo(geom, area_alvo))
            elif intersecao.area >= 20.0:
                partes.extend(dividir_poligono_recursivo(intersecao, area_alvo))

    return partes if partes else [poligono]

novos_registros = []
proximo_id = int(gdf_final["id_casa"].max()) + 1

for _, row in gdf_suspeitos_grandes.iterrows():
    partes = dividir_poligono_recursivo(row["poligono"], AREA_ALVO)

    for parte in partes:
        cx = parte.centroid.x
        cy = parte.centroid.y

        novos_registros.append({
            "id_casa":       proximo_id,
            "grupo_id":      row["grupo_id"],
            "centroide":     Point(cx, cy),
            "poligono":      parte,
            "area_casa_m2":  parte.area,
            "n_edificacoes": 1,
            "raio_grupo_m":  0.0,
            "lat_centroide": 0.0, # Preenchido via vetorização depois
            "lon_centroide": 0.0, # Preenchido via vetorização depois
            "qtd_ucs":       0,
            "consumo_total": 0.0,
            "classes_ucs":   "SEM MEDIDOR",
        })
        proximo_id += 1

if novos_registros:
    # Otimização: CRS to_crs aplicado de forma vetorizada fora do loop
    gdf_novos = gpd.GeoDataFrame(novos_registros, geometry="centroide", crs="EPSG:31983")
    cent_wgs = gdf_novos.to_crs(epsg=4326)
    gdf_novos["lat_centroide"] = cent_wgs.geometry.y.round(6).values
    gdf_novos["lon_centroide"] = cent_wgs.geometry.x.round(6).values

    print(f"   Sub-polígonos gerados (~50m2) : {len(gdf_novos):,}")

    # ── REALOCAÇÃO DE UCs ÓRFÃS PARA OS NOVOS SUB-POLÍGONOS ─────────────────
    # Os sub-polígonos são novos e ainda não participaram do greedy.
    # Aproveitamos as UCs que ainda estão órfãs para tentar associá-las
    # aos sub-polígonos antes de calcular estimativas.
    ucs_casadas_ate_agora = cruzamento_greedy["Código ID"].unique()
    orfas_para_novos = gdf_ucs[~gdf_ucs["Código ID"].isin(ucs_casadas_ate_agora)].copy()

    print(f"   UCs órfãs disponíveis para realocação: {len(orfas_para_novos):,}")

    if not orfas_para_novos.empty:
        # Usa polígono real dos sub-polígonos para o join (mais preciso que centroide)
        gdf_novos_pol = (
            gdf_novos[["id_casa", "poligono"]]
            .rename(columns={"poligono": "geometry"})
            .set_geometry("geometry")
            .set_crs(epsg=31983)
        )

        resgate_novos = gpd.sjoin_nearest(
            orfas_para_novos,
            gdf_novos_pol,
            how="inner",
            max_distance=CABO_MAX_M,
            distance_col="distancia"
        )
        resgate_novos = (
            resgate_novos.sort_values("distancia")
            .drop_duplicates(subset=["Código ID"], keep="first")
        )

        if not resgate_novos.empty:
            resgate_df2 = resgate_novos[["Código ID", "id_casa", "distancia"]].copy()

            # Adiciona ao cruzamento_greedy (UCs já atribuídas têm prioridade)
            cruzamento_greedy = pd.concat(
                [cruzamento_greedy, resgate_df2], ignore_index=True
            ).drop_duplicates(subset=["Código ID"], keep="first")

            # Atualiza qtd_ucs e consumo_total nos novos sub-polígonos
            agrupado_novos = resgate_novos.groupby("id_casa").agg(
                qtd_ucs       = ("Código ID",      "count"),
                consumo_total = ("Consumo Mensal", "sum"),
                classes_ucs   = ("Classe", lambda x: ", ".join(x.astype(str).unique()))
            ).reset_index()

            gdf_novos = gdf_novos.merge(
                agrupado_novos, on="id_casa", how="left", suffixes=("", "_r")
            )
            for col in ["qtd_ucs", "consumo_total", "classes_ucs"]:
                col_r = col + "_r"
                if col_r in gdf_novos.columns:
                    mask = gdf_novos[col_r].notna()
                    gdf_novos.loc[mask, col] = gdf_novos.loc[mask, col_r]
                    gdf_novos = gdf_novos.drop(columns=[col_r])

            gdf_novos["qtd_ucs"]       = gdf_novos["qtd_ucs"].fillna(0).astype(int)
            gdf_novos["consumo_total"] = gdf_novos["consumo_total"].fillna(0)
            gdf_novos["classes_ucs"]   = gdf_novos["classes_ucs"].fillna("SEM MEDIDOR")

            print(f"   UCs realocadas para sub-polígonos : {len(resgate_df2):,}")
            print(f"   Sub-polígonos que receberam UC    : {(gdf_novos['qtd_ucs']>0).sum():,}")
            print(f"   Sub-polígonos ainda sem UC        : {(gdf_novos['qtd_ucs']==0).sum():,}")
        else:
            print("   Nenhuma UC órfã encontrada próxima aos sub-polígonos.")
    else:
        print("   Nenhuma UC órfã disponível para realocação.")

    gdf_final = pd.concat([gdf_resto, gdf_novos], ignore_index=True)
else:
    print("   Nenhum suspeito atendeu aos critérios para divisão.")
    gdf_final = gdf_resto.copy()

# =============================================================================
# PASSO 5.6: APLICAÇÃO DOS 3 PERFIS DE CONSUMO PARA SUSPEITAS
# =============================================================================
print("\n5.6. Aplicando os 3 perfis de consumo para as edificações suspeitas...")

KWH_POR_M2_MES = (1.5 * 24 * 30) / 1000
CONSUMO_ARTIGO_VERAO = 460.0
CONSUMO_ARTIGO_INVERNO = 190.0
CONSUMO_ESTIMADO_PROPRIO = 105.77

mask_com_medidor = gdf_final["qtd_ucs"] > 0
mask_sem_medidor = gdf_final["qtd_ucs"] == 0

# Criando as colunas no DataFrame principal (Consumo em kWh)
for col in ["est_1_cientifica", "est_2_verao", "est_2_inverno", "est_3_propria"]:
    gdf_final[col] = 0.0
    gdf_final.loc[mask_com_medidor, col] = gdf_final.loc[mask_com_medidor, "consumo_total"]

# Cálculo Perfil 1 (Multiplica pela área do polígono)
gdf_final.loc[mask_sem_medidor, "est_1_cientifica"] = gdf_final.loc[mask_sem_medidor, "area_casa_m2"] * KWH_POR_M2_MES
# Cálculo Perfil 2 e 3 (Valor fixo aplicado a cada polígono/fatia cortada)
gdf_final.loc[mask_sem_medidor, "est_2_verao"]   = CONSUMO_ARTIGO_VERAO
gdf_final.loc[mask_sem_medidor, "est_2_inverno"] = CONSUMO_ARTIGO_INVERNO
gdf_final.loc[mask_sem_medidor, "est_3_propria"] = CONSUMO_ESTIMADO_PROPRIO

# NOVO CÁLCULO FINANCEIRO (R$)
for col in ["est_1_cientifica", "est_2_verao", "est_2_inverno", "est_3_propria"]:
    col_valor = col.replace("est_", "valor_")
    gdf_final[col_valor] = 0.0
    gdf_final.loc[mask_sem_medidor, col_valor] = gdf_final.loc[mask_sem_medidor, col] * PRECO_KWH

# Atualiza o DataFrame focado apenas nas suspeitas
gdf_sem_medidor = gdf_final[mask_sem_medidor].copy()

print(f"   Perda P1 (Científico) : {gdf_sem_medidor['est_1_cientifica'].sum():,.0f} kWh | R$ {gdf_sem_medidor['valor_1_cientifica'].sum():,.2f}")
print(f"   Perda P2 (Verão)      : {gdf_sem_medidor['est_2_verao'].sum():,.0f} kWh | R$ {gdf_sem_medidor['valor_2_verao'].sum():,.2f}")
print(f"   Perda P2 (Inverno)    : {gdf_sem_medidor['est_2_inverno'].sum():,.0f} kWh | R$ {gdf_sem_medidor['valor_2_inverno'].sum():,.2f}")
print(f"   Perda P3 (Própria)    : {gdf_sem_medidor['est_3_propria'].sum():,.0f} kWh | R$ {gdf_sem_medidor['valor_3_propria'].sum():,.2f}")

# =============================================================================
# PASSO 6: EXPORTAR PLANILHAS
# =============================================================================
print("\n6. Exportando planilhas...")
ucs_casadas_ids = cruzamento_greedy["Código ID"].unique()
gdf_ucs_orfas   = gdf_ucs[~gdf_ucs["Código ID"].isin(ucs_casadas_ids)]

colunas_uc = [c for c in df_ucs.columns if c != "geometry"]

# Tabela das suspeitas agora com as 4 colunas de kWh e 4 colunas de R$
gdf_sem_medidor[[
    "id_casa", "area_casa_m2", "n_edificacoes",
    "raio_grupo_m", "lat_centroide", "lon_centroide",
    "est_1_cientifica", "valor_1_cientifica", 
    "est_2_verao", "valor_2_verao", 
    "est_2_inverno", "valor_2_inverno", 
    "est_3_propria", "valor_3_propria"
]].to_excel(ARQ_SEM_MEDIDOR, index=False)
print(f"   ✓ {ARQ_SEM_MEDIDOR} ({len(gdf_sem_medidor):,} suspeitas)")

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

def cor_por_uc(qtd):
    if qtd == 0: return "red"
    if qtd == 1: return "green"
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
# Otimização: Simplifica polígonos apenas para plotagem, evita travamento no navegador
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
        fields=["id_casa","area_casa_m2","n_edificacoes",
                "lat_centroide","lon_centroide","qtd_ucs"],
        aliases=["ID:","Área (m²):","Polígonos no grupo:",
                 "Lat centroide:","Lon centroide:","Medidores:"],
        localize=True,
        sticky=True,
    ),
).add_to(grupo_poligonos)
grupo_poligonos.add_to(mapa)

# ── CAMADA 2: Cluster de marcadores ─────────────────────────────────────────
grupo_cluster = folium.FeatureGroup(name="Edificações (cluster)", show=True)
cluster = MarkerCluster().add_to(grupo_cluster)

for _, row in gdf_plot.iterrows():
    
    if row["qtd_ucs"] == 0:
        texto_consumo = (
            f"<b style='color:red;'>Estimativa de Furto/Mês:</b><br>"
            f" ↳ P1 (1,5 W/m²): <b>{row['est_1_cientifica']:.1f} kWh</b> (<span style='color:#cc0000;'>R$ {row['valor_1_cientifica']:.2f}</span>)<br>"
            f" ↳ P2 (Verão): <b>{row['est_2_verao']:.0f} kWh</b> (<span style='color:#cc0000;'>R$ {row['valor_2_verao']:.2f}</span>)<br>"
            f" ↳ P2 (Inverno): <b>{row['est_2_inverno']:.0f} kWh</b> (<span style='color:#cc0000;'>R$ {row['valor_2_inverno']:.2f}</span>)<br>"
            f" ↳ P3 (Próprio): <b>{row['est_3_propria']:.1f} kWh</b> (<span style='color:#cc0000;'>R$ {row['valor_3_propria']:.2f}</span>)"
        )
    else:
        texto_consumo = f"<b style='color:green;'>Consumo Medido (Real):</b> {row['consumo_total']:.0f} kWh"

    folium.CircleMarker(
        location=[row["lat_centroide"], row["lon_centroide"]],
        radius=6,
        color=cor_por_uc(row["qtd_ucs"]),
        fill=True,
        fill_color=cor_por_uc(row["qtd_ucs"]),
        fill_opacity=0.85,
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
        <div style='text-align:center; width:200px; font-family:Arial;'>
            <b>Campo Belo — M1</b><br>
            Raio: {RAIO_M/1000:.0f} km<br><br>
            Total edificações: {len(gdf_final):,}<br>
            🟢 Com medidor: {(gdf_final['qtd_ucs']>0).sum():,}<br>
            🔴 Sem medidor: {len(gdf_sem_medidor):,}<br><br>
            UCs no raio: {len(gdf_ucs):,}<br>
            UCs órfãs: {len(gdf_ucs_orfas):,}
        </div>""", max_width=300),
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(mapa)

# Legenda
legenda_html = f"""
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:15px; border-radius:8px;
     border:2px solid #ccc; font-family:Arial; font-size:13px;">
  <b>Metodologia 1 (Múltiplas UCs)</b><br>
  <i style="font-size:11px;">conf≥0.75 | área≥{AREA_MIN_M2:.0f}m² | cabo≤{CABO_MAX_M:.0f}m</i><br><br>
  <span style="background:red;padding:2px 10px;border-radius:50%;">&nbsp;</span>
  &nbsp;<b>Sem medidor ({len(gdf_sem_medidor):,})</b><br><br>
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
# PASSO EXTRA: VERIFICAÇÃO MANUAL DE ANOMALIAS (PODE APAGAR DEPOIS)
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
    Sem medidor: {len(gdf_sem_medidor):,}</b>
</h3>'''
mapa.get_root().html.add_child(folium.Element(titulo_html))

mapa.save(ARQ_SAIDA_MAPA)
print(f"\n✓ Concluído:")
print(f"  → {ARQ_SAIDA_MAPA}")
print(f"  → {ARQ_SEM_MEDIDOR}")
print(f"  → {ARQ_ORFAS}")