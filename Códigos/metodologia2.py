import gc
import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, box
# Usaremos ferramentas modernas do Geopandas/Shapely, evitando o antigo unary_union

import folium
from folium.plugins import MarkerCluster

# =============================================================================
# CONFIGURAÇÕES DA INICIAÇÃO CIENTÍFICA
# =============================================================================
ARQ_OPENBUILDINGS = "Arquivos/OpBuildingCPS.gz"
ARQ_UCS           = "Arquivos/lista_ucs_fernanda_joao.xlsx"
ARQ_SAIDA_MAPA    = "meto2_ic.html"
ARQ_GRADE         = "m2_grade_ranqueada.xlsx"
ARQ_REGIOES       = "m2_regioes_suspeitas.xlsx"

RAIO_M          = 3_000
LON_CENTRO      = -47.1205
LAT_CENTRO      = -23.0384
CONFIANCA_MIN   = 0.75
AREA_MIN_M2     = 50.0
BUFFER_MESCLA_M = 0.3
CHUNKSIZE       = 200_000

# Parâmetros específicos da Metodologia 2
TAMANHO_CELULA_M = 200.0  # Tamanho do quadrado da grade (metros)
RANQUE_MINIMO    = 0.3    # Célula é suspeita se a proporção (casas sem medidor) >= 30%
CABO_MAX_M       = 30.0   # Limite físico da rede aérea (30 metros)
DIST_MESCLA_REGIAO = 15.0 # Distância para fundir casas suspeitas em uma "Região Amarela"

# =============================================================================
# FUNÇÕES DE APOIO MATEMÁTICO E GEOGRÁFICO
# =============================================================================

# (Abaixo, funções do Algoritmo de Welzl omiti os comentários para focar no fluxo principal, 
# mas a lógica matemática está intocada).
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
# PASSO 1: PROCESSAR OPENBUILDINGS (Os polígonos de telhados)
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

for i, chunk in enumerate(pd.read_csv(ARQ_OPENBUILDINGS, usecols=list(DTYPE.keys()), dtype=DTYPE, chunksize=CHUNKSIZE)):
    mask = (
        (chunk["confidence"] >= CONFIANCA_MIN) &
        (chunk["longitude"]  >= lon_min) & (chunk["longitude"] <= lon_max) &
        (chunk["latitude"]   >= lat_min) & (chunk["latitude"]  <= lat_max)
    )
    chunk = chunk.loc[mask, "geometry"]
    if not chunk.empty:
        geometrias.extend(chunk.apply(wkt.loads).tolist())
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
# PASSO 2: MESCLAGEM DE TELHADOS VIZINHOS
# =============================================================================
DIST_MESCLA_M = 5.0  

print(f"\n2. Mesclando edificações (centroides a <= {DIST_MESCLA_M}m)...")

gdf_m = gdf.copy()
gdf_m["cx"] = gdf_m.geometry.centroid.x
gdf_m["cy"] = gdf_m.geometry.centroid.y

centroides_buf = gpd.GeoDataFrame(
    {"idx_orig": gdf_m.index},
    geometry=gpd.points_from_xy(gdf_m["cx"], gdf_m["cy"]).buffer(DIST_MESCLA_M / 2),
    crs="EPSG:31983"
)

# Uso do método moderno union_all() para evitar o erro DeprecationWarning!
unido = centroides_buf.geometry.union_all()
grupos = (
    gpd.GeoDataFrame(geometry=[unido], crs="EPSG:31983")
    .explode(index_parts=False).reset_index(drop=True)
)
grupos["grupo_id"] = range(len(grupos))

gdf_pontos = gpd.GeoDataFrame(
    {"idx_orig": gdf_m.index},
    geometry=gpd.points_from_xy(gdf_m["cx"], gdf_m["cy"]),
    crs="EPSG:31983"
)
join = gpd.sjoin(gdf_pontos, grupos, how="left", predicate="within")
join = join.drop_duplicates(subset=["idx_orig"], keep="first")
gdf_m["grupo_id"] = join.set_index("idx_orig")["grupo_id"].values

print("   Criando polígonos unidos por grupo...")
edificacoes = []

for grupo_id, grupo in gdf_m.groupby("grupo_id"):
    # Mais uma vez: Usando a forma moderna union_all() da GeoSeries
    poligono_grupo = grupo.geometry.union_all()
    cx = poligono_grupo.centroid.x
    cy = poligono_grupo.centroid.y

    edificacoes.append({
        "grupo_id":      grupo_id,
        "centroide":     Point(cx, cy),
        "poligono":      poligono_grupo,
        "area_casa_m2":  grupo["area_m2"].sum(),
        "n_poligonos":   len(grupo),
    })

gdf_edificacoes = gpd.GeoDataFrame(edificacoes, geometry="centroide", crs="EPSG:31983")
gdf_edificacoes["id_casa"] = range(1, len(gdf_edificacoes)+1)

print(f"   {len(gdf_edificacoes):,} edificações definitivas.")
del gdf, gdf_m; gc.collect()

# =============================================================================
# PASSO 3: PROCESSAR MEDIDORES (UCs)
# =============================================================================
print("\n3. Processando UCs (Medidores)...")

df_ucs = pd.read_excel(ARQ_UCS)
df_ucs = df_ucs.dropna(subset=["Coordenada"]).copy()
df_ucs["geometry"] = df_ucs["Coordenada"].apply(wkt.loads)
gdf_ucs_todas = gpd.GeoDataFrame(df_ucs, geometry="geometry", crs="EPSG:4326").to_crs(epsg=31983)

centro_metros = gpd.GeoSeries([centro_shapely], crs="EPSG:4326").to_crs(epsg=31983).iloc[0]
gdf_ucs_todas["dist_centro"] = gdf_ucs_todas.geometry.distance(centro_metros)

gdf_ucs = gdf_ucs_todas[gdf_ucs_todas["dist_centro"] <= RAIO_M].copy()
print(f"   {len(gdf_ucs):,} UCs válidas no raio.")

# =============================================================================
# PASSO 4: CRIAR GRADE DE CÉLULAS (A BASE DA METODOLOGIA 2)
# =============================================================================
print(f"\n4. Criando grade de {TAMANHO_CELULA_M:.0f}m x {TAMANHO_CELULA_M:.0f}m...")

limite_metros = limite_estudo.to_crs(epsg=31983)
xmin, ymin, xmax, ymax = limite_metros.total_bounds

# Gerando a grade de forma rápida ("Pythonica") sem usar laços while
x_coords = np.arange(xmin, xmax, TAMANHO_CELULA_M)
y_coords = np.arange(ymin, ymax, TAMANHO_CELULA_M)

celulas = [box(x, y, x + TAMANHO_CELULA_M, y + TAMANHO_CELULA_M) for x in x_coords for y in y_coords]

gdf_grade = gpd.GeoDataFrame(geometry=celulas, crs="EPSG:31983")
gdf_grade = gpd.clip(gdf_grade, limite_metros).reset_index(drop=True)
gdf_grade["celula_id"] = range(len(gdf_grade))

print(f"   {len(gdf_grade):,} células criadas.")

# =============================================================================
# PASSO 5: CONTAR CONSTRUÇÕES, MEDIDORES E CALCULAR RANQUE
# =============================================================================
print("\n5. Calculando ranque de suspeita por célula...")

gdf_edif_pts = gpd.GeoDataFrame({"id_casa": gdf_edificacoes["id_casa"]}, geometry=gdf_edificacoes["centroide"], crs="EPSG:31983")

qt_construcoes = gpd.sjoin(gdf_edif_pts, gdf_grade, how="left", predicate="within").groupby("celula_id").size().rename("qt_construcoes")
qt_medidores   = gpd.sjoin(gdf_ucs, gdf_grade, how="left", predicate="within").groupby("celula_id").size().rename("qt_medidores")

gdf_grade = gdf_grade.join(qt_construcoes, on="celula_id").join(qt_medidores, on="celula_id")
gdf_grade["qt_construcoes"] = gdf_grade["qt_construcoes"].fillna(0).astype(int)
gdf_grade["qt_medidores"]   = gdf_grade["qt_medidores"].fillna(0).astype(int)

# Cálculo do Ranque (Protegendo contra divisão por zero)
gdf_grade["ranque"] = 0.0
mask_vazias = gdf_grade["qt_construcoes"] > 0
gdf_grade.loc[mask_vazias, "ranque"] = (
    (gdf_grade.loc[mask_vazias, "qt_construcoes"] - gdf_grade.loc[mask_vazias, "qt_medidores"]) /
     gdf_grade.loc[mask_vazias, "qt_construcoes"]
).clip(0, 1)

gdf_grade["ranque"] = gdf_grade["ranque"].round(3)

# Células Suspeitas (Filtra ruídos pedindo pelo menos 2 construções)
gdf_suspeitas = gdf_grade[(gdf_grade["ranque"] >= RANQUE_MINIMO) & (gdf_grade["qt_construcoes"] >= 2)].copy()

# =============================================================================
# PASSO 6: ENCONTRAR E MESCLAR AS REGIÕES AMARELAS (PÁGINA 39 DA DISSERTAÇÃO)
# =============================================================================
print("\n6. Identificando Regiões Clandestinas Maiores (Descarte e Fusão)...")

def formar_regioes_amarelas(
    gdf_casas: gpd.GeoDataFrame, 
    gdf_med: gpd.GeoDataFrame, 
    gdf_malha_suspeita: gpd.GeoDataFrame, 
    cabo_max: float, 
    dist_mescla: float
) -> gpd.GeoDataFrame:
    """
    1. Filtra as casas que caíram na malha vermelha.
    2. Descarta as que estão perto de medidores.
    3. Fonde as que sobram para criar as "Regiões Amarelas" clandestinas.
    
    Args:
        gdf_casas (gpd.GeoDataFrame): Tabela de edificações.
        gdf_med (gpd.GeoDataFrame): Tabela de medidores.
        gdf_malha_suspeita (gpd.GeoDataFrame): Tabela da grade ranqueada.
        cabo_max (float): Distância máxima permitida para o medidor.
        dist_mescla (float): Distância para unir casas suspeitas.
        
    Returns:
        gpd.GeoDataFrame: Geometrias unidas representando as regiões suspeitas.
    """
    
    # 1. Definir a geometria ativa (polígono real)
    casas_geo = gdf_casas.set_geometry("poligono").copy()
    
    # [CORREÇÃO 1] Garantir que o CRS não se perca ao trocar a geometria
    casas_geo.crs = gdf_casas.crs 
    
    # 2. PRIMEIRO JOIN: Pegar apenas as casas dentro das células suspeitas da grade
    casas_na_malha = gpd.sjoin(
        casas_geo, 
        gdf_malha_suspeita[["geometry"]], 
        how="inner", 
        predicate="intersects"
    )
    casas_na_malha = casas_na_malha.drop_duplicates(subset=["id_casa"])
    
    if casas_na_malha.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_casas.crs)

    # [CORREÇÃO 2] Jogar fora a coluna 'index_right' gerada pelo primeiro join
    if "index_right" in casas_na_malha.columns:
        casas_na_malha = casas_na_malha.drop(columns=["index_right"])

    # 3. SEGUNDO JOIN: Medir a distância para o medidor mais próximo
    distancias = gpd.sjoin_nearest(
        casas_na_malha, 
        gdf_med, 
        how="left", 
        distance_col="dist_medidor"
    ).drop_duplicates(subset=["id_casa"])
    
    # 4. O DESCARTE: Ficam apenas as casas que estão além da distância do cabo
    remanescentes = distancias[distancias["dist_medidor"] > cabo_max].copy()
    
    if remanescentes.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_casas.crs)
        
    # 5. A FUSÃO: Infla (buffer) e une as casas próximas para formar a grande região
    poligonos_inflados = remanescentes.geometry.buffer(dist_mescla / 2)
    uniao_total = poligonos_inflados.union_all()
    
    # Criamos o GeoDataFrame final com a geometria resultante
    regioes = gpd.GeoDataFrame(geometry=[uniao_total], crs=gdf_casas.crs)
    regioes = regioes.explode(index_parts=False).reset_index(drop=True)
    regioes["area_regiao_m2"] = regioes.geometry.area
    
    return regioes

# Chamando a nossa nova ferramenta
gdf_regioes_amarelas = formar_regioes_amarelas(
    gdf_edificacoes, gdf_ucs, gdf_suspeitas, CABO_MAX_M, DIST_MESCLA_REGIAO
)

print(f"   {len(gdf_regioes_amarelas)} grandes regiões suspeitas foram consolidadas.")

print("\n" + "="*55)
print("RESUMO FINAL — METODOLOGIA 2:")
print(f"  Total de células na grade        : {len(gdf_grade):,}")
print(f"  Células suspeitas (ranque≥{RANQUE_MINIMO})  : {len(gdf_suspeitas):,}")
print(f"  Grandes Regiões Amarelas         : {len(gdf_regioes_amarelas):,}")
print("="*55+"\n")


# =============================================================================
# PASSO 7: EXPORTAR PLANILHAS
# =============================================================================
print("7. Exportando planilhas...")

gdf_grade_wgs = gdf_grade.to_crs(epsg=4326)
gdf_grade_wgs[["celula_id","qt_construcoes","qt_medidores","ranque"]].to_excel(ARQ_GRADE, index=False)

gdf_sus_wgs = gdf_suspeitas.to_crs(epsg=4326)
gdf_sus_wgs[["celula_id","qt_construcoes","qt_medidores","ranque"]].to_excel(ARQ_REGIOES, index=False)


# =============================================================================
# PASSO 8: GERAR MAPA FINAL DA METODOLOGIA 2
# =============================================================================
print("\n8. Gerando mapa...")

# -----------------------------------------------------------------------------
# A. PREPARAÇÃO DOS DADOS (Conversão para coordenadas de GPS - EPSG:4326)
# -----------------------------------------------------------------------------
# Pegamos as edificações, ignoramos a coluna 'centroide', ativamos a borda ('poligono') e convertemos
gdf_poligonos_wgs = gdf_edificacoes.drop(columns=["centroide"]).set_geometry("poligono").set_crs(epsg=31983).to_crs(epsg=4326)
# Garantimos que só polígonos reais entrem no mapa
gdf_poligonos_wgs = gdf_poligonos_wgs[gdf_poligonos_wgs.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

# Convertemos as grades, regiões suspeitas e medidores para o formato do mapa
gdf_grade_mapa    = gdf_grade.to_crs(epsg=4326)
gdf_sus_mapa      = gdf_suspeitas.to_crs(epsg=4326)
gdf_amarelas_mapa = gdf_regioes_amarelas.to_crs(epsg=4326)
gdf_ucs_wgs       = gdf_ucs.to_crs(epsg=4326)

# -----------------------------------------------------------------------------
# B. FERRAMENTAS DE APOIO (Com Type Hinting)
# -----------------------------------------------------------------------------
def obter_cor_por_ranque(ranque: float) -> str:
    """
    Define a cor da célula no mapa baseada no seu nível de suspeita.
    O Type Hinting '-> str' indica que esta função obrigatoriamente devolve um texto.
    """
    if ranque >= 0.9: return "#8B0000"  # Vermelho muito escuro (Crítico)
    if ranque >= 0.7: return "#cc0000"  # Vermelho
    if ranque >= 0.5: return "#ff4400"  # Laranja-avermelhado
    if ranque >= 0.3: return "#ff8800"  # Laranja
    return "#ffcc00"                    # Amarelo

def gerar_titulo_mapa(celulas: int, residencias: int, medidores: int, regioes: int) -> str:
    """
    Gera o código HTML para o título que ficará no topo da tela.
    Usamos f-strings (f"...") para injetar as variáveis dentro do texto de forma limpa.
    """
    return f"""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 10px; font-family: Arial, sans-serif;">
        <h3 style="margin: 0; font-size: 18px; color: #333;">
            <b>Metodologia 2: Mapeamento de Perdas Não Técnicas</b>
        </h3>
        <h4 style="margin: 5px 0 0 0; font-size: 14px; color: #555; font-weight: normal;">
            Células Suspeitas: <b>{celulas:,}</b> | 
            Construções na Malha: <b>{residencias:,}</b> | 
            Medidores na Malha: <b>{medidores:,}</b> <br>
            Regiões Clandestinas (Amarelo): <b style="color: #b8b512;">{regioes:,}</b>
        </h4>
    </div>
    """

# -----------------------------------------------------------------------------
# C. CONSTRUÇÃO DO MAPA (Camada por Camada)
# -----------------------------------------------------------------------------
# Criamos a "tela em branco" centralizada nas coordenadas do estudo
mapa = folium.Map(location=[LAT_CENTRO, LON_CENTRO], zoom_start=15, tiles=None)

# Adicionamos opções de fundo (Satélite ou Mapa de Ruas)
folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satélite", overlay=False, control=True,
).add_to(mapa)
folium.TileLayer("CartoDB positron", name="Mapa de Ruas", overlay=False, control=True).add_to(mapa)

# CAMADA 1: Polígonos das Casas (Cinza transparente)
grupo_poligonos = folium.FeatureGroup(name="1. Polígonos das Edificações", show=True)
folium.GeoJson(
    gdf_poligonos_wgs,
    style_function=lambda _: {"fillColor": "#555555", "color": "#222222", "weight": 1.0, "fillOpacity": 0.15},
    tooltip=folium.GeoJsonTooltip(fields=["id_casa", "area_casa_m2"], aliases=["ID:", "Área (m²):"])
).add_to(grupo_poligonos)
grupo_poligonos.add_to(mapa)

# CAMADA 2: Regiões Clandestinas Maiores (Amarelo) - Conforme a Metodologia 2 da dissertação
grupo_amarelas = folium.FeatureGroup(name="2. Regiões Clandestinas (Amarelo)", show=True)
if not gdf_amarelas_mapa.empty:
    folium.GeoJson(
        gdf_amarelas_mapa,
        style_function=lambda _: {"fillColor": "#e8e41a", "color": "#b8b512", "weight": 2.0, "fillOpacity": 0.4},
        tooltip=folium.GeoJsonTooltip(fields=["area_regiao_m2"], aliases=["Área da Clandestina (m²):"])
    ).add_to(grupo_amarelas)
grupo_amarelas.add_to(mapa)

# CAMADA 3: Células Suspeitas (Malha Vermelha/Laranja)
grupo_suspeitas = folium.FeatureGroup(name=f"3. Células Suspeitas (Ranque≥{RANQUE_MINIMO})", show=True)
folium.GeoJson(
    gdf_sus_mapa,
    style_function=lambda f: {
        "fillColor": obter_cor_por_ranque(f["properties"]["ranque"]), 
        "color": "#8B0000", 
        "weight": 1.5, 
        "fillOpacity": 0.3
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["celula_id", "qt_construcoes", "qt_medidores", "ranque"], 
        aliases=["Célula:", "Construções:", "Medidores:", "Ranque:"]
    )
).add_to(grupo_suspeitas)
grupo_suspeitas.add_to(mapa)

# CAMADA 4: Grade Completa (Com baixa opacidade)
grupo_grade = folium.FeatureGroup(name="4. Grade Completa", show=False)
folium.GeoJson(
    gdf_grade_mapa[gdf_grade_mapa["qt_construcoes"] > 0],
    style_function=lambda f: {
        "fillColor": obter_cor_por_ranque(f["properties"]["ranque"]), 
        "color": "#333333", 
        "weight": 0.3, 
        "fillOpacity": 0.1
    },
).add_to(grupo_grade)
grupo_grade.add_to(mapa)

# CAMADA 5: Medidores UCs (Pontinhos Verdes)
grupo_ucs = folium.FeatureGroup(name="5. Medidores (UCs)", show=False)
folium.GeoJson(
    gdf_ucs_wgs,
    style_function=lambda _: {"radius": 3, "fillColor": "#00aa00", "color": "#006600", "weight": 1, "fillOpacity": 0.7},
    marker=folium.CircleMarker(radius=3, fill=True)
).add_to(grupo_ucs)
grupo_ucs.add_to(mapa)

# CAMADA 6: Círculo Azul da Área de Estudo
folium.Circle(
    location=[LAT_CENTRO, LON_CENTRO],
    radius=RAIO_M,
    color="#0055ff",
    weight=2.5,
    fill=False,
    dash_array="8",
    tooltip=f"Área de estudo: {RAIO_M/1000:.0f} km de raio",
).add_to(mapa)

# Adiciona o controle de camadas no canto superior direito
folium.LayerControl(position="topright", collapsed=False).add_to(mapa)

# -----------------------------------------------------------------------------
# D. FINALIZAÇÃO: Inserir Título e Salvar
# -----------------------------------------------------------------------------
# Extraímos os números finais para o painel de informações
tot_celulas = len(gdf_suspeitas)
tot_construcoes = int(gdf_suspeitas["qt_construcoes"].sum())
tot_medidores = int(gdf_suspeitas["qt_medidores"].sum())
tot_regioes = len(gdf_regioes_amarelas)

# Geramos o HTML e adicionamos ao mapa
html_titulo = gerar_titulo_mapa(tot_celulas, tot_construcoes, tot_medidores, tot_regioes)
mapa.get_root().html.add_child(folium.Element(html_titulo))

# Salvamos o arquivo final
mapa.save(ARQ_SAIDA_MAPA)
print(f"✓ Concluído com sucesso! Mapa renderizado em: {ARQ_SAIDA_MAPA}")