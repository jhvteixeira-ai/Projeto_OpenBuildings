import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
from shapely.ops import unary_union
import folium
import gc

# =============================================================================
# CONFIGURAÇÕES GERAIS DA PESQUISA
# =============================================================================
ARQ_OPENBUILDINGS = "OpBuildingCPS.gz"
ARQ_UCS           = "lista_ucs_fernanda_joao.xlsx"
ARQ_SAIDA_MAPA    = "analise_irregularidades_ic.html"
ARQ_SAIDA_ORFAS   = "ucs_sem_edificacao.xlsx"
ARQ_SEM_MEDIDOR   = "edificacoes_sem_medidor.xlsx"
ARQ_UCS_FORA      = "ucs_fora_do_raio.xlsx"

RAIO_M            = 3_000
LON_CENTRO        = -47.1205
LAT_CENTRO        = -23.0384

CONFIANCA_MIN     = 0.60
AREA_MIN_M2       = 30.0
BUFFER_MESCLA_M   = 0.05
CHUNKSIZE         = 200_000

# Raio de busca para casar UC com edificação
# Infla o PONTO da UC (não a casa!) — casas não se tocam
TOLERANCIA_UC_M   = 15.0

# =============================================================================
# PASSO 1: PROCESSAR OPENBUILDINGS (meu código)
# =============================================================================
print("1. Processando arquivo do satélite (OpenBuildings)...")

centro_shapely = Point(LON_CENTRO, LAT_CENTRO)

limite_estudo = (
    gpd.GeoDataFrame(geometry=[centro_shapely], crs="EPSG:4326")
    .to_crs(epsg=31983)
    .buffer(RAIO_M)
    .to_crs(epsg=4326)
)

lon_min, lat_min, lon_max, lat_max = limite_estudo.total_bounds

DTYPE = {
    "confidence": "float32",
    "latitude":   "float32",
    "longitude":  "float32",
    "geometry":   "object",
}

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
        geometrias.extend(chunk.apply(wkt.loads).tolist())
    del chunk
    gc.collect()

    if (i + 1) % 10 == 0:
        print(f"   ...{(i+1)*CHUNKSIZE:,} linhas lidas | {len(geometrias):,} polígonos coletados")

print(f"   Leitura concluída: {len(geometrias):,} polígonos brutos.")

gdf_casas = gpd.GeoDataFrame(geometry=geometrias, crs="EPSG:4326")
del geometrias
gc.collect()

gdf_casas = gpd.clip(gdf_casas, limite_estudo).to_crs(epsg=31983)

unido = unary_union(gdf_casas.buffer(BUFFER_MESCLA_M))
del gdf_casas
gc.collect()

gdf_casas = (
    gpd.GeoDataFrame(geometry=[unido], crs="EPSG:31983")
    .explode(index_parts=False)
    .reset_index(drop=True)
)
del unido
gc.collect()

gdf_casas["geometry"]     = gdf_casas.buffer(-BUFFER_MESCLA_M)
gdf_casas["area_casa_m2"] = gdf_casas.geometry.area
gdf_casas                 = gdf_casas[gdf_casas["area_casa_m2"] >= AREA_MIN_M2].copy()
gdf_casas["id_casa"]      = range(1, len(gdf_casas) + 1)

print(f"   ✓ {len(gdf_casas):,} edificações processadas e consolidadas.")

# =============================================================================
# PASSO 2: PROCESSAR UCs (lógica do Gemini)
# =============================================================================
print("\n2. Processando UCs da Concessionária (Excel)...")

df_ucs = pd.read_excel(ARQ_UCS)
df_ucs["geometry"] = df_ucs["Coordenada"].apply(wkt.loads)

gdf_ucs_todas = gpd.GeoDataFrame(df_ucs, geometry="geometry", crs="EPSG:4326").to_crs(epsg=31983)

centro_metros = gpd.GeoSeries([centro_shapely], crs="EPSG:4326").to_crs(epsg=31983).iloc[0]
gdf_ucs_todas["dist_centro"] = gdf_ucs_todas.geometry.distance(centro_metros)

gdf_ucs      = gdf_ucs_todas[gdf_ucs_todas["dist_centro"] <= RAIO_M].copy()
gdf_ucs_fora = gdf_ucs_todas[gdf_ucs_todas["dist_centro"] >  RAIO_M].copy()

print(f"   ✓ {len(gdf_ucs):,} UCs válidas dentro do raio de {RAIO_M/1000}km.")

# =============================================================================
# PASSO 3: CRUZAMENTO ESPACIAL (lógica do Gemini, buffer corrigido)
#
# CORREÇÃO vs. Gemini original:
#   Gemini  → infla as CASAS  → casas vizinhas se fundem → ERRADO
#   Este    → infla os PONTOS das UCs → casas intactas  → CORRETO
# =============================================================================
# =============================================================================
# PASSO 3: CRUZAMENTO ESPACIAL INTELIGENTE (sjoin_nearest)
# =============================================================================
print("\n3. Realizando Cruzamento Espacial (Procurando a casa mais próxima)...")

# O sjoin_nearest conecta o ponto da UC à casa MAIS PRÓXIMA matematicamente.
# max_distance=15.0 garante que ele não vá buscar uma casa no outro quarteirão.
# distance_col salva a distância exata em metros para o nosso controle.
cruzamento = gpd.sjoin_nearest(
    gdf_ucs,
    gdf_casas[["id_casa", "area_casa_m2", "geometry"]],
    how="inner",
    max_distance=15.0,
    distance_col="distancia_uc_casa"
)

# Se duas casas estiverem à EXATA mesma distância milimétrica do medidor,
# o pandas pode duplicar a linha. O drop_duplicates limpa isso.
ucs_antes = len(cruzamento)
cruzamento = cruzamento.drop_duplicates(subset=["Código ID"], keep="first")
ucs_duplicadas_evitadas = ucs_antes - len(cruzamento)

# Agrupa por edificação
agrupado = cruzamento.groupby("id_casa").agg(
    qtd_ucs       = ("Código ID",      "count"),
    consumo_total = ("Consumo Mensal", "sum"),
    classes_ucs   = ("Classe",         lambda x: ", ".join(x.astype(str).unique()))
).reset_index()

# Cola os dados de UC nas edificações originais
gdf_final = gdf_casas.merge(agrupado, on="id_casa", how="left")
gdf_final["qtd_ucs"]       = gdf_final["qtd_ucs"].fillna(0).astype(int)
gdf_final["consumo_total"] = gdf_final["consumo_total"].fillna(0)
gdf_final["classes_ucs"]   = gdf_final["classes_ucs"].fillna("SEM MEDIDOR")

# =============================================================================
# PASSO 4: UCs ÓRFÃS E EDIFICAÇÕES SEM MEDIDOR
# =============================================================================
ucs_casadas_ids = cruzamento["Código ID"].unique()
gdf_ucs_orfas   = gdf_ucs[~gdf_ucs["Código ID"].isin(ucs_casadas_ids)]
gdf_sem_medidor = gdf_final[gdf_final["qtd_ucs"] == 0]

print("\n" + "=" * 55)
print("RELATÓRIO FINAL DO CRUZAMENTO:")
print(f"  Medidores em zona de conflito (resolvidos): {ucs_duplicadas_evitadas:,}")
print(f"  Medidores casados com uma edificação       : {len(ucs_casadas_ids):,}")
print(f"  Medidores órfãos (sem edificação em 5m)   : {len(gdf_ucs_orfas):,}")
print(f"  Edificações COM medidor (≥ 1 UC)           : {(gdf_final['qtd_ucs'] > 0).sum():,}")
print(f"  Edificações SEM medidor (0 UCs)            : {len(gdf_sem_medidor):,}  ← foco da IC")
print(f"  UCs fora do raio de pesquisa               : {len(gdf_ucs_fora):,}")
print("=" * 55 + "\n")

# =============================================================================
# PASSO 5: EXPORTAR PLANILHAS
# =============================================================================
print("5. Exportando planilhas...")

colunas_uc = [c for c in df_ucs.columns if c != "geometry"]

gdf_sem_medidor.to_crs(epsg=4326)[["id_casa", "area_casa_m2"]].to_excel(ARQ_SEM_MEDIDOR, index=False)
print(f"   ✓ {ARQ_SEM_MEDIDOR} ({len(gdf_sem_medidor):,} edificações)")

gdf_ucs_fora.to_crs(epsg=4326)[colunas_uc].to_excel(ARQ_UCS_FORA, index=False)
print(f"   ✓ {ARQ_UCS_FORA} ({len(gdf_ucs_fora):,} UCs)")

gdf_ucs_orfas.to_crs(epsg=4326)[colunas_uc].to_excel(ARQ_SAIDA_ORFAS, index=False)
print(f"   ✓ {ARQ_SAIDA_ORFAS} ({len(gdf_ucs_orfas):,} UCs)")

# =============================================================================
# PASSO 6: GERAR MAPA
# =============================================================================
print("\n6. Gerando mapa interativo...")

gdf_final = gdf_final.to_crs(epsg=4326)

def cor_por_uc(qtd):
    if qtd == 0: return "#ff0000"   # vermelho — sem medidor (foco da IC)
    if qtd == 1: return "#0088ff"   # azul     — ligação regular
    return "#00cc44"                # verde    — múltiplas UCs

mapa = folium.Map(location=[LAT_CENTRO, LON_CENTRO], zoom_start=15, tiles=None)

folium.TileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satélite", overlay=False, control=True,
).add_to(mapa)

folium.TileLayer("CartoDB positron", name="Mapa de Ruas", overlay=False, control=True).add_to(mapa)

folium.GeoJson(
    gdf_final,
    name="Edificações (Status de Ligação)",
    style_function=lambda x: {
        "fillColor":   cor_por_uc(x["properties"]["qtd_ucs"]),
        "color":       cor_por_uc(x["properties"]["qtd_ucs"]),
        "weight":      1.5,
        "fillOpacity": 0.45,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["id_casa", "area_casa_m2", "qtd_ucs", "consumo_total", "classes_ucs"],
        aliases=["ID:", "Área (m²):", "Qtd. Medidores:", "Consumo (kWh):", "Perfil:"],
        localize=True,
    ),
).add_to(mapa)

folium.Circle(
    location=[LAT_CENTRO, LON_CENTRO],
    radius=RAIO_M, color="#ffffff", weight=2, fill=False, dash_array="8",
    tooltip=f"Raio de estudo: {RAIO_M/1000:.0f} km",
).add_to(mapa)

texto_popup = f"""
    <div style='text-align:center; width:210px; font-family:Arial,sans-serif;'>
        <b>Jardim Campo Belo — {RAIO_M/1000:.0f} km</b><br><br>
        Total edificações: {len(gdf_final):,}<br>
        🔵 Com medidor: {(gdf_final['qtd_ucs'] > 0).sum():,}<br>
        🔴 <b>Sem medidor: {len(gdf_sem_medidor):,}</b><br><br>
        UCs no raio: {len(gdf_ucs):,}<br>
        UCs órfãs: {len(gdf_ucs_orfas):,}
    </div>
"""
folium.Marker(
    location=[LAT_CENTRO, LON_CENTRO],
    tooltip="Clique para ver o resumo",
    popup=folium.Popup(texto_popup, max_width=300),
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(mapa)

legenda = f"""
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:white; padding:15px; border-radius:8px;
     border:2px solid #ccc; font-family:Arial; font-size:13px;">
  <b>Status de Conexão à Rede</b><br><br>
  <span style="background:#ff0000;padding:2px 12px;">&nbsp;</span>
  &nbsp;<b>0 UCs — Alerta ({len(gdf_sem_medidor):,})</b><br><br>
  <span style="background:#0088ff;padding:2px 12px;">&nbsp;</span>
  &nbsp;1 UC — Regular<br><br>
  <span style="background:#00cc44;padding:2px 12px;">&nbsp;</span>
  &nbsp;2+ UCs — Múltiplas ligações<br><br>
  <hr style="margin:8px 0;">
  <i>Total: {len(gdf_final):,} edificações</i>
</div>
"""
mapa.get_root().html.add_child(folium.Element(legenda))
# --- ADICIONE ISTO NO PASSO 6 (Antes do folium.LayerControl) ---

# Criamos um grupo (FeatureGroup) para poder ligar/desligar os pontinhos pretos
grupo_orfas = folium.FeatureGroup(name="UCs Órfãs (Sem Casa)", show=True).add_to(mapa)

for _, row in gdf_ucs_orfas.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=2,           # Pontinho bem pequeno
        color="#000000",    # Cor Preta
        fill=True,
        fillOpacity=1.0,
        tooltip=f"UC Órfã: {row['Código ID']} | Classe: {row['Classe']}"
    ).add_to(grupo_orfas)
folium.LayerControl(position="topright", collapsed=False).add_to(mapa)

titulo_html = f'''
    <h3 align="center" style="font-size:18px; font-family:Arial,sans-serif;">
        <b>Edificações: {len(gdf_final):,} | Com medidor: {(gdf_final['qtd_ucs']>0).sum():,} | Sem medidor: {len(gdf_sem_medidor):,}</b>
    </h3>
'''
mapa.get_root().html.add_child(folium.Element(titulo_html))

mapa.save(ARQ_SAIDA_MAPA)
print(f"\n✓ Arquivos gerados:")
print(f"  → {ARQ_SAIDA_MAPA}")
print(f"  → {ARQ_SEM_MEDIDOR}")
print(f"  → {ARQ_UCS_FORA}")
print(f"  → {ARQ_SAIDA_ORFAS}")