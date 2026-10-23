import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import folium
from folium.plugins import MarkerCluster

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
ARQUIVO_UCS = "lista_ucs_fernanda_joao.xlsx"  # Retornado para o formato Excel .xlsx
RAIO_M      = 3_000                          # Raio da área de estudo em metros
LON_CENTRO  = -47.1205                       # Longitude do ponto central
LAT_CENTRO  = -23.0384                       # Latitude do ponto central

# =============================================================================
# PASSO 1: LER O ARQUIVO EXCEL
# =============================================================================
print("1. Lendo arquivo de unidades consumidoras...")

# Lemos o arquivo Excel. Agora vai funcionar pois instalamos o openpyxl
df = pd.read_excel(ARQUIVO_UCS)
print(f"   Total de UCs no arquivo: {len(df):,}")

# =============================================================================
# PASSO 2: VALIDAR E RECALCULAR DISTÂNCIAS
# =============================================================================
print("\n2. Validando distâncias espaciais...")

# Criamos o ponto central e convertemos para metros (EPSG:31983)
centro_shapely = Point(LON_CENTRO, LAT_CENTRO)
centro_gdf     = gpd.GeoDataFrame(geometry=[centro_shapely], crs="EPSG:4326").to_crs(epsg=31983)

# Convertendo a coluna de texto WKT para geometrias reais do Shapely
df["geometry"] = df["Coordenada"].apply(wkt.loads)

# Transformando o Pandas DataFrame em um GeoPandas GeoDataFrame (projetado em metros)
gdf_all = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(epsg=31983)

# Calculamos a distância de todas as UCs até o centro
gdf_all["dist_recalc"] = gdf_all.geometry.distance(centro_gdf.geometry[0])

# Voltamos para graus (EPSG:4326) para que o Folium consiga desenhar no mapa
gdf_all = gdf_all.to_crs(epsg=4326)

# =============================================================================
# PASSO 3: FILTRAR PELO RAIO
# =============================================================================
print(f"\n3. Filtrando UCs dentro do raio de {RAIO_M/1000:.0f} km...")

# Separamos quem está dentro e quem está fora do raio estipulado
gdf_dentro = gdf_all[gdf_all["dist_recalc"] <= RAIO_M].copy()
gdf_fora   = gdf_all[gdf_all["dist_recalc"] >  RAIO_M].copy()

# Calculando quantas UCs dividem a exata mesma coordenada (sobreposição)
coordenadas_repetidas = gdf_dentro.duplicated(subset=['geometry']).sum()

print("\n" + "=" * 50)
print("RESULTADOS:")
print(f"  Total de UCs no arquivo    : {len(gdf_all):,}")
print(f"  UCs dentro do raio (≤ 3km) : {len(gdf_dentro):,}")
print(f"  UCs fora do raio   (> 3km) : {len(gdf_fora):,}")
print(f"  UCs sobrepostas (mesmo XY) : {coordenadas_repetidas:,}")
print("-" * 50)
print("DISTRIBUIÇÃO POR CLASSE (dentro do raio):")
print(gdf_dentro["Classe"].value_counts().to_string())
print("=" * 50 + "\n")
print(gdf_dentro.duplicated(subset=['geometry']).sum())

# =============================================================================
# PASSO 4: GERAR MAPA INTERATIVO (COM CLUSTERING)
# =============================================================================
print("4. Gerando mapa interativo...")

def cor_classe(classe: str) -> str:
    """
    Retorna a cor em hexadecimal baseada nas duas primeiras letras da classe da UC.
    Utilizamos Type Hinting (classe: str) -> str para indicar que a função
    recebe uma string e retorna uma string.
    """
    classe = str(classe)
    if classe.startswith("RE"): return "#ff4444"  # Residencial - vermelho
    if classe.startswith("CO"): return "#4488ff"  # Comercial   - azul
    if classe.startswith("IN"): return "#ff8800"  # Industrial  - laranja
    if classe.startswith("RU"): return "#22bb44"  # Rural       - verde
    return "#aaaaaa"                              # Outros      - cinza

# Criamos o mapa base
mapa = folium.Map(location=[LAT_CENTRO, LON_CENTRO], zoom_start=14, tiles=None)

# Camada de Satélite
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Satélite", overlay=False, control=True,
).add_to(mapa)

# Camada de Mapa de Ruas Padrão
folium.TileLayer(
    tiles="CartoDB positron", name="Mapa de Ruas", overlay=False, control=True,
).add_to(mapa)

# CRIANDO O CLUSTER: Isso agrupa as UCs próximas para não travar o navegador
cluster_ucs = MarkerCluster(name="Unidades Consumidoras").add_to(mapa)

# Adicionamos cada UC DENTRO do Cluster para garantir uma boa visualização
for _, row in gdf_dentro.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=6,
        color=cor_classe(row["Classe"]),
        fill=True,
        fill_opacity=0.9,
        weight=1,
        tooltip=(
            f"<b>Código:</b> {row['Código ID']}<br>"
            f"<b>Classe:</b> {row['Classe']}<br>"
            f"<b>Bairro:</b> {row['Bairro']}<br>"
            f"<b>Consumo:</b> {row['Consumo Mensal']} kWh<br>"
            f"<b>Distância:</b> {row['dist_recalc']:.0f} m"
        ),
    ).add_to(cluster_ucs)

# Linha tracejada do raio de estudo
folium.Circle(
    location=[LAT_CENTRO, LON_CENTRO],
    radius=RAIO_M, color="#0055ff", weight=2,
    fill=False, dash_array="8",
    tooltip=f"Raio de estudo: {RAIO_M/1000:.0f} km",
).add_to(mapa)

# Marcador central interativo
texto_popup = f"""
    <div style='text-align:center; width:180px;'>
        <b>Centro de Pesquisa</b><br>
        Raio: {RAIO_M/1000:.0f} km<br>
        <b>UCs dentro do raio: {len(gdf_dentro):,}</b>
    </div>
"""
folium.Marker(
    location=[LAT_CENTRO, LON_CENTRO],
    tooltip="Clique para ver o resumo",
    popup=folium.Popup(texto_popup, max_width=300),
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(mapa)

# Legenda HTML personalizada
legenda_html = f"""
<div style="position:fixed; bottom:40px; left:40px; z-index:1000;
     background:white; padding:12px; border-radius:8px;
     border:2px solid #ccc; font-family:Arial,sans-serif; font-size:13px;">
  <b>Classe das UCs</b><br><br>
  <span style="background:#ff4444;padding:2px 12px;">&nbsp;</span>&nbsp;Residencial (RE)<br><br>
  <span style="background:#4488ff;padding:2px 12px;">&nbsp;</span>&nbsp;Comercial (CO)<br><br>
  <span style="background:#ff8800;padding:2px 12px;">&nbsp;</span>&nbsp;Industrial (IN)<br><br>
  <span style="background:#22bb44;padding:2px 12px;">&nbsp;</span>&nbsp;Rural (RU)<br><br>
  <span style="background:#aaaaaa;padding:2px 12px;">&nbsp;</span>&nbsp;Outros<br><br>
  <b>Total mapeado: {len(gdf_dentro):,}</b>
</div>
"""
mapa.get_root().html.add_child(folium.Element(legenda_html))

# Controle para alternar entre as camadas de mapa/satélite
folium.LayerControl(position="topright", collapsed=False).add_to(mapa)

# Título da página web
titulo_html = f'''
    <h3 align="center" style="font-size:20px; font-family:Arial,sans-serif;">
        <b>Unidades Consumidoras mapeadas no raio de {RAIO_M/1000:.0f} km</b>
    </h3>
'''
mapa.get_root().html.add_child(folium.Element(titulo_html))

# Salvamos o arquivo final
mapa.save("ucs_raio_3km.html")
print("Pronto! Mapa salvo como 'ucs_raio_3km.html'.")