# Análise Geoespacial de Edificações via OpenBuildings API 🗺️

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-1.x-green.svg)](https://geopandas.org/)
[![Folium](https://img.shields.io/badge/Folium-Map-orange.svg)](https://python-visualization.github.io/folium/)

## 📌 Sobre o Projeto

Este repositório contém o código-fonte desenvolvido como parte das atividades de Iniciação Científica. O objetivo principal do script é realizar a extração, o tratamento e a visualização de *footprints* (polígonos de telhados) de edificações urbanas utilizando a base de dados do **Google OpenBuildings**. 

O script foi otimizado para lidar com grandes volumes de dados geoespaciais, processando arquivos pesados em lotes (*chunks*) e mesclando fragmentos de telhados para estimar com precisão a área construída em uma região de estudo delimitada (neste caso, com raio de 3 km a partir do centro estabelecido).

## ⚙️ Funcionalidades

O algoritmo realiza as seguintes operações em sequência:
1. **Delimitação de Área:** Criação de um *buffer* espacial delimitando a região de estudo a partir de coordenadas centrais (Latitude/Longitude).
2. **Processamento em Lotes (*Chunking*):** Leitura eficiente do arquivo `.gz` nativo, filtrando dados irrelevantes e mantendo o uso de memória RAM baixo.
3. **Filtro de Confiança e Geometria:** Exclusão de polígonos com nível de confiança da IA abaixo de 60% e que estejam fora do *bounding box* da região de interesse.
4. **Mesclagem Espacial:** Uso de técnicas de *buffer* (expansão e retração) combinadas com a função `unary_union` para unificar polígonos de telhados fragmentados em uma mesma edificação.
5. **Cálculo de Área e Limpeza:** Cálculo da área individual (em m²) e remoção de artefatos menores que 30 m².
6. **Mapeamento Interativo:** Geração de um mapa HTML interativo utilizando a biblioteca Folium, permitindo a visualização espacial dos resultados e *tooltips* com a metragem de cada residência.

## 🛠️ Tecnologias e Dependências

Para executar este projeto, você precisará do **Python 3** e das seguintes bibliotecas:

* `pandas`
* `geopandas`
* `shapely`
* `folium`
