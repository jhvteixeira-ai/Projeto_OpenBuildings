import pandas as pd

# Nome do arquivo CSV de entrada
arquivo_entrada = 'lista_ucs_fernanda_joao.xlsx'
arquivo_saida = 'ucs_separadas_por_tipo.xlsx'

# O Pandas carrega a base de dados inteira para a memória de uma vez
df = pd.read_excel(arquivo_entrada)

# Cria uma coluna temporária pegando as duas primeiras letras da coluna 'Classe'
# Exemplo: 'RE1' vira 'RE', 'CO9' vira 'CO'
df['Categoria'] = df['Classe'].astype(str).str[:2]

# Dicionário mapeando as siglas para os nomes completos que irão nomear as abas
mapa_categorias = {
    'RE': 'Residencial',
    'CO': 'Comercial',
    'IN': 'Industrial',
    'RU': 'Rural',
    'IP': 'Iluminacao_Publica',
    'PP': 'Poder_Publico'
}

# Aplica o mapeamento. Se encontrar uma sigla diferente, mantém a própria sigla
df['Tipo_Aba'] = df['Categoria'].map(mapa_categorias).fillna(df['Categoria'])

# Abre o criador de arquivos Excel
with pd.ExcelWriter(arquivo_saida, engine='openpyxl') as writer:
    # Agrupa os dados pelo tipo de aba que acabamos de descobrir
    for tipo, grupo in df.groupby('Tipo_Aba'):
        
        # Remove as colunas auxiliares que criamos para não poluir a planilha final
        grupo_limpo = grupo.drop(columns=['Categoria', 'Tipo_Aba'])
        
        # Grava o grupo na sua respectiva aba
        grupo_limpo.to_excel(writer, sheet_name=tipo, index=False)

print(f"Sucesso! Arquivo '{arquivo_saida}' gerado com as abas separadas.")