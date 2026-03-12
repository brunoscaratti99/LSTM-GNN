# Previsão de Precipitação com GNN + LSTM

Este repositório apresenta uma implementação de um modelo **Graph Neural Network (GNN) combinado com Long Short-Term Memory (LSTM)** para **previsão de precipitação utilizando dados de estações meteorológicas**.

O modelo explora:

- **dependências espaciais entre estações meteorológicas** através de grafos
- **dependências temporais das variáveis meteorológicas** através de redes recorrentes

Essa abordagem permite realizar **aprendizado espaço-temporal**, sendo adequada para problemas de previsão meteorológica e eventos climáticos extremos.

---

# Motivação

Eventos de precipitação intensa têm se tornado mais frequentes devido às mudanças climáticas, podendo causar:

- enchentes
- deslizamentos de terra
- danos à infraestrutura

Modelos tradicionais muitas vezes apresentam dificuldades para capturar simultaneamente:

- relações espaciais entre diferentes estações
- evolução temporal das variáveis atmosféricas

As **Graph Neural Networks (GNN)** permitem modelar relações espaciais em forma de grafos, enquanto **LSTM** são eficazes para aprender padrões temporais em séries temporais.

A combinação dessas duas arquiteturas permite capturar **dependências espaço-temporais complexas**.

---

# Arquitetura do Modelo

O modelo possui dois componentes principais.

## GNN – Aprendizado Espacial

A GNN modela a relação espacial entre as estações meteorológicas.

- Cada **estação é representada como um nó do grafo**
- As **arestas representam relações espaciais** (por exemplo, distância ou correlação)

A GNN gera **representações espaciais (embeddings)** para cada estação.

---

## LSTM – Aprendizado Temporal

A LSTM recebe sequências temporais das variáveis meteorológicas e aprende a evolução dos padrões atmosféricos ao longo do tempo.

Entrada típica:

```
[variáveis meteorológicas de t-H até t]
```

Saída:

```
precipitação em t+1
```

---

## Fluxo Geral do Modelo

```
Variáveis meteorológicas
        │
        ▼
Graph Neural Network
 (relações espaciais)
        │
        ▼
      LSTM
 (dinâmica temporal)
        │
        ▼
Previsão de precipitação
```

---

# Estrutura do Repositório

```
LSTM-GNN/

├── data/              # dados e pré-processamento
├── models/            # arquiteturas GNN e LSTM
├── training/          # scripts de treinamento
├── evaluation/        # métricas e avaliação
├── notebooks/         # experimentos e análise exploratória
├── utils/             # funções auxiliares
└── README.md
```

---

# Variáveis de Entrada

As variáveis meteorológicas utilizadas podem incluir:

- temperatura do ar
- ponto de orvalho (dew point)
- pressão atmosférica
- umidade relativa
- velocidade do vento
- precipitação passada

Essas variáveis são fornecidas como **séries temporais para cada estação meteorológica**.

---

# Instalação

Clone o repositório:

```bash
git clone https://github.com/brunoscaratti99/LSTM-GNN.git
cd LSTM-GNN
```

Crie um ambiente virtual:

```bash
conda create -n gnn_lstm python=3.10
conda activate gnn_lstm
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Principais bibliotecas utilizadas:

- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Xarray
- Scikit-learn

---

# Treinamento

Exemplo de execução do treinamento:

```python
python train.py
```

Etapas típicas do treinamento:

1. Pré-processamento dos dados
2. Construção do grafo das estações
3. Geração de janelas temporais (sliding windows)
4. Treinamento do modelo
5. Validação

---

# Métricas de Avaliação

O desempenho do modelo pode ser avaliado utilizando:

- **MAE** – Mean Absolute Error
- **MSE** – Mean Squared Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coeficiente de determinação

---

# Trabalhos Futuros

Possíveis extensões do projeto incluem:

- uso de **Graph Attention Networks (GAT)**
- **matriz de adjacência dinâmica**
- previsão **multi-passos**
- detecção de **eventos extremos de precipitação**
- integração com **dados de reanálise climática (ERA5)**

---

# Citação

Caso utilize este repositório em sua pesquisa, cite:

```bibtex
@software{lstm_gnn_precipitation,
  author = {Veloso, Bruno},
  title = {Previsão de Precipitação com GNN + LSTM},
  year = {2026},
  url = {https://github.com/brunoscaratti99/LSTM-GNN}
}
```

---

# Licença

Este projeto está disponível sob a licença **MIT**.