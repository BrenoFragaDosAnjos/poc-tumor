# Classificador de Tumor Cerebral em Ressonância Magnética — Prova de Conceito

Prova de conceito para classificar imagens de ressonância magnética cerebral em quatro categorias utilizando um modelo de Machine Learning e uma interface em Streamlit.

## Classes

A aplicação trabalha com as seguintes classes de saída:

- glioma;
- meningioma;
- ausência de tumor;
- tumor de hipófise.

## Fluxo da aplicação

```text
Imagem de RMI
   |
   v
Upload no Streamlit
   |
   v
Redimensionamento + normalização
   |
   v
Modelo TensorFlow Lite
   |
   v
Inferência
   |
   v
Probabilidades por classe
   |
   v
Gráfico interativo com Plotly
```

## Tecnologias utilizadas

- Python
- TensorFlow Lite
- experimentos com PyTorch
- Streamlit
- NumPy
- Pandas
- Plotly
- Pillow

## Estrutura do repositório

```text
.
├── app.py
├── model.pth
├── requirements.txt
├── teste.py
├── trusted/
└── README.md
```

O arquivo `app.py` é a aplicação principal em Streamlit e utiliza o fluxo de inferência com TensorFlow Lite.

O arquivo `teste.py` mantém uma implementação experimental em PyTorch para testes e evolução do modelo.

## Executando localmente

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente e instale as dependências:

```bash
pip install -r requirements.txt
```

Execute a aplicação:

```bash
streamlit run app.py
```

## Carregamento do modelo

A aplicação baixa o modelo TensorFlow Lite quando ele não está disponível localmente e inicializa o interpretador TFLite para realizar a inferência.

## Estado atual

Este repositório é uma **prova de conceito**, não um dispositivo médico.

Os resultados do modelo não devem ser utilizados para diagnóstico, definição de tratamento ou qualquer decisão clínica.

## Melhorias planejadas

- separar inferência, pré-processamento e interface em módulos distintos;
- documentar o processo de treinamento do modelo;
- documentar origem e licença do dataset;
- adicionar métricas como precisão, recall, F1-score e matriz de confusão;
- criar testes automatizados para pré-processamento e inferência;
- containerizar a aplicação;
- fixar versões das dependências para melhorar a reprodutibilidade.

## Valor como projeto de portfólio

O projeto demonstra um fluxo aplicado de visão computacional que conecta pré-processamento de imagens, inferência de modelo, interface para o usuário e visualização dos resultados.
