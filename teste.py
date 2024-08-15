# # import streamlit as st
# from PIL import Image
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import pandas as pd
# import io
# import plotly.express as px
# import streamlit as st





# # Função para carregar o modelo PyTorch
# @st.cache_resource
# def carrega_modelo():
#     # Carregar o modelo .pth
#     modelo = torch.load('modelo.pth', map_location=torch.device('cpu'))
#     modelo.eval()  # Colocar o modelo em modo de avaliação
#     return modelo

# # Função para carregar e pré-processar a imagem
# def carrega_imagem():
#     uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
#     if uploaded_file is not None:
#         image_data = uploaded_file.read()
#         image = Image.open(io.BytesIO(image_data))
#         st.image(image)
#         st.success("Imagem carregada com sucesso!")

#         # Redimensionando a imagem para 300x300 pixels
#         transform = transforms.Compose([
#             transforms.Resize((300, 300)),
#             transforms.ToTensor(),  # Converte a imagem para tensor
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normaliza a imagem
#         ])
#         image = transform(image)
#         image = image.unsqueeze(0)  # Adiciona uma dimensão extra para o batch
#         return image

# # Função para fazer a previsão
# def previsao(modelo, image):
#     with torch.no_grad():  # Desativa o cálculo de gradiente
#         output = modelo(image)
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
#         st.write("Saída bruta do modelo:", probabilities.numpy())  # Para debug

#         classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
#         df = pd.DataFrame()
#         df['classes'] = classes
#         df['probabilidades (%)'] = 100 * probabilities.numpy()
#         fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
#                      title='Probabilidade de Tumor Cerebral em RMI')
#         st.plotly_chart(fig)

# def main():
#     st.set_page_config(page_title="Classifica Imagens de RMI", page_icon="🧠")
#     st.write("# Classificação de Tumor Cerebral em imagens de RMI! 🧠")

#     modelo = carrega_modelo()
#     image = carrega_imagem()

#     if image is not None:
#         previsao(modelo, image)

# if __name__ == "__main__":
#     main()

# Importações
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import io
import plotly.express as px
import streamlit as st
import gdown
import os
import gdown
import torch
import streamlit as st
import os

# Função para carregar o modelo PyTorch
@st.cache_resource
def carrega_modelo():
    # URL do arquivo no Google Drive
    file_id = "1B7gL1-Vq7fqKQy5viCNBQl-dMXAA3KEy"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "modelo.pth"

    # Verifique se o arquivo já foi baixado
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # Carregar o modelo
    modelo = torch.load(output, map_location=torch.device('cpu'))
    modelo.eval()  # Colocar o modelo em modo de avaliação
    return modelo

# Função para carregar e pré-processar a imagem
def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.success("Imagem carregada com sucesso!")

        # Redimensionando a imagem para 300x300 pixels
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),  # Converte a imagem para tensor
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normaliza a imagem
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # Adiciona uma dimensão extra para o batch
        return image

# Função para fazer a previsão
def previsao(modelo, image):
    with torch.no_grad():  # Desativa o cálculo de gradiente
        output = modelo(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        st.write("Saída bruta do modelo:", probabilities.numpy())  # Para debug

        classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        df = pd.DataFrame()
        df['classes'] = classes
        df['probabilidades (%)'] = 100 * probabilities.numpy()
        fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                     title='Probabilidade de Tumor Cerebral em RMI')
        st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Classifica Imagens de RMI", page_icon="🧠")
    st.write("# Classificação de Tumor Cerebral em imagens de RMI! 🧠")

    modelo = carrega_modelo()
    image = carrega_imagem()

    if image is not None:
        previsao(modelo, image)

if __name__ == "__main__":
    main()




