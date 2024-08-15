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

import gdown
import torch
import streamlit as st
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import io
import plotly.express as px

# Definição do modelo (ajuste conforme sua arquitetura)
class MeuModelo(nn.Module):
    def __init__(self):
        super(MeuModelo, self).__init__()
        # Defina a arquitetura do modelo aqui
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*300*300, 4)  # Ajuste o tamanho do input e output conforme necessário

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Achata o tensor para o Fully Connected layer
        x = self.fc1(x)
        return x

# Função para carregar o modelo PyTorch
@st.cache_resource
def carrega_modelo():
    file_id = "1B7gL1-Vq7fqKQy5viCNBQl-dMXAA3KEy"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "modelo.pth"

    def download_file_from_google_drive(url, destination):
        gdown.download(url, destination, quiet=False)

    if not os.path.exists(output):
        download_file_from_google_drive(url, output)

    modelo = MeuModelo()  # Instancie o modelo
    modelo.load_state_dict(torch.load(output, map_location=torch.device('cpu')))  # Carregue o estado
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





