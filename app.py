# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import io
# import gdown
# import plotly.express as px

# #https://drive.google.com/file/d/1B7gL1-Vq7fqKQy5viCNBQl-dMXAA3KEy/view?usp=drive_link

# @st.cache_resource
# def carrega_modelo():
#     interpreter = tf.lite.Interpreter(model_path='modelo.tflite')
#     interpreter.allocate_tensors()
#     return interpreter

# def carrega_imagem():
#     uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
#     if uploaded_file is not None:
#         image_data = uploaded_file.read()
#         image = Image.open(io.BytesIO(image_data))
#         st.image(image)
#         st.success("Imagem carregada com sucesso!")

#         # Redimensionando a imagem para 300x300 pixels
#         image = image.resize((300, 300))
#         image = np.array(image, dtype=np.float32)
#         image = image / 255.0  # Normalização para o intervalo [0, 1]
#         image = np.expand_dims(image, axis=0)
#         return image

# def previsao(interpreter, image):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])

#     st.write("Saída bruta do modelo:", output_data)  # Para debug

#     classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
#     df = pd.DataFrame()
#     df['classes'] = classes
#     df['probabilidades (%)'] = 100 * output_data[0]
#     fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
#                  title='Probabilidade de Tumor Cerebral em RMI')
#     st.plotly_chart(fig)

# def main():
#     st.set_page_config(page_title="Classifica Imagens de RMI", page_icon="🧠")
#     st.write("# Classificação de Tumor Cerebral em imagens de RMI! 🧠")

#     interpreter = carrega_modelo()
#     image = carrega_imagem()

#     if image is not None:
#         previsao(interpreter, image)

# if __name__ == "__main__":
#     main()

# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import io
# import gdown
# import plotly.express as px
# import os

# # URL do Google Drive e nome do arquivo
# file_id = "1B7gL1-Vq7fqKQy5viCNBQl-dMXAA3KEy"
# url = f"https://drive.google.com/uc?id={file_id}"
# output = "modelo.tflite"

# # Função para baixar o modelo do Google Drive
# def download_file_from_google_drive(url, destination):
#     gdown.download(url, destination, quiet=False)

# # Função para carregar o modelo TensorFlow Lite
# @st.cache_resource
# def carrega_modelo():
#     # Baixa o modelo se ele não existir
#     if not os.path.exists(output):
#         download_file_from_google_drive(url, output)

#     # Carrega o modelo TensorFlow Lite
#     interpreter = tf.lite.Interpreter(model_path=output)
#     interpreter.allocate_tensors()
#     return interpreter

# # Função para carregar e pré-processar a imagem
# def carrega_imagem():
#     uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
#     if uploaded_file is not None:
#         image_data = uploaded_file.read()
#         image = Image.open(io.BytesIO(image_data))
#         st.image(image)
#         st.success("Imagem carregada com sucesso!")

#         # Redimensionando a imagem para 300x300 pixels
#         image = image.resize((300, 300))
#         image = np.array(image, dtype=np.float32)
#         image = image / 255.0  # Normalização para o intervalo [0, 1]
#         image = np.expand_dims(image, axis=0)
#         return image

# # Função para fazer a previsão com o modelo TensorFlow Lite
# def previsao(interpreter, image):
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     interpreter.set_tensor(input_details[0]['index'], image)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])

#     st.write("Saída bruta do modelo:", output_data)  # Para debug

#     classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
#     df = pd.DataFrame()
#     df['classes'] = classes
#     df['probabilidades (%)'] = 100 * output_data[0]
#     fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
#                  title='Probabilidade de Tumor Cerebral em RMI')
#     st.plotly_chart(fig)

# def main():
#     st.set_page_config(page_title="Classifica Imagens de RMI", page_icon="🧠")
#     st.write("# Classificação de Tumor Cerebral em imagens de RMI! 🧠")

#     interpreter = carrega_modelo()
#     image = carrega_imagem()

#     if image is not None:
#         previsao(interpreter, image)

# if __name__ == "__main__":
#     main()

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import gdown
import plotly.express as px
import os

# URL do Google Drive e nome do arquivo
file_id = "1B7gL1-Vq7fqKQy5viCNBQl-dMXAA3KEy"
url = f"https://drive.google.com/uc?id={file_id}"
output = "modelo.tflite"

# Função para baixar o modelo do Google Drive
def download_file_from_google_drive(url, destination):
    gdown.download(url, destination, quiet=False)

# Função para carregar o modelo TensorFlow Lite
@st.cache_resource
def carrega_modelo():
    # Baixa o modelo se ele não existir
    if not os.path.exists(output):
        download_file_from_google_drive(url, output)

    # Carrega o modelo TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=output)
    interpreter.allocate_tensors()
    return interpreter

# Função para carregar e pré-processar a imagem
def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte uma imagem aqui ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.success("Imagem carregada com sucesso!")

        # Redimensionando a imagem para 300x300 pixels
        image = image.resize((300, 300))
        image = np.array(image, dtype=np.float32)
        image = image / 255.0  # Normalização para o intervalo [0, 1]
        image = np.expand_dims(image, axis=0)  # Adiciona uma dimensão para o batch
        return image

# Função para fazer a previsão com o modelo TensorFlow Lite
def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    st.write("Saída bruta do modelo:", output_data)  # Para debug

    classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]
    fig = px.bar(df, y='classes', x='probabilidades (%)', orientation='h', text='probabilidades (%)',
                 title='Probabilidade de Tumor Cerebral em RMI')
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Classifica Imagens de RMI", page_icon="🧠")
    st.write("# Classificação de Tumor Cerebral em imagens de RMI! 🧠")

    interpreter = carrega_modelo()
    image = carrega_imagem()

    if image is not None:
        previsao(interpreter, image)

if __name__ == "__main__":
    main()

