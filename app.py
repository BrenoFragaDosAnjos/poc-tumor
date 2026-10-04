from __future__ import annotations

import io
import os

import gdown
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_FILE_ID = "1FreadAikK0I94Fpc_mA8_gKhAZgNRVbV"
MODEL_PATH = "modelo.tflite"
CLASSES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]


def download_model() -> None:
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    result = gdown.download(url, MODEL_PATH, quiet=False)
    if not result or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Não foi possível baixar o modelo TensorFlow Lite.")


@st.cache_resource
def load_model() -> tf.lite.Interpreter:
    if not os.path.exists(MODEL_PATH):
        download_model()

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    st.image(image, caption="Imagem selecionada", use_container_width=True)

    resized = image.resize((300, 300))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(interpreter: tf.lite.Interpreter, image: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    return output.astype(float)


def probability_chart(probabilities: np.ndarray):
    dataframe = pd.DataFrame(
        {
            "Classe": CLASSES,
            "Probabilidade (%)": probabilities * 100,
        }
    )
    return px.bar(
        dataframe,
        y="Classe",
        x="Probabilidade (%)",
        orientation="h",
        text="Probabilidade (%)",
        title="Probabilidade por classe",
    )


def main() -> None:
    st.set_page_config(
        page_title="Classificador de Tumor Cerebral",
        page_icon="🧠",
    )

    st.title("Classificação de tumor cerebral em imagens de RMI")
    st.caption(
        "Proof of concept para demonstração técnica. "
        "Não utilizar para diagnóstico ou decisão clínica."
    )

    uploaded_file = st.file_uploader(
        "Selecione uma imagem",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        return

    try:
        interpreter = load_model()
        image = preprocess_image(uploaded_file)
        probabilities = predict(interpreter, image)

        if len(probabilities) != len(CLASSES):
            raise ValueError(
                f"O modelo retornou {len(probabilities)} classes, "
                f"mas a aplicação espera {len(CLASSES)}."
            )

        best_index = int(np.argmax(probabilities))
        st.subheader("Resultado")
        st.metric(
            "Classe com maior probabilidade",
            CLASSES[best_index],
            f"{probabilities[best_index] * 100:.1f}%",
        )
        st.plotly_chart(
            probability_chart(probabilities),
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Não foi possível concluir a inferência: {exc}")


if __name__ == "__main__":
    main()
