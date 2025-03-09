import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generar_respuesta(txt):
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )
    divisor_texto = CharacterTextSplitter()
    textos = divisor_texto.split_text(txt)
    documentos = [Document(page_content=t) for t in textos]
    cadena = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return cadena.run(documentos)

st.set_page_config(
    page_title="Resumidor de Texto"
)
st.title("Resumidor de Texto")

entrada_texto = st.text_area(
    "Ingresa tu texto",
    "",
    height=200
)

resultado = []
with st.form("formulario_resumir", clear_on_submit=True):
    openai_api_key = st.text_input(
        "Clave API de OpenAI",
        type="password",
        disabled=not entrada_texto
    )
    enviado = st.form_submit_button("Enviar")
    if enviado and openai_api_key.startswith("sk-"):
        respuesta = generar_respuesta(entrada_texto)
        resultado.append(respuesta)
        del openai_api_key

if len(resultado):
    st.info(respuesta)
