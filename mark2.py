import streamlit as st
import sqlite3
from PyPDF2 import PdfReader
from docx import Document
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
import re

load_dotenv()
genai.configure(api_key=os.getenv("OPENAI_API_KEY"))

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def init_db():
    conn = sqlite3.connect('songs_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS songs
                 (id INTEGER PRIMARY KEY, name TEXT, content TEXT, theme TEXT)''')
    conn.commit()
    return conn

def song_exists(conn, name):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM songs WHERE name = ?", (name,))
    return c.fetchone()[0] > 0

def save_song(conn, name, content, theme):
    c = conn.cursor()
    c.execute("INSERT INTO songs (name, content, theme) VALUES (?, ?, ?)", (name, content, theme))
    conn.commit()

def get_all_songs(conn):
    c = conn.cursor()
    c.execute("SELECT name, content, theme FROM songs ORDER BY name ASC")
    return c.fetchall()

@st.cache_data
def get_file_text(file):
    if file.name.lower().endswith('.pdf'):
        return get_pdf_text(file)
    elif file.name.lower().endswith('.docx'):
        return get_docx_text(file)
    else:
        return ""

@st.cache_data
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for i, page in enumerate(pdf_reader.pages):
        text += f"\n--- Página {i+1} del documento {pdf_file.name} ---\n"
        text += page.extract_text()
    return text

@st.cache_data
def get_docx_text(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(_embeddings, text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=_embeddings)
    return vector_store

def get_song_analysis_chain():
    prompt_template = """
    Analiza el contenido de esta canción y determina su tema principal.
    Proporciona un breve resumen del tema y el tono emocional de la canción.

    Contenido de la canción:
    {context}

    Tema y análisis:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def analyze_song(content):
    chain = get_song_analysis_chain()
    # Creamos un LangchainDocument con el contenido de la canción
    doc = LangchainDocument(page_content=content)
    response = chain.invoke(
        {"input_documents": [doc], "question": ""}
    )
    return response["output_text"]

def get_playlist_creation_chain():
    prompt_template = """
    Basándote en los temas de las canciones proporcionadas y el tema solicitado por el usuario, 
    crea una lista de 2 canciones coherente. Selecciona solo las canciones que mejor se ajusten al tema solicitado.
    Explica brevemente por qué estas canciones van bien juntas 
    y cómo se relacionan con el tema solicitado.

    Tema solicitado por el usuario: {user_theme}

    Canciones disponibles y sus temas:
    {context}

    Lista de reproducción:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",system_instructions = "Eres un experto en música cristiana, del sector reformado, no catolico. Crees en la soberanía de Dios, entiendes a la biblia como la palabra de Dios y crees en Jesus como el Hijo de DIos que murió en la Cruz y resucito al tercer día. Entiendes que las canciones no deben dibagar en su temática y tienes que poner La Palabra de DIos en lo más alto", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_theme", "context"])    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def create_playlist(songs_with_themes, user_theme):
    chain = get_playlist_creation_chain()
    context = "\n".join([f"Canción: {song[0]}, Tema: {song[2]}" for song in songs_with_themes])
    doc = LangchainDocument(page_content=context)
    response = chain.invoke(
        {"input_documents": [doc], "user_theme": user_theme, "question": ""}
    )
    return response["output_text"]

def display_songs(conn, search_term=""):
    songs = get_all_songs(conn)
    if songs:
        st.subheader("Canciones Disponibles")
        
        # Añadir campo de búsqueda
        search_term = st.text_input("Buscar canción:", value=search_term)
        
        if search_term:
            # Crear un patrón de búsqueda más flexible
            search_pattern = re.compile(r'.*'.join(re.escape(term.lower()) for term in search_term.split()), re.IGNORECASE)
            
            # Filtrar canciones basadas en el patrón de búsqueda
            filtered_songs = [song for song in songs if search_pattern.search(song[0])]
        else:
            filtered_songs = songs
        
        if filtered_songs:
            for i, song in enumerate(filtered_songs, 1):
                st.write(f"{i}. {song[0]}")
                
            st.info(f"Mostrando {len(filtered_songs)} de {len(songs)} canciones.")
        else:
            st.info("No se encontraron canciones que coincidan con la búsqueda.")
    else:
        st.info("No hay canciones en la base de datos. Por favor, carga nuevas canciones primero.")

def main():
    st.set_page_config(page_title="Analizador de Canciones y Creador de Playlists", layout="wide")

    st.markdown(
        """
        <style>
        /* ... (estilos previos) ... */
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <h1 style="text-align: center; animation: colorChange 5s infinite;">
        Bienvenido soy Mark2 y voy a ayudarte a analizar Canciones y Crear listas para cantar en las reuniones
        </h1>
        <h3 style='text-align: center; color: #888888;'>Carga tus canciones y crea playlists temáticas</h3>
        """,
        unsafe_allow_html=True
    )

    conn = init_db()
    embeddings = get_embeddings()

    menu = ["Inicio", "Ver Canciones", "Cargar Nuevas Canciones", "Crear Playlist"]
    choice = st.sidebar.selectbox("Menú", menu)

    if choice == "Inicio":
        st.write("Bienvenido a la aplicación de análisis de canciones y creación de playlists. Utiliza el menú de la izquierda para navegar.")

    elif choice == "Ver Canciones":
        display_songs(conn)

    elif choice == "Cargar Nuevas Canciones":
        st.subheader("Cargar nuevas canciones")
        uploaded_files = st.file_uploader("Sube archivos PDF o Word, o selecciona una carpeta", accept_multiple_files=True, type=['pdf', 'docx'])
        
        if st.button("Procesar Archivos"):
            if uploaded_files:
                with st.spinner("Procesando archivos... Por favor, espera."):
                    for file in uploaded_files:
                        if not song_exists(conn, file.name):
                            content = get_file_text(file)
                            theme = analyze_song(content)
                            save_song(conn, file.name, content, theme)
                            st.success(f"Canción '{file.name}' procesada y guardada.")
                        else:
                            st.warning(f"La canción '{file.name}' ya existe en la base de datos.")
                
                st.success("Todas las canciones han sido procesadas. Puedes crear playlists ahora.")
            else:
                st.warning("Por favor, sube al menos un archivo antes de procesar.")

    elif choice == "Crear Playlist":
        st.subheader("Crear Playlist")
        songs = get_all_songs(conn)
        if songs:
            user_theme = st.text_input("Ingresa el tema para tu playlist:")
            
            if st.button("Generar Playlist"):
                if user_theme:
                    playlist = create_playlist(songs, user_theme)
                    st.write("Playlist generada:", playlist)
                else:
                    st.warning("Por favor, ingresa un tema para la playlist antes de generarla.")
        else:
            st.warning("No hay canciones en la base de datos. Por favor, carga nuevas canciones primero.")

    conn.close()

if __name__ == "__main__":
    main()