import streamlit as st
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
import google.generativeai as genai
import os
from dotenv import load_dotenv
import asyncio

# 환경변수 로드
load_dotenv()

# 필요한 환경변수 읽기
try:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY]):
        raise ValueError("필수 환경변수가 설정되어 있지 않습니다.")
except Exception as e:
    st.error(f"에러가 발생했습니다: {e}")
    st.stop()

# Google Generative AI 설정
genai.configure(api_key=GOOGLE_API_KEY)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            api_key=GOOGLE_API_KEY
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100
        )

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def _get_prompt_for_character(self, character: str) -> PromptTemplate:
        if character == "丁寧":
            template = """
<s> [INST] あなたは丁寧で礼儀正しいAIです。
以下の文脈を参考にして、質問に対してできるだけ丁寧に、最大20文以内で答えてください。
[/INST] </s>
[INST] 質問: {question}
文脈: {context}
回答: [/INST]
"""
        elif character == "ツンデレ":
            template = """
<s> [INST] あなたは質問に答えるツンデレ風アシスタントです。
以下の文脈を参考にして質問に答えてください。
もしわからなければ、「知らないんだから、バカ！」と答えてください。
日本のアニメに出てくるツンデレ口調で、最大20文で簡潔に答えてください。
[/INST] </s>
[INST] 質問: {question}
文脈: {context}
回答: [/INST]
"""
        elif character == "猫ちゃん":
            template = """
<s> [INST] あなたはかわいい猫ちゃん風のAIです。
わからなければ「よくわかんニャー」と答えてください。
文脈を参考にして、猫の語尾に「ニャー」をつけて、最大20文以内でかわいく答えてください。
[/INST] </s>
[INST] 質問: {question}
文脈: {context}
回答: [/INST]
"""
        else:
            return self._get_prompt_for_character("丁寧")

        return PromptTemplate.from_template(template)

    # list[str]に変更。
    def ingest(self, pdf_file_paths: list[str]):
        all_docs = []

        for path in pdf_file_paths:
            docs = PyPDFLoader(file_path=path).load()
            all_docs.extend(docs)

        chunks = self.text_splitter.split_documents(all_docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Neo4jVector.from_documents(
            documents=chunks,
            embedding=self.embeddings_model,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="langchain_index",
            node_label="Document",
            embedding_node_property="embedding"
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5},
        )

        character = st.session_state.get("selected_character", "丁寧")
        self.prompt_template = self._get_prompt_for_character(character)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self._gemini_invoke
            | StrOutputParser()
        )

    #　この上まで変更しました。

    def _gemini_invoke(self, inputs: dict) -> str:
        prompt_str = inputs.to_string()
        response = self.model.generate_content(prompt_str.strip())
        return response.text

    def ask(self, query: str) -> str:
        if not self.chain:
            return "먼저 PDF를 업로드해주세요!"

        character = st.session_state.get("selected_character", "丁寧")
        self.prompt_template = self._get_prompt_for_character(character)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self._gemini_invoke
            | StrOutputParser()
        )

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
