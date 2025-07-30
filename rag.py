import os
from dotenv import load_dotenv

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import streamlit as st

# .env ファイルから環境変数を読み込む
load_dotenv()

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # Neo4j接続情報
        self.NEO4J_URI = os.getenv("NEO4J_URI")
        self.NEO4J_USER = os.getenv("NEO4J_USER")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

        # モデルの設定
        self.model = ChatOllama(model="gemma3:4b")
        
        # テキスト分割器の設定
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        
        # プロンプトテンプレート設定 (韓国語と日本語の回答)
        self.prompt = PromptTemplate.from_template("""
        <start_of_turn>user
        Please answer in Korean and Japanese.
        韓国語と日本語で答えてください。
        Question: {question}
        Context: {context}
        <end_of_turn>
        <start_of_turn>model
        Here is the answer based on the provided context:
        ---
        Reference Context:
        {context}
        """)

    def ingest(self, pdf_file_path: str):
        # PDFをロード
        st.write(f"📄 PDF 파일 로드 시작: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        st.write(f"✅ PDF 로드 완료, 문서 수: {len(docs)}")

        # テキスト分割
        st.write("✂️ 텍스트 분할 시작")
        chunks = self.text_splitter.split_documents(docs)
        st.write(f"✅ 텍스트 분할 완료, 청크 수: {len(chunks)}")

        # 埋め込みの作成
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="RETRIEVAL_DOCUMENT")

        # デバッグ用: chunksの型と最初の要素を確認
        print(f"Type of chunks: {type(chunks)}")
        if chunks:
            print(f"Type of first chunk: {type(chunks[0])}")
            print(f"Content of first chunk (if possible): {chunks[0]}")

        # テキストと埋め込みを保存
        text_embeddings = [(chunk.page_content, embeddings.embed_query(chunk.page_content)) for chunk in chunks]
        metadatas = [{'reference_text': chunk.page_content} for chunk in chunks]

        # Neo4jに保存
        self.vector_store = Neo4jVector.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings,
            metadatas=metadatas,
            url=self.NEO4J_URI,
            username=self.NEO4J_USER,
            password=self.NEO4J_PASSWORD
        )
        st.write("✅ Neo4j 벡터 스토어에 저장 완료")

        # リトリバーの設定
        self.retriever = self.vector_store.as_retriever()
        st.write("✅ 리트리버 설정 완료")

        # チャットチェーン設定
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        st.write("✅ 챗 체인 생성 완료")

    def ask(self, query: str):
        if not self.chain:
            return "먼저 PDF 파일을 추가해주세요."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

