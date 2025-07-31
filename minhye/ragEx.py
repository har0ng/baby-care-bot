import os
from dotenv import load_dotenv # .env 파일 로드를 위해 추가

# .env 파일에서 환경 변수를 로드합니다.
# 이 함수는 가능한 한 빨리 호출되어야 합니다.
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

# FastEmbedEmbeddings는 더 이상 사용하지 않지만, import 자체는 남아있어도 오류는 아님.
# from langchain_community.embeddings import FastEmbedEmbeddings
# GoogleGenerativeAIEmbeddings를 사용하기 위해 이 줄이 필요합니다.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # ChatOllama 모델 설정
        self.model = ChatOllama(model="gemma3:4b")
        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # 프롬프트 템플릿 설정 (한국어/일본어 답변 지시 포함)
        self.prompt = PromptTemplate.from_template(
        """
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
        """
        )

    def ingest(self, pdf_file_path: str):
        st.write(f"📄 PDF 파일 로드 시작: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        st.write(f"✅ PDF 로드 완료, 문서 수: {len(docs)}")

        st.write("✂️ 텍스트 분할 시작")
        chunks = self.text_splitter.split_documents(docs)
        st.write(f"✅ 텍스트 분할 완료, 청크 수: {len(chunks)}")

        st.write("🔍 메타데이터 필터링 시작")
        chunks = filter_complex_metadata(chunks)
        st.write(f"✅ 메타데이터 필터링 완료, 필터링 후 청크 수: {len(chunks)}")

        st.write("⚙️ 임베딩 및 벡터 스토어 생성 시작")

        # --- GOOGLE_API_KEY 로드 확인 코드 (정확한 들여쓰기) ---
        api_key_check = os.getenv("GOOGLE_API_KEY")
        if api_key_check:
            st.write(f"✅ GOOGLE_API_KEY 로드됨 (일부 문자열 숨김): {api_key_check[:5]}...{api_key_check[-5:]}")
        else:
            st.write("❌ GOOGLE_API_KEY가 로드되지 않았습니다! .env 파일 또는 환경 변수를 확인하세요.")
        # --- 확인 코드 끝 ---

        # Chroma 벡터 스토어 생성 (documents=chunks 중복 제거)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="RETRIEVAL_DOCUMENT")
        )
        st.write("✅ 벡터 스토어 생성 완료")

        # 리트리버 설정
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        st.write("✅ 리트리버 설정 완료")

        # 체인 설정
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        st.write("✅ 챗 체인 생성 완료")

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None