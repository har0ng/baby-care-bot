import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
import nest_asyncio
import streamlit as st
import uuid

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# import settings as sets  # NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD 포함

# 중첩 이벤트 루프 방지
nest_asyncio.apply()

# 환경변수 로드
load_dotenv()

class ChatPDF:
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.chat_history = []
        self.current_user_id = None

        # LLM 모델 및 텍스트 분할기 설정
        self.model = ChatOllama(model="gemma3:4b")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        # 프롬프트 템플릿 (역할: 츤데레 어시스턴트, 한국어+일본어 답변)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] 당신은 질문에 대답하는 츤데레 어시스턴트야.
            아래는 지금까지의 대화야:

            {history}

            이번 질문: {question}
            참고 자료 (컨텍스트): {context}
            일본어와 한국어로 3문장 이내로 귀엽게 대답해줘! [/INST]
            """
        )

    def set_user(self, user_id: str):
        self.current_user_id = user_id
        # Neo4j에 User 노드가 없으면 생성
        with self.vector_store._driver.session() as session:
            session.run("MERGE (:User {id: $uid})", uid=user_id)

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

        st.write("⚙️ 임베딩 모델 초기화...")
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="RETRIEVAL_DOCUMENT")
        st.write("✅ 임베딩 모델 초기화 완료")

        st.write("🔢 청크 임베딩 생성 중...")
        embed_info = []
        for i, chunk in enumerate(chunks):
            embedding_vector = embeddings_model.embed_query(chunk.page_content)
            embed_info.append({
                'text': chunk.page_content,
                'embedding': embedding_vector,
                'reference_text': chunk.metadata.get('source', f'chunk_{i}')
            })
        st.write(f"✅ {len(embed_info)}개의 청크 임베딩 생성 완료")

        st.write("📊 Neo4j에 임베딩 업로드 준비 중...")
        text_embeddings = [(e['text'], e['embedding']) for e in embed_info]
        metadatas = [{'reference_text': e['reference_text']} for e in embed_info]

        self.vector_store = Neo4jVector.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings_model,
            metadatas=metadatas,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            database="neo4j",
            index_name="langchain_index",
            node_label="Document",
            embedding_node_property="embedding"
        )
        st.write("✅ 벡터 스토어 생성 완료")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5}
        )
        st.write("✅ 리트리버 설정 완료")

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        st.write("✅ 챗 체인 생성 완료")

        # 사용자 노드 생성 (초기화 후)
        if self.current_user_id:
            self.set_user(self.current_user_id)

    def get_past_user_interactions(self, user_id: str, limit: int = 3):
        with self.vector_store._driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $uid})-[:ASKED]->(q:Question)-[:RECEIVED_ANSWER]->(a:Answer)
                RETURN q.text AS question, a.text AS answer
                ORDER BY q.timestamp DESC
                LIMIT $limit
                """,
                uid=user_id,
                limit=limit
            )
            interactions = result.data()
            return interactions[::-1]  # 시간 순서대로 정렬

    def _enrich_query(self, query: str, history: list):
        if not history:
            return query
        history_text = "\n".join([f"Q{i+1}: {h['question']}\nA{i+1}: {h['answer']}" for i, h in enumerate(history)])
        return f"{history_text}\n\n현재 질문: {query}"

    def ask(self, query: str):
        if not self.chain:
            return "먼저 PDF 파일을 업로드해주세요."

        # if not self.current_user_id:
        #     return "사용자 ID가 설정되지 않았습니다."

        # 질문 저장
        question_id = str(uuid.uuid4())
        with self.vector_store._driver.session() as session:
            session.run(
            """
            MERGE (q:Question {id: $qid})
            SET q.text = $qtext, q.timestamp = datetime()
            """,
            qid=question_id, qtext=query
        )

        # 과거 대화 불러오기 및 쿼리 강화
        past_history = []
        enriched_query = self._enrich_query(query, past_history)

        # 체인 실행
        response = self.chain.invoke({"question": enriched_query, "history": ""})

        # 답변 저장
        answer_id = str(uuid.uuid4())
        with self.vector_store._driver.session() as session:
            session.run(
                """
                MERGE (a:Answer {id: $aid})
                SET a.text = $atext, a.timestamp = datetime()
                MERGE (q:Question {id: $qid})
                MERGE (q)-[:RECEIVED_ANSWER]->(a)
                """,
                aid=answer_id, atext=response, qid=question_id
            )

        # 대화 히스토리 업데이트 (optional)
        self.chat_history.append((query, response))

        return response

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.chat_history = []
        st.write("초기화 완료!")

