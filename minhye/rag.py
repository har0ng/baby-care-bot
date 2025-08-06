import os
from dotenv import load_dotenv
import nest_asyncio
import streamlit as st
import uuid

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import tool

nest_asyncio.apply()
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

class ChatPDF:
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.chat_history = []
        self.current_user_id = None

        # 모델명 "chat-bison-001" 사용
        self.model = ChatGoogleGenerativeAI(model="models/chat-bison-001", temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

    def set_user(self, user_id: str):
        self.current_user_id = user_id
        if self.vector_store:
            with self.vector_store._driver.session() as session:
                session.run("MERGE (:User {id: $uid})", uid=user_id)
        else:
            st.warning("경고: PDF가 업로드되지 않아 사용자 노드를 생성할 수 없습니다. 먼저 ingest를 실행하세요.")

    def ingest(self, pdf_file_path: str):
        st.write(f"📄 PDF 로드 시작: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        st.write(f"✅ PDF 로드 완료, 문서 수: {len(docs)}")

        st.write("✂️ 텍스트 분할 중")
        chunks = self.text_splitter.split_documents(docs)
        st.write(f"✅ 텍스트 분할 완료, 청크 수: {len(chunks)}")

        st.write("🔍 메타데이터 필터링 중")
        chunks = filter_complex_metadata(chunks)
        st.write(f"✅ 메타데이터 필터링 완료, 청크 수: {len(chunks)}")

        st.write("⚙️ 임베딩 모델 초기화")
        embeddings_model = GoogleGenerativeAIEmbeddings(model="embedding-001", task_type="RETRIEVAL_DOCUMENT")
        st.write("✅ 임베딩 모델 초기화 완료")

        st.write("🔢 임베딩 생성 중")
        embed_info = []
        for i, chunk in enumerate(chunks):
            embedding_vector = embeddings_model.embed_query(chunk.page_content)
            embed_info.append({
                'text': chunk.page_content,
                'embedding': embedding_vector,
                'reference_text': chunk.metadata.get('source', f'chunk_{i}')
            })
        st.write(f"✅ {len(embed_info)}개의 임베딩 생성 완료")

        st.write("📊 Neo4j 벡터 스토어에 업로드 중")
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

        self.setup_agent()
        st.write("✅ 에이전트 설정 완료")

        if self.current_user_id:
            self.set_user(self.current_user_id)

    def setup_agent(self):
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            st.error("오류: Google API 키 또는 CSE ID가 설정되지 않았습니다. .env 파일을 확인하세요.")
            return

        search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

        google_search_tool = Tool(
            name="google_search",
            description="최신 정보나 웹 검색이 필요할 때 사용합니다.",
            func=search.run,
        )

        # 여기서 docstring 꼭 넣기
        @tool("rag_search")
        def rag_search(query: str):
            """
            PDF 문서 내에서 관련 정보를 검색합니다.
            """
            if not self.retriever:
                return "PDF가 업로드되지 않았습니다. 먼저 ingest 해주세요."
            try:
                docs = self.retriever.get_relevant_documents(query)
                if not docs:
                    return "PDF에서 관련 정보를 찾을 수 없습니다."
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                return f"PDF 검색 중 오류 발생: {e}"

        tools = [google_search_tool, rag_search]

        prompt_template = PromptTemplate.from_template(
            """
            <s> [INST] 당신은 사용자 질문에 답하는 친절하고 유능한 어시스턴트입니다.
            주어진 질문에 답하기 위해 어떤 도구를 사용할지 신중하게 판단하세요.
            만약 PDF 문서에 정보가 있다면 'rag_search' 도구를 사용하고,
            최신 정보나 일반적인 웹 검색이 필요하다면 'google_search' 도구를 사용하세요.

            아래는 지금까지의 대화입니다:
            {history}

            이번 질문: {input}
            {agent_scratchpad}
            [/INST]
            """
        )

        agent = create_tool_calling_agent(self.model, tools, prompt_template)
        self.chain = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    def get_past_user_interactions(self, user_id: str, limit: int = 3):
        if not self.vector_store:
            return []
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
            return interactions[::-1]

    def _format_history(self, history: list):
        formatted_history = []
        for item in history:
            formatted_history.append(f"사용자: {item['question']}")
            formatted_history.append(f"어시스턴트: {item['answer']}")
        return "\n".join(formatted_history)

    def ask(self, query: str):
        if not self.chain:
            st.warning("경고: 에이전트가 설정되지 않았습니다. PDF를 먼저 업로드하세요.")
            return "에이전트가 설정되지 않았습니다. 먼저 PDF를 업로드해주세요."

        past_history_data = []
        if self.current_user_id:
            past_history_data = self.get_past_user_interactions(self.current_user_id)
        formatted_history = self._format_history(past_history_data)

        question_id = str(uuid.uuid4())
        if self.vector_store:
            with self.vector_store._driver.session() as session:
                session.run(
                    """
                    MERGE (q:Question {id: $qid})
                    SET q.text = $qtext, q.timestamp = datetime()
                    """,
                    qid=question_id, qtext=query
                )
        else:
            st.warning("경고: PDF가 업로드되지 않아 질문을 저장할 수 없습니다.")

        try:
            response = self.chain.invoke({"input": query, "history": formatted_history})
            answer_text = response['output']
        except Exception as e:
            st.error(f"에이전트 실행 중 오류 발생: {e}")
            answer_text = "죄송합니다. 질문 처리 중 오류가 발생했습니다. 다시 시도해 주세요."

        answer_id = str(uuid.uuid4())
        if self.vector_store:
            with self.vector_store._driver.session() as session:
                session.run(
                    """
                    MERGE (a:Answer {id: $aid})
                    SET a.text = $atext, a.timestamp = datetime()
                    MERGE (q:Question {id: $qid})
                    MERGE (q)-[:RECEIVED_ANSWER]->(a)
                    """,
                    aid=answer_id, atext=answer_text, qid=question_id
                )
                if self.current_user_id:
                    session.run(
                        """
                        MATCH (u:User {id: $uid}), (q:Question {id: $qid})
                        MERGE (u)-[:ASKED]->(q)
                        """,
                        uid=self.current_user_id, qid=question_id
                    )
        else:
            st.warning("경고: PDF가 업로드되지 않아 답변을 저장할 수 없습니다.")

        self.chat_history.append((query, answer_text))
        return answer_text

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.chat_history = []
        self.current_user_id = None
        st.write("초기화 완료!")
