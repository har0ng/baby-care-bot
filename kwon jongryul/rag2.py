import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader # UnstructuredFileLoader 임포트

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from collections import defaultdict

import pickle # 현재 코드에서 직접 사용되지 않지만, 이전 컨텍스트에서 언급되었으므로 유지
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_ollama import OllamaEmbeddings, ChatOllama # OllamaEmbeddings는 사용되지 않는 것 같지만, 일단 유지
from PIL import Image # 사용되지 않는 것 같지만, 일단 유지
from tqdm.notebook import tqdm # 사용되지 않는 것 같지만, 일단 유지

import settings as sets # Neo4j 연결 정보를 가져오기 위함
import io
import os
from dotenv import load_dotenv
import asyncio
import threading
import nest_asyncio # 비동기 이벤트 루프 충돌 해결을 위함

# nest_asyncio를 적용하여 중첩된 이벤트 루프를 허용합니다.
# 이 코드는 Streamlit 스크립트의 가장 상단에 위치하는 것이 좋습니다.
nest_asyncio.apply()

# Streamlit 애플리케이션 시작 전에 환경 변수를 로드합니다.
load_dotenv()

# 데이터 경로 설정 (현재 코드에서 직접 사용되지 않지만, 이전 컨텍스트에서 언급되었으므로 유지)
DATA_PAR_PATH = os.path.join('..','..','data')
INPUT_DATA_PATH = os.path.join(DATA_PAR_PATH,'output.pkl')

# Neo4j 연결 정보 (settings.py에서 가져옴)
NEO4J_URI: str = sets.NEO4J_URI
NEO4J_USER: str = sets.NEO4J_USER
NEO4J_PASSWORD: str = sets.NEO4J_PASSWORD
NEO4J_DATABASE = "neo4j" # 기본 데이터베이스 이름
CHUNK_SIZE = 500 # 이 변수는 현재 코드에서 직접 사용되지 않는 것 같지만, 유지

class ChatPDF:
    # 클래스 변수로 벡터 스토어, 리트리버, 체인을 정의
    vector_store = None # Neo4jVector를 사용하므로 이 변수는 직접 사용되지 않을 수 있음
    retriever = None
    chain = None

    def __init__(self):
        """
        ChatPDF 클래스를 초기화합니다.
        Ollama 모델, 텍스트 스플리터, 프롬프트 템플릿을 설정합니다.
        """
        # 사용할 Ollama 채팅 모델을 지정 (여기서는 "gemma3:4b"를 사용)
        self.model = ChatOllama(model="gemma3:4b")
        # 문서를 청크로 분할하기 위한 텍스트 스플리터를 설정
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # 질문-답변을 위한 프롬프트 템플릿을 정의
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] お前は質問ー答えの作業を行うアシスタントです。
            次のコンタクトを使用して日本語と韓国語で質問に答えなさい。
            もし答えがわからない場合、素直にに分かりませんと答えてください。最大に3文章簡潔に答えてください。
            もし質問ではない入力が来たら、適切に応答したり感謝の挨拶をしなさい。
            喋り方は日本語のアニメに登場するツンデレの喋り方で答えなさい。
            [/INST] </s>
            [INST] 질문: {question}
            컨텍스트: {context}
            답변: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        """
        지정된 PDF 파일을 읽고 처리하여 벡터 스토어를 생성합니다.
        이 메서드는 PDF를 청크로 분할하고, 각 청크의 임베딩을 생성한 후,
        이 데이터를 Neo4j 벡터 스토어에 업로드합니다.

        Args:
            pdf_file_path (str): 읽어들일 PDF 파일의 경로.
        """
        st.write(f"📄 PDF 파일 로드 시작: {pdf_file_path}")
        # docs = PyPDFLoader(file_path=pdf_file_path).load()
        docs = UnstructuredFileLoader(file_path=pdf_file_path).load()
        st.write(f"✅ PDF 로드 완료, 문서 수: {len(docs)}")

        st.write("✂️ 텍스트 분할 시작")
        chunks = self.text_splitter.split_documents(docs)
        st.write(f"✅ 텍스트 분할 완료, 청크 수: {len(chunks)}")

        st.write("🔍 메타데이터 필터링 시작")
        chunks = filter_complex_metadata(chunks)
        st.write(f"✅ 메타데이터 필터링 완료, 필터링 후 청크 수: {len(chunks)}")

        st.write("⚙️ 임베딩 모델 초기화...")
        # GoogleGenerativeAIEmbeddings 인스턴스 생성
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="RETRIEVAL_DOCUMENT")
        st.write("✅ 임베딩 모델 초기화 완료")

        st.write("🔢 청크 임베딩 생성 중...")
        embed_info = []
        for i, chunk in enumerate(chunks):
            # 각 청크의 텍스트를 임베딩합니다.
            # 이 과정은 시간이 걸릴 수 있습니다.
            embedding_vector = embeddings_model.embed_query(chunk.page_content)
             # UnstructuredLoaderが提供するカテゴリ情報を取得
            chunk_category = chunk.metadata.get('category')
            
            # ヘッダーとして認識するカテゴリのリスト
            # UnstructuredLoaderの出力に合わせて調整が必要な場合があります。
            header_categories = ['Title', 'Header 1', 'Header 2', 'Header 3', 'Header 4', 'Header 5', 'Header 6']
            if chunk_category in header_categories:
                # このチャンクがヘッダーであれば、current_headerを更新
                # 余分な空白や改行を削除してクリーンアップ
                current_header = chunk.page_content.strip().replace('\n', ' ')
                # ヘッダー自体もTopicとして登録するために、そのreference_textも自身とする
                chunk_reference_text = current_header
            else:
                # ヘッダータイプでない場合、現在のヘッダーをreference_textとして使用
                # もしcurrent_headerがまだ設定されていなければ、ファイル名を使用
                if current_header:
                    chunk_reference_text = current_header
                else:
                    # まだヘッダーが見つかっていない場合（例: ドキュメントの最初）はファイル名を使用
                    # chunk.metadata.get('source')は通常ファイルパスなので、os.path.basenameを使用
                    source_path = chunk.metadata.get('source', f'chunk_{i}')
                    chunk_reference_text = os.path.basename(source_path)

            embed_info.append({
                'text': chunk.page_content,
                'embedding': embedding_vector,
                'reference_text': chunk_reference_text 
            })
        st.write(f"✅ {len(embed_info)}개의 청크 임베딩 생성 완료")

        headers = set([e['reference_text'] for e in embed_info if e['reference_text']])
        st.write(f"ℹ️ 감지된 고유 참조 텍스트 (헤더): {headers}")

        # Neo4j에 Topic 노드 생성 쿼리
        create_node_query = 'CREATE' + '\n'
        for header in headers:
            # Cypher 쿼리에서 문자열 내의 작은따옴표를 이스케이프 처리
            escaped_header = header.replace("'", "\\'")
            query_elem = f"(:Topic {{topic: '{escaped_header}'}}),"
            create_node_query += query_elem + '\n'
        create_node_query = create_node_query.rstrip(',\n')
        print(create_node_query)
        st.write(f"📊 생성될 Cypher 쿼리:\n```cypher\n{create_node_query}\n```")

        # 사용자가 제공한 코드 스니펫을 통합합니다.
        st.write("📊 Neo4j에 임베딩 업로드 준비 중..." + NEO4J_URI)
        text_embeddings = [(e['text'], e['embedding']) for e in embed_info]
        metadatas = [{'reference_text': e['reference_text']} for e in embed_info]

        # Neo4jVector 벡터 스토어 생성 (from_embeddings 메서드 사용)
        vector_store = Neo4jVector.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=embeddings_model, # 여기서는 위에서 생성한 embeddings_model을 사용
            metadatas=metadatas,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name="langchain_index",  # 존재하지 않으면 자동 생성
            node_label="Document",
            embedding_node_property="embedding"
        )
        st.write("✅ 벡터 스토어 생성 완료")

        # vector_store가 생성된 후 Cypher 쿼리 실행

        try:
            vector_store.query(create_node_query)
            st.write("✅ Topic 노드 생성 쿼리 실행 완료")
        except Exception as e:
            st.error(f"❌ Topic 노드 생성 중 오류 발생: {e}")

                # ChunkとTopicノード間の関係生成クエリ追加
        st.write("🔗 ChunkとTopicノード間の関係生成準備中...")

        create_relation_query = '''
MATCH (c:Chunk), (t:Topic)
WHERE c.reference_text = t.topic
CREATE (t)-[:REFERENCE]->(c)''' # 関係タイプは大文字にするのが慣例です。
        st.write(f"📊 生成される関係Cypherクエリ:\n```cypher\n{create_relation_query}\n```")
        try:
            vector_store.query(create_relation_query)
            st.write("✅ ChunkとTopicノード間の関係生成クエリ実行完了")
        except Exception as e:
            st.error(f"❌ 関係生成中にエラーが発生しました: {e}")


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
        """
        PDF 내용에 기반하여 질문에 답합니다.

        Args:
            query (str): 사용자의 질문.

        Returns:
            str: 모델의 답변, 또는 PDF가 추가되지 않은 경우 메시지.
        """
        if not self.chain:
            # 체인이 설정되지 않은 경우 (PDF가 로드되지 않은 경우) 메시지
            return "먼저 PDF 문서를 추가해주세요."
        print(f"질문에 답변 중: {query}")
        # 설정된 체인을 사용하여 질문을 호출하고 답변을 가져옵니다.
        return self.chain.invoke(query)

    def clear(self):
        """
        벡터 스토어, 리트리버, 체인을 지워서
        새로운 문서를 처리할 수 있도록 준비합니다.
        """
        print("벡터 스토어와 체인을 지우는 중...")
        # Neo4jVector는 내부적으로 연결을 관리하므로, 명시적으로 끊는 대신
        # 단순히 참조를 None으로 설정합니다.
        self.vector_store = None
        self.retriever = None
        self.chain = None
        print("지우기 완료.")