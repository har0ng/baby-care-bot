# rag.py
# LangChainとNeo4jを使用してRAGシステムのコアロジックを実装します。
# LangChain과 Neo4j를 사용하여 RAG 시스템의 핵심 로직을 구현합니다.

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
import asyncio # asyncio 모듈을 임포트합니다.

# .env ファイルから環境変数をロードします。
# .envファイルから環境変数をロードします。
load_dotenv()

# 環境変数が設定されているか確認します。
# 環境変数が設定されているか確認します。
try:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY]):
        raise ValueError("必要な環境変数が設定されていません。")
except Exception as e:
    st.error(f"エラーが発生しました：{e}")
    st.stop()

# Google Generative AIを構成します。
# Google Generative AI를 구성합니다.
genai.configure(api_key=GOOGLE_API_KEY)

class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        """
        ChatPDFクラスを初期化し、必要な設定をします。
        ChatPDF 클래스를 초기화하고 필요한 설정을 수행합니다.
        """
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
        
        self.prompt_template = PromptTemplate.from_template(
            """
            <s> [INST] あなたは質問応答アシスタントです。
            以下のコンテキストを参考にして質問に答えてください。
            もし答えがわからない場合、単に「分かりません」と答えてください。
            日本語のアニメに登場するツンデレの口調で、最大3文で簡潔に答えてください。
            [/INST] </s>
            [INST] 質問: {question}
            コンテキスト: {context}
            回答: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        """
        PDFファイルを読み込み、テキストを分割してNeo4jにアップロードします。
        PDF 파일을 읽고, 텍스트를 분할하여 Neo4j에 업로드합니다.

        Args:
            pdf_file_path (str): アップロードするPDFファイルのパス。
        """
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
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
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self._gemini_invoke
            | StrOutputParser()
        )
    
    def _gemini_invoke(self, inputs: dict) -> str:
        """
        プロンプトとコンテキストを使用してGeminiモデルを呼び出します。
        프롬프트와 컨텍스트를 사용하여 Gemini 모델을 호출합니다.
        """
        prompt_str = inputs.to_string()
        
        response = self.model.generate_content(prompt_str.strip())
        return response.text
        
    def ask(self, query: str) -> str:
        """
        PDFの内容に基づいて質問に答えます。
        PDF의 내용에 기반하여 질문에 답합니다.
        """
        if not self.chain:
            return "흥、まずPDFファイルをアップロードしないと、何も答えられないじゃない！"
        
        return self.chain.invoke(query)

    def clear(self):
        """
        ベクトルストア、リトリーバー、チェーンを初期化します。
        벡터 스토어, 리트리버, 체인을 초기화합니다.
        """
        self.vector_store = None
        self.retriever = None
        self.chain = None

# # 必要なライブラリをインポートします。
# # 필요한 라이브러리를 가져옵니다.
# import streamlit as st
# from langchain_community.vectorstores import Neo4jVector
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.vectorstores.utils import filter_complex_metadata
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# from serpapi import GoogleSearch # SerpApi를 사용하여 웹 검색을 수행합니다.
# import json
# import asyncio # asyncio 모듈을 임포트합니다.

# # .env 파일에서 환경 변수를 로드합니다.
# # .envファイルから環境変数をロードします。
# load_dotenv()

# # 환경 변수가 설정되었는지 확인합니다.
# # 環境変数が設定されているか確認します。
# try:
#     NEO4J_URI = os.getenv("NEO4J_URI")
#     NEO4J_USER = os.getenv("NEO4J_USER")
#     NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     # SerpApi를 사용하려면 SerpApi 키가 필요합니다.
#     # SerpApiを使用するにはSerpApiのキーが必要です。
#     SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

#     if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY, SERPAPI_API_KEY]):
#         st.error("必要な環境変数が設定されていません。SerpApiのキーも必要です。")
#         st.stop()
# except Exception as e:
#     st.error(f"エラーが発生しました：{e}")
#     st.stop()

# # Google Generative AI를 구성합니다.
# # Google Generative AIを設定します。
# genai.configure(api_key=GOOGLE_API_KEY)

# # 웹 검색 기능을 수행하는 함수를 정의합니다.
# # ウェブ検索機能を実行する関数を定義します。
# def custom_google_search(query: str):
#     """
#     SerpApi를 사용하여 웹 검색을 수행합니다.
#     SerpApiを使用してウェブ検索を実行します。
#     """
#     try:
#         params = {
#             "q": query,
#             "api_key": SERPAPI_API_KEY,
#             "hl": "ja",  # 일본어 검색
#         }
#         search_client = GoogleSearch(params)
#         results = search_client.get_dict()
#         return results.get("organic_results", [])
#     except Exception as e:
#         st.error(f"ウェブ検索中にエラーが発生しました：{e}")
#         return None

# class ChatPDF:
#     vector_store = None
#     retriever = None
#     chain = None

#     def __init__(self):
#         """
#         ChatPDF 클래스를 초기화하고 필요한 설정을 수행합니다.
#         ChatPDFクラスを初期化し、必要な設定を行います。
#         """
#         # RuntimeError: There is no current event loop... 에러를 해결하기 위해 추가된 코드입니다.
#         # RuntimeError: There is no current event loop... エラーを解決するために追加されたコードです。
#         try:
#             loop = asyncio.get_running_loop()
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
            
#         self.embeddings_model = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001", 
#             task_type="RETRIEVAL_DOCUMENT",
#             api_key=GOOGLE_API_KEY
#         )
        
#         # 오타를 수정했습니다. RecursiveCharacterTextSplitter가 올바른 이름입니다.
#         # タイプミスを修正しました。RecursiveCharacterTextSplitterが正しい名前です。
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1024, 
#             chunk_overlap=100
#         )
        
#         self.model = genai.GenerativeModel("gemini-2.5-flash")
        
#         self.prompt_template = PromptTemplate.from_template(
#             """
#             <s> [INST] あなたは質問応答アシスタントです。
#             以下のコンテキストを参考にして質問に答えてください。
#             もし答えがわからない場合、単に「分かりません」と答えてください。
#             日本語のアニメに登場するツンデレの口調で、最大3文で簡潔に答えてください。
#             [/INST] </s>
#             [INST] 質問: {question}
#             コンテキスト: {context}
#             回答: [/INST]
#             """
#         )

#     def ingest(self, pdf_file_path: str):
#         """
#         PDF 파일을 읽고, 텍스트를 분할하여 Neo4j에 업로드합니다.
#         PDFファイルを読み込み、テキストを分割してNeo4jにアップロードします。

#         Args:
#             pdf_file_path (str): アップロードするPDFファイルのパス。
#         """
#         st.write(f"📄 PDFファイルの読み込みを開始します: {pdf_file_path}")
#         docs = PyPDFLoader(file_path=pdf_file_path).load()
#         st.write(f"✅ PDFファイルの読み込み完了、文書数: {len(docs)}")

#         st.write("✂️ テキスト分割を開始します。")
#         chunks = self.text_splitter.split_documents(docs)
#         st.write(f"✅ テキスト分割完了、チャンク数: {len(chunks)}")
        
#         st.write("🔍 メタ데이터 필터링을 시작합니다.")
#         chunks = filter_complex_metadata(chunks)
#         st.write(f"✅ 메타데이터 필터링 완료、チャン크 수: {len(chunks)}")

#         st.write("📊 Neo4jにテキストと埋め込みをアップロードします...")
#         self.vector_store = Neo4jVector.from_documents(
#             documents=chunks,
#             embedding=self.embeddings_model,
#             url=NEO4J_URI,
#             username=NEO4J_USER,
#             password=NEO4J_PASSWORD,
#             index_name="langchain_index",
#             node_label="Document",
#             embedding_node_property="embedding"
#         )
#         st.write("✅ Neo4jへのアップロード完了。")

#         self.retriever = self.vector_store.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={
#                 "k": 3,
#                 "score_threshold": 0.5,
#             },
#         )
#         st.write("✅ リトリーバー設定完了。")
        
#         # 원래의 올바른 체인 구성을 되돌렸습니다.
#         # 元の正しいチェーン構成に戻しました。
#         self.chain = (
#             {"context": self.retriever, "question": RunnablePassthrough()}
#             | self.prompt_template
#             | self._gemini_invoke
#             | StrOutputParser()
#         )
#         st.write("✅ チャットチェーン設定完了。")
    
#     def _gemini_invoke(self, inputs: dict) -> str:
#         """
#         프롬프트와 컨텍스트를 사용하여 Gemini 모델을 호출합니다.
#         プロンプトとコンテキストを使用してGeminiモデルを呼び出します。
        
#         TypeError를 해결하기 위해 이 메서드는 이제 StringPromptValue를 직접 받도록 수정되었습니다.
#         TypeErrorを解決するため、このメソッドはStringPromptValueを直接受け取るように修正されました。
#         """
#         # StringPromptValue 객체를 받으므로, to_string() 메서드를 사용해 문자열로 변환합니다.
#         # StringPromptValueオブジェクトを受け取るため、to_string()メソッドを使用して文字列に変換します。
#         prompt_str = inputs.to_string()
        
#         response = self.model.generate_content(prompt_str.strip())
#         return response.text
        
#     def ask(self, query: str) -> str:
#         """
#         PDF의 내용에 기반하여 질문에 답합니다.
#         PDFの内容に基づいて質問に答えます。
#         """
#         if not self.chain:
#             return "흥、먼저 PDF 문서를 업로드하세요。"
        
#         st.write(f"質問に答えています: {query}")
#         return self.chain.invoke(query)

#     def clear(self):
#         """
#         벡터 스토어, 리트리버, 체인을 초기화합니다.
#         ベクトルストア、リトリーバー、チェーンを初期化します。
#         """
#         self.vector_store = None
#         self.retriever = None
#         self.chain = None

# # Streamlit UI를 정의합니다.
# # Streamlit UIを定義します。
# if "chat_assistant" not in st.session_state:
#     st.session_state["chat_assistant"] = ChatPDF()

# st.title("Chat with PDF and Web Search")

# # 탭을 사용하여 기능을 나눕니다.
# # タブを使用して機能を分けます。
# tab1, tab2 = st.tabs(["PDFチャット", "ウェブ検索"])

# with tab1:
#     st.header("PDFチャット")
    
#     # PDF 파일 업로드 부분
#     # PDFファイルアップロード部分
#     uploaded_file = st.file_uploader("PDFファイルをアップロードしてください。", type="pdf")
#     if uploaded_file:
#         # 임시 파일로 저장하고 처리합니다.
#         # 一時ファイルとして保存し、処理します。
#         with open("temp_pdf.pdf", "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # PDF를 Neo4j에 ingest합니다.
#         # PDFをNeo4j에 인제스트합니다.
#         st.session_state["chat_assistant"].ingest("temp_pdf.pdf")
#         st.success("PDFの処理が完了しました！質問してください！")
        
#     # 채팅 인터페이스
#     # チャットインターフェース
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if prompt := st.chat_input("PDFの内容について質問してください..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         with st.chat_message("assistant"):
#             response = st.session_state["chat_assistant"].ask(prompt)
#             st.markdown(response)
#         st.session_state.messages.append({"role": "assistant", "content": response})

# with tab2:
#     st.header("ウェブ検索")
    
#     # 웹 검색 UI
#     # ウェブ検索UI
#     web_search_query = st.text_input("ウェブ検索のキーワードを入力してください。", key="web_search_input")
#     search_button = st.button("検索")
    
#     if search_button and web_search_query:
#         st.write(f"ウェブで「{web_search_query}」を検索しています...")
        
#         # SerpApi를 호출하여 검색 결과를 가져옵니다.
#         # SerpApiを呼び出して検索結果を取得します。
#         search_results = custom_google_search(web_search_query)
        
#         if search_results:
#             st.subheader("検索結果")
#             # 검색 결과를 순회하며 제목, 스니펫, URL을 표시합니다.
#             # 検索結果をループしてタイトル、スニペット、URLを表示します。
#             for result in search_results:
#                 if 'title' in result and 'snippet' in result:
#                     st.markdown(f"### [{result['title']}]({result['link']})")
#                     st.write(result['snippet'])
#                     if 'link' in result:
#                         st.write(f"URL: {result['link']}")
#                     st.markdown("---")
#         else:
#             st.warning("検索結果が見つかりませんでした。")
