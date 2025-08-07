# 必要なライブラリをインポートします。
# 필요한 라이브러리를 가져옵니다.
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

# .env 파일에서 환경 변수를 로드합니다.
# .envファイルから環境変数をロードします。
load_dotenv()

# 환경 변수가 설정되었는지 확인합니다.
# 環境変数が設定されているか確認します。
try:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # SerpApi를 사용하려면 SerpApi 키가 필요합니다.
    # SerpApiを使用するにはSerpApiのキーが必要です。
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GOOGLE_API_KEY, SERPAPI_API_KEY]):
        st.error("必要な環境変数が設定されていません。SerpApiのキーも必要です。")
        st.stop()
except Exception as e:
    st.error(f"エラーが発生しました：{e}")
    st.stop()

# Google Generative AI를 구성합니다.
# Google Generative AIを設定します。
genai.configure(api_key=GOOGLE_API_KEY)


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        """
        ChatPDF 클래스를 초기화하고 필요한 설정을 수행합니다.
        ChatPDFクラスを初期化し、必要な設定を行います。
        """
        # RuntimeError: There is no current event loop... 에러를 해결하기 위해 추가된 코드입니다.
        # RuntimeError: There is no current event loop... エラーを解決するために追加されたコードです。
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
        
        # 오타를 수정했습니다. RecursiveCharacterTextSplitter가 올바른 이름입니다.
        # タイプミスを修正しました。RecursiveCharacterTextSplitterが正しい名前です。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=100
        )
        
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
    def _get_prompt_for_character(self, character: str) -> PromptTemplate:
        if character == "丁寧":
            template = """
<s> [INST] あなたは丁寧で礼儀正しいAIです。
以下の文脈を参考にして、質問に対してできるだけ丁寧に、マックダウン形式で綺麗に最大20文以内で答えてください。
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
日本のアニメに出てくるツンデレ口調で、マックダウン形式で綺麗に最大20文で簡潔に答えてください。
[/INST] </s>
[INST] 質問: {question}
文脈: {context}
回答: [/INST]
"""
        elif character == "猫ちゃん":
            template = """
<s> [INST] あなたはかわいい猫ちゃん風のAIです。
わからなければ「よくわかんニャー」と答えてください。
文脈を参考にして、猫の語尾に「ニャー」をつけて、マックダウン形式で綺麗に最大20文以内でかわいく答えてください。
[/INST] </s>
[INST] 質問: {question}
文脈: {context}
回答: [/INST]
"""
        elif character == "猫ちゃん":
            template = """
<s> [INST] あなたはかわいい猫ちゃん風のAIです。
わからなければ「申し訳ございません。わたくしにはわかりかねます。ご主人様。」と答えてください。
文脈を参考にして、いつも「ご主人様」を最大に自然につけて日本のアニメに登場する日本語のメイドの口調で、マックダウン形式で綺麗に
最大20文以内でかわいく答えてください。
[/INST] </s>
[INST] 質問: {question}
文脈: {context}
回答: [/INST]
"""
        else:
            # 기본은 丁寧
            return self._get_prompt_for_character("丁寧")

        return PromptTemplate.from_template(template)


    def ingest(self, pdf_file_paths: list[str]):
        all_docs = []
        """
        PDF 파일을 읽고, 텍스트를 분할하여 Neo4j에 업로드합니다.
        PDFファイルを読み込み、テキストを分割してNeo4jにアップロードします。

        Args:
            pdf_file_path (str): アップロードするPDFファイルのパス。
        """
        for path in pdf_file_paths:
            docs = PyPDFLoader(file_path=path).load()
            all_docs.extend(docs)
            st.write(f"📄 PDFファイルの読み込みを開始します: {path}")
        
        st.write(f"✅ PDFファイルの読み込み完了、文書数: {len(docs)}")

        st.write("✂️ テキスト分割を開始します。")
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
        st.write("✅ Neo4jへのアップロード完了。")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        st.write("✅ リトリーバー設定完了。")
        character = st.session_state.get("selected_character", "丁寧")
        self.prompt_template = self._get_prompt_for_character(character)

        # 원래의 올바른 체인 구성을 되돌렸습니다.
        # 元の正しいチェーン構成に戻しました。
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self._gemini_invoke
            | StrOutputParser()
        )
        st.write("✅ チャットチェーン設定完了。")

    def _gemini_invoke(self, inputs: dict) -> str:
        """
        프롬프트와 컨텍스트를 사용하여 Gemini 모델을 호출합니다.
        プロンプトとコンテキストを使用してGeminiモデルを呼び出します。
        
        TypeError를 해결하기 위해 이 메서드는 이제 StringPromptValue를 직접 받도록 수정되었습니다.
        TypeErrorを解決するため、このメソッドはStringPromptValueを直接受け取るように修正されました。
        """
        # StringPromptValue 객체를 받으므로, to_string() 메서드를 사용해 문자열로 변환합니다.
        # StringPromptValueオブジェクトを受け取るため、to_string()メソッドを使用して文字列に変換します。
        prompt_str = inputs.to_string()
        
        response = self.model.generate_content(prompt_str.strip())
        return response.text
        
    def ask(self, query: str) -> str:
        """
        PDF의 내용에 기반하여 질문에 답합니다.
        PDFの内容に基づいて質問に答えます。
        """
        if not self.chain:
            return "흥、먼저 PDF 문서를 업로드하세요。"
        # 캐릭터 변경을 반영하기 위해 프롬프트 재설정
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
        """
        벡터 스토어, 리트리버, 체인을 초기화합니다.
        ベクトルストア、リトリーバー、チェーンを初期化します。
        """
        self.vector_store = None
        self.retriever = None
        self.chain = None

# Streamlit UI를 정의합니다.
# Streamlit UIを定義します。
if "chat_assistant" not in st.session_state:
    st.session_state["chat_assistant"] = ChatPDF()

