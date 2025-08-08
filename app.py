import streamlit as st
import os
import tempfile
from rag import ChatPDF # rag.py에서 ChatPDF 클래스를 가져옵니다.
from serpapi import GoogleSearch # 웹 검색을 위한 SerpApi를 가져옵니다.
from dotenv import load_dotenv

# 환경 변수를 로드합니다.
# 環境変数をロードします。
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

st.set_page_config(page_title="PDFチャットとウェブ検索")

# 웹 검색 기능을 수행하는 함수를 정의합니다.
# ウェブ検索機能を実行する関数を定義します。
def custom_google_search(query: str):
    """
    SerpApi를 사용하여 웹 검색을 수행합니다.
    SerpApiを使用してウェブ検索を実行します。
    """
    try:
        params = {
            "q": query + "の小児科の住所と電話番号、営業時間、ホームページ",
            "api_key": SERPAPI_API_KEY,
            "hl": "ja",  # 일본어 검색
        }
        search_client = GoogleSearch(params)
        results = search_client.get_dict()
        return results.get("organic_results", [])
    except Exception as e:
        st.error(f"ウェブ検索中にエラーが発生しました：{e}")
        return None

# 세션 상태를 초기화합니다.
# セッション状態を初期化します。
if "chat_assistant" not in st.session_state:
    st.session_state["chat_assistant"] = ChatPDF()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_character" not in st.session_state:
    st.session_state.selected_character = "丁寧"


st.title("PDFチャットとウェブ検索")

character_options = ["丁寧", "ツンデレ", "猫ちゃん", "メイド"]
selected_character = st.selectbox("🧑‍🎤 キャラクターを選んでください:", character_options, 
                                  index=character_options.index(st.session_state.selected_character))
st.session_state.selected_character = selected_character


def get_character(char):
    if char == "ツンデレ":
        return "日本のアニメに登場する日本語のツンデレの口調で"
    if char == "猫ちゃん":
        return "日本語の語尾に最大に自然に「にゃん」を付けて可愛い口調で"
    if char == "メイド":
        return "日本のアニメに登場する日本語のメイドの口調で"
    return "一般的な喋り方で答えてください"


# 탭을 사용하여 기능을 나눕니다.
# タブを使用して機能を分けます。
tab1, tab2 = st.tabs(["PDFチャット", "ウェブ検索"])

with tab1:
    st.header("PDFチャット")
    
    # PDF 파일 업로드 부분
    # PDFファイルアップロード部分
    uploaded_files = st.file_uploader(
    "PDFファイルをアップロードしてください。", 
    type="pdf",  
    accept_multiple_files=True 
    )


    if uploaded_files:
        file_paths = []#　配列追加。
        # 임시 파일로 저장하고 처리합니다.
        # 一時ファイルとして保存し、処理します。
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tf.write(uploaded_file.getbuffer())
                file_paths.append(tf.name)
        # PDF를 Neo4j에 ingest합니다.
        # PDFをNeo4jにインジェストします。
        st.session_state["chat_assistant"].ingest(file_paths)


        for path in file_paths:
            os.remove(path)
        st.success("PDFの処理が完了しました！質問してください！")
        
    # 채팅 인터페이스
    # チャットインターフェース
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("PDFの内容について質問してください..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state["chat_assistant"].ask(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

from langchain.prompts import PromptTemplate
import google.generativeai as genai
import base64
# Gemini 모델 초기화 (API 키 필요)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

web_prompt_template = PromptTemplate.from_template(
    """
    <s> [INST] あなたは情報収集アシスタントです。
    以下のウェブ検索結果を参考にして病院別に整理してください。
    必ず含めないといけない要素は病院の住所、電話番号、名前、科、簡単な情報、出来ればホームページリンクや営業時間を含めてください。
    {character}、コンセプトに似合う絵文字を使ってhtml形式にインラインcssで喋り方のコンセプトに合わせて作って簡潔に3文以上で答えてください。
    もし小児科と関係ない結果が出てきたら地名の入力を頼んでください。
    [/INST] </s>
    ウェブ検索結果: {context}
    """
)

def build_web_context(results):
    context = ""
    for r in results:
        if 'title' in r and 'snippet' in r:
            context += f"{r['title']}\n{r['snippet']}\n{r.get('link', '')}\n\n"
    return context.strip()
def ask_gemini_about_web_results(results):
    context = build_web_context(results)
    character = st.session_state.get("selected_character", "丁寧")      
    prompt_text = web_prompt_template.format(context=context, character=get_character(character))
    response = gemini_model.generate_content(prompt_text.strip())
    return response.text
loading_placeholder = st.empty()
with tab2:
    st.header("近い小児科の検索")

    web_search_query = st.text_input("地名を入力してください。", key="web_search_input")
    search_button = st.button("検索")

    if search_button and web_search_query:
        st.session_state["loading"] = True
        st.write(f"ウェブで「{web_search_query}」の近くの病院を検索しています...")
        if st.session_state.get("loading"):
                char = "images/spinner.gif"
                if st.session_state["selected_character"] == "ツンデレ":
                    char = "images/Tsundere.gif"
                elif st.session_state["selected_character"] == "猫ちゃん":
                    char = "images/cat.gif"
                elif st.session_state["selected_character"] == "メイド":
                    char = "images/maid.gif"
                with open(char, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode("utf-8")

                loading_placeholder.markdown(
                    f"""
                    <div style="width: 100%; display: flex; justify-content: center;">
                        <img src="data:image/gif;base64,{b64}" width="300">
                    <div/>
                    """,
                    unsafe_allow_html=True
                )

        search_results = custom_google_search(web_search_query)
        if search_results:
            st.subheader("検索結果")
            st.html(ask_gemini_about_web_results(search_results))
            st.session_state["loading"] = False
            loading_placeholder.empty() 
            st.markdown("---")

        else:
            st.warning("検索結果がありません。地名の確認をお願いします。")
