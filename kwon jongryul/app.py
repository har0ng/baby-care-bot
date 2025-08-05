# import os
# import tempfile
# import streamlit as st
# from streamlit_chat import message
# from rag import ChatPDF

# st.set_page_config(page_title="チャットPDF")

# def display_messages():
#     st.subheader("チャット")
#     for i, (msg, is_user) in enumerate(st.session_state["messages"]):
#         message(msg, is_user=is_user, key=str(i))
#     st.session_state["thinking_spinner"] = st.empty()

# def process_input():
#     if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
#         user_text = st.session_state["user_input"].strip()
#         with st.session_state["thinking_spinner"], st.spinner("考え中"):
#             agent_text = st.session_state["assistant"].ask(user_text)
#         st.session_state["messages"].append((user_text, True))
#         st.session_state["messages"].append((agent_text, False))

#         st.session_state["user_input"] = ""

# def read_and_save_file():
#     st.session_state["assistant"].clear()
#     st.session_state["messages"] = []
#     st.session_state["user_input"] = ""
#     for file in st.session_state["file_uploader"]:
#         with tempfile.NamedTemporaryFile(delete=False) as tf:
#             tf.write(file.getbuffer())
#             file_path = tf.name
#         with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
#             st.session_state["assistant"].ingest(file_path)
#         os.remove(file_path)

# def page():
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#     if "assistant" not in st.session_state:
#         st.session_state["assistant"] = ChatPDF()
#     st.header("チャットPDF")
#     st.subheader("文書アップロード")
#     st.file_uploader(
#         "Upload document",
#         type=["pdf"],
#         key="file_uploader",
#         on_change=read_and_save_file,
#         label_visibility="collapsed",
#         accept_multiple_files=True,
#     )
#     st.session_state["ingestion_spinner"] = st.empty()
#     display_messages()
#     st.text_input("Message", key="user_input", on_change=process_input)

# if __name__ == "__main__":
#     page()


# app.py
# Streamlit 앱의 UI와 사용자 상호작용을 처리합니다.
# StreamlitアプリのUIとユーザーインタラクションを処理します。

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
            "q": query + "の小児科の住所と電話番号",
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

st.title("PDFチャットとウェブ検索")

# 탭을 사용하여 기능을 나눕니다.
# タブを使用して機能を分けます。
tab1, tab2 = st.tabs(["PDFチャット", "ウェブ検索"])

with tab1:
    st.header("PDFチャット")
    
    # PDF 파일 업로드 부분
    # PDFファイルアップロード部分
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください。", type="pdf")
    if uploaded_file:
        # 임시 파일로 저장하고 처리합니다.
        # 一時ファイルとして保存し、処理します。
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name
        
        # PDF를 Neo4j에 ingest합니다.
        # PDFをNeo4jにインジェストします。
        st.session_state["chat_assistant"].ingest(file_path)
        os.remove(file_path)
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

with tab2:
    st.header("近い小児科の検索")
    
    # 웹 검색 UI
    # ウェブ検索UI
    web_search_query = st.text_input("地名を入力してください。", key="web_search_input")

    search_button = st.button("検索")
    
    if search_button and web_search_query:
        st.write(f"ウェブで「{web_search_query}」の近くの病院を検索しています...")
        
        # SerpApi를 호출하여 검색 결과를 가져옵니다.
        # SerpApiを呼び出して検索結果を取得します。
        search_results = custom_google_search(web_search_query)
        
        if search_results:
            st.subheader("検索結果")
            # 검색 결과를 순회하며 제목, 스니펫, URL을 표시합니다.
            # 検索結果をループしてタイトル、スニペット、URLを表示します。
            for result in search_results:
                if 'title' in result and 'snippet' in result:
                    st.markdown(f"### [{result['title']}]({result['link']})")
                    st.write(result['snippet'])
                    if 'link' in result:
                        st.write(f"URL: {result['link']}")
                    st.markdown("---")
        else:
            st.warning("検索結果が見つかりませんでした。")
