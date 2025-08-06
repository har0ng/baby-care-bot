import streamlit as st
import os
import tempfile
from rag import ChatPDF
from serpapi import GoogleSearch
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

st.set_page_config(page_title="PDFチャットとウェブ検索")

# 세션 상태 초기화
if "chat_assistant" not in st.session_state:
    st.session_state["chat_assistant"] = ChatPDF()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_character" not in st.session_state:
    st.session_state.selected_character = "丁寧"

# 페이지 제목
st.title("PDFチャットとウェブ検索")

# 캐릭터 선택은 항상 위에서 노출되도록
character_options = ["丁寧", "ツンデレ", "猫ちゃん"]
selected_character = st.selectbox("🧑‍🎤 キャラクターを選んでください:", character_options, 
                                  index=character_options.index(st.session_state.selected_character))
st.session_state.selected_character = selected_character

# 탭 UI
tab1, tab2 = st.tabs(["PDFチャット", "ウェブ検索"])

# 🔹 PDF 챗봇 탭
with tab1:
    st.header("📄 PDFチャット")
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください。", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name

        st.session_state["chat_assistant"].ingest(file_path)
        os.remove(file_path)
        st.success("✅ PDFの処理が完了しました！質問してください！")

    # 채팅 인터페이스
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

# 🔹 웹 검색 탭
with tab2:
    st.header("🔎 近い小児科の検索")

    web_search_query = st.text_input("地名を入力してください。", key="web_search_input")
    search_button = st.button("検索")

    if search_button and web_search_query:
        st.write(f"「{web_search_query}」周辺の病院を検索中...")
        search_results = custom_google_search(web_search_query)

        if search_results:
            st.subheader("検索結果")
            for result in search_results:
                if 'title' in result and 'snippet' in result:
                    st.markdown(f"### [{result['title']}]({result['link']})")
                    st.write(result['snippet'])
                    if 'link' in result:
                        st.write(f"URL: {result['link']}")
                    st.markdown("---")
        else:
            st.warning("検索結果が見つかりませんでした。")

