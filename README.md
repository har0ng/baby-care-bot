＃成功した時の画面（neo4j db,gemini api, serp api 必要）<br>
-プロジェクトの管理者がserverを開いた時だけ見るのが可能です。<br>
https://bug-free-carnival-wrvq6pv6q6rc54p7-8501.app.github.dev/
## 概要

このプロジェクトに含まれる主な機能は以下の通りです：

- PDFファイルをアップロードして、内容に基づいた質問応答
- 地名を入力して、周辺の小児科情報をSerpApi + Geminiで検索・要約
- キャラクター別の口調（丁寧、ツンデレ、猫ちゃん、メイド）による自然な回答
- LangChainとNeo4jによるベクトル検索ベースのドキュメントQAシステム
- Streamlitによる直感的なWebインターフェース

## プロジェクト構成

├── .gitignore　<br>
├── Dockerfile.txt           # Docker 環境設定ファイル <br>
├── README.md                # プロジェクト説明書 <br>
├── app.py                   # Streamlit ウェブUI <br>
├── chunk.py                 # PDFのテキスト・画像抽出処理 <br>
├── rag.py                   # PDF QA・検索処理（RAG） <br>
├── requirements.txt         # Python パッケージ一覧 <br>
└── setting.py               # 環境変数の読み込み（.env）<br>

- セットアップ方法（Installation）<br>
-Python 3.11 以上をインストール 

### 2. 仮想環境の作成（任意）
-python -m venv venv <br>
-source venv/bin/activate # macOS/Linux <br>
-venv\Scripts\activate # Windows <br>



-```bash 
-python -m venv venv <br>
-source venv/bin/activate  # macOS/Linux　<br>
-venv\Scripts\activate     # Windows <br>
-依存ライブラリのインストール  


### 3. 依存パッケージのインストール
-pip install -r requirements.txt

### 4. .env ファイルの作成
-ルートディレクトリに `.env` ファイルを作成し、以下の内容を記載してください。 (重要） <br>
-NEO4J_URI=bolt://localhost:???? <br>
-NEO4J_USER=neo4j <br>
-NEO4J_PASSWORD=your_password <br>

-GOOGLE_API_KEY=your_google_genai_key <br>
-SERPAPI_API_KEY=your_serpapi_key <br>

## 実行方法
-```bash <br>
-streamlit run app.py <br>
 

## Docker での実行（オプション）
-```bash <br>
-docker build -t pdf-chatbot -f Dockerfile.txt . <br>
-docker run -p 8501:8501 --env-file .env pdf-chatbot <br>


---

## 主な機能

### PDFチャット

- アップロードされたPDFをNeo4jに格納し、Geminiで質問に回答します
- 回答はMarkdown形式で整形され、最大20文以内に要約されます

### ウェブ検索と要約

- 入力された地名に基づいてSerpApiで検索を行い、周辺の小児科を収集します
- Geminiが検索結果をHTML形式で要約し、キャラクターに応じたスタイルで返答します

---

## キャラクターの種類

-| キャラクター名 | 説明 |
-|----------------|------|
-| 丁寧           | 礼儀正しく丁寧な話し方 |
-| ツンデレ       | ツンデレ風のぶっきらぼうな口調 |
-| 猫ちゃん       | 「にゃん」を語尾につけた猫風の話し方 |
-| メイド         | 「ご主人様」と呼ぶメイド風の話し方 |

---

## 使用技術

- LangChain
- Google Generative AI (Gemini)
- Neo4j Vector Database
- Streamlit
- SerpApi
- Unstructured（PDF解析）
- Docker

---

## 注意事項

- `.env`ファイルが存在しないと実行できません(重要)
- Google Generative AI と SerpApi のAPIキーを取得しておく必要があります
- Neo4jは別途ローカルもしくはクラウド環境で動作している必要があります

---
Neo4jは別途ローカルまたはクラウド上で稼働している必要があります
