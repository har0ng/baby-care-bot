# ChatPDF
## プロジェクト概要  
ChatPDFはPDF文書をアップロードして内容を解析し、自然言語の質問にPDFの内容をもとに回答するウェブアプリケーションです。  
Streamlitを使い、簡単かつ迅速にインタラクティブなチャットボットUIを提供します。

## 主な機能  
- PDFファイルアップロード対応（.pdf形式）  
- アップロードされたPDFのテキストや画像を分割・処理  
- 文書内容をベクトル化しNeo4jのベクトルストアに保存  
- 自然言語の質問に対してPDFの内容を活用し回答  
- 高度な非同期処理とセッション状態の維持  

## インストールと実行方法

### 1. 必要環境  
- Python 3.11  以上
- Docker（任意）  
- Neo4jデータベース
- gemini API key
- neo4j

### 2. リポジトリのクローンと依存パッケージのインストール  
```bash
git clone [リポジトリURL]
cd [プロジェクトフォルダ]
pip install -r requirements.txt

###　3.環境変数の設定
-プロジェクトルートに.envファイルを作成し、以下の内容を入力してください。
GOOGLE_API_KEY=your_google_api_key_here
NEO4J_URI=bolt://your_neo4j_host:????
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

### 4. アプリケーションの実行
```bash
streamlit run app.py

-Dockerコンテナで実行する場合は以下を実行してください。

```bash
docker build -t chatpdf-app .
docker run -p 8501:8501 --env-file .env chatpdf-app

## 使い方
Web UIからPDF文書をアップロードします。

PDFのベクトル化とNeo4jへの保存処理が行われます。

テキスト入力欄に質問を入力すると、gemeniベースのチャットボットがPDF内容に沿った回答をします。

必要に応じて「Clear」ボタンなどで状態をリセットできます。

## 主要コード説明
app.py（Streamlitウェブアプリ）
UIの構築とセッション状態の管理

ファイルアップロード時にChatPDFクラスのingest()を呼び出し

ユーザーの質問を処理し回答を表示

chatpdf.py（コアロジック）
PDFの読み込みとテキスト分割（PyPDFLoader、RecursiveCharacterTextSplitter使用）

埋め込み生成（GoogleGenerativeAIEmbeddings使用）

Neo4jベクトルストアへの埋め込み保存・検索機能の実装

Ollamaチャットボットモデルで質問・回答処理

settings.py（環境変数の読み込み）
.envファイルからNeo4j接続情報をロード
