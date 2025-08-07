# Python 3.11 슬림 버전 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 파일 전체 복사
COPY . .

# 의존성 설치 (requirements.txt 파일이 있어야 함)
RUN pip install --no-cache-dir -r requirements.txt

# 포트 노출 (Streamlit 기본 포트)
EXPOSE 8501

# Streamlit 앱 실행 (환경 변수 필요하면 --env-file 옵션으로)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
