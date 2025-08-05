# Python 3.9을 기반으로 한 가벼운 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 로컬 파일을 컨테이너로 복사
COPY . .

# 라이브러리 설치 (파일이 있다면)
RUN pip install --no-cache-dir -r requirements.txt

# .env 파일이 필요한 경우, ENV로 환경변수 지정 가능
# 예: ENV OPENAI_API_KEY=your-api-key

# 컨테이너가 실행될 때 실행할 명령어
CMD ["python", "app.py"]
