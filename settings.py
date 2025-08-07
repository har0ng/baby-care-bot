import os

from dotenv import load_dotenv


env_file_path: str = os.path.join('.','.env')
load_dotenv(env_file_path)


NEO4J_URI: str = os.environ['NEO4J_URI']
NEO4J_USER: str = os.environ['NEO4J_USER']
NEO4J_PASSWORD: str = os.environ['NEO4J_PASSWORD']
