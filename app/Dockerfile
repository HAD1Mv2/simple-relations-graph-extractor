FROM python:3.11

RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install transformers==5.17.0
RUN pip install spacy==3.8.16
RUN python -m spacy download en_core_web_sm
RUN pip install git+https://github.com/HAD1Mv2/fastcoref_torch2.9.git
RUN pip install "fastapi[standard]==0.142.1"
RUN pip install nltk==3.10.3
RUN python -m nltk.downloader 'punkt_tab'
RUN pip install numpy==2.4.6

COPY app.py app/
COPY utils app/utils/
COPY relation_extract app/relation_extract/

CMD [ "python","-m", "app.app" ]