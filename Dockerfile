FROM python:3.12

WORKDIR /workspace
ADD requirements.txt app.py logger.py .env /workspace/
# RUN pip install --no-cache-dir -r /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
EXPOSE 7799
CMD ["python", "/workspace/app.py"]