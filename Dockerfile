FROM python:3.12

WORKDIR /workspace
ADD requirements.txt app.py /workspace/
RUN pip install --no-cache-dir -r /workspace/requirements.txt
EXPOSE 7799
CMD ["python", "/workspace/app.py"]