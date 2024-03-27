FROM python:3.8
COPY . /app
WORKDIR /app
COPY ./data /data
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
