FROM python:3.11-slim

WORKDIR /app

COPY . .

# Install tesseract-cr and poppler-utils
RUN apt-get update && apt-get install poppler-utils tesseract-ocr libtesseract-dev libleptonica-dev -y
# Need to install these libraries to support cv2 issues
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Install python packages
RUN pip install -r requirements.txt

EXPOSE 5001
CMD ["python", "-m", "flask", "--app", "src.app", "run", "--host=0.0.0.0", "--port=5001"]