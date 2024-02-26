# установим образ
# Здесь можно использовать разные образы, для более простого проекта на python
FROM python:3.10.2-slim-buster

# скопируем файл с необходимыми библиотекми, которые хотим установить
COPY ./requirements.txt /root/requirements.txt

# обновим pip и установим библиотеки из requirements, --ignore-installed - переустановка пакетов, если они уже есть
RUN pip install --upgrade pip
RUN pip install -r /root/requirements.txt

# создание рабочей директории
WORKDIR /root/docker_test

# копирование всех файлов, которые не указаны в dockerignore в новую директорию
COPY . /root/docker_test

# запуск скрипта, для нового проекта поставить и запустить train.py
CMD ["python", "predict.py"]
