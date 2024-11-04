FROM python:3.10.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./webapp /code/app


CMD ["fastapi", "run", "app/main.py", "--port", "8080"]