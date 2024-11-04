from fastapi import FastAPI
import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"Root": "Dir"}

@app.get("/secret")
def read_secret():
    secret_string = os.environ.get("API_KEY", "Not Found")
    return {"API_KEY": secret_string}

@app.get("/configMap")
def read_config():
    config_string = os.environ.get("APP_ENV", "Not Found")
    return {"APP_ENV": config_string}