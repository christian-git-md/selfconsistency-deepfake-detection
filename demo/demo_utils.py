import requests
import numpy as np


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def format_imshow(distances):
    if np.mean(distances > 128) > .5:
        distances = 255 - distances
    baseline = 90
    distances = distances.astype(np.int16)
    distances = np.maximum(0, distances - baseline) * (1 / (1 - baseline / 256))
    distances = distances.astype(np.uint8)
    return distances