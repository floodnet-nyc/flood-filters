import os

BASE_URL = 'https://docs.google.com/uc?export=download&confirm=t&id={}'
def ensure_checkpoint(file_id, path):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        import urllib.request
        print("No checkpoint found. Downloading...")
        progress = lambda i, s, t: print(f'downloading checkpoint to {path}: {i * s / t:.2%}', end="\r")
        urllib.request.urlretrieve(BASE_URL.format(file_id), path, progress)
    return path
