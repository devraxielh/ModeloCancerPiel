import os
def split_file(file_path, chunk_size_mb=99):
    chunk_size = chunk_size_mb * 1024 * 1024
    with open(file_path, 'rb') as f:
        i = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            with open(f"{file_path}.part{i:03d}", 'wb') as chunk_file:
                chunk_file.write(chunk)
            i += 1
    print(f"Archivo dividido en {i} partes.")
split_file("best_model.pt")
