import cv2
import imagehash
from PIL import Image
import numpy as np
import os
import pytesseract
import re
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil

def is_a_vertical_frame(video_frame):
    width, height = video_frame.size
    return width < height

def get_last_vertical_frame(video_filename):
    video_capture = cv2.VideoCapture(video_filename)
    last_frame_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    last_vertical_frame = None
    success, image = video_capture.read()
    frame_count = 0
    while success:
        if frame_count == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            first_frame_image = Image.fromarray(image)
            if not is_a_vertical_frame(first_frame_image):
                return None  # Not a vertical video, so we skip it

        if frame_count == last_frame_number - 12:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            last_vertical_frame = Image.fromarray(image)

        success, image = video_capture.read()
        frame_count += 1

    video_capture.release()
    return last_vertical_frame

def change_nonwhite_pixels_to_black(image):
    white_inferior_limit = (150, 150, 150)
    replacement_color = (0, 0, 0)
    image_data = np.array(image)
    image_data[(image_data < white_inferior_limit).all(axis=-1)] = replacement_color
    return Image.fromarray(image_data, mode="RGB")

def get_p_hash(image):
    return imagehash.phash(image)

def extract_tiktok_username(image, username_file="tiktok_usernames.txt"):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    _, thresh = cv2.threshold(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)
    username_pattern = r'@[\w.]+'
    usernames = re.findall(username_pattern, text)

    # Verifica se encontrou algum username
    if usernames:
        # Escreve o(s) username(s) encontrado(s) no arquivo
        with open(username_file, 'a') as f:
            for username in usernames:
                f.write(f"{username}\n")  # Escreve cada username em uma nova linha
        return usernames[0]  # Retorna o primeiro username encontrado
    return None

def calculate_probability(distance, max_distance=10):
    return max(0, (max_distance - distance) / max_distance)

def is_tiktok(last_frame, TIKTOK_LAST_FRAME_P_HASH, username_file="tiktok_usernames.txt"):
    last_frame_p_hash = get_p_hash(last_frame)
    distance = last_frame_p_hash - TIKTOK_LAST_FRAME_P_HASH
    probability = calculate_probability(distance)

    if distance <= 10:
        username = extract_tiktok_username(last_frame, username_file)
        return True, username, probability
    else:
        return False, None, probability

def is_kwai(last_frame, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH):
    last_frame_p_hash = get_p_hash(last_frame)
    distance_older = last_frame_p_hash - OLDER_KWAI_LAST_FRAME_P_HASH
    probability_older = calculate_probability(distance_older)

    if distance_older <= 10:
        return True, probability_older
    else:
        treated_last_frame = change_nonwhite_pixels_to_black(last_frame)
        treated_last_frame_p_hash = get_p_hash(treated_last_frame)
        distance_newer = treated_last_frame_p_hash - NEWER_KWAI_LAST_FRAME_P_HASH
        probability_newer = calculate_probability(distance_newer)

        if distance_newer <= 10:
            return True, probability_newer
        else:
            return False, max(probability_older, probability_newer)

def process_video(video_path, TIKTOK_LAST_FRAME_P_HASH, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH, username_file="tiktok_usernames.txt"):
    try:
        last_frame = get_last_vertical_frame(video_path)
        if last_frame is None:
            return None  # Skip non-vertical videos

        video_is_tiktok, username, tiktok_probability = is_tiktok(last_frame, TIKTOK_LAST_FRAME_P_HASH, username_file)
        if video_is_tiktok:
            return {
                "file": video_path,
                "platform": "TikTok",
                "username": username,
                "probability": tiktok_probability
            }

        video_is_kwai, kwai_probability = is_kwai(last_frame, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH)
        if video_is_kwai:
            return {
                "file": video_path,
                "platform": "Kwai",
                "username": None,
                "probability": kwai_probability
            }

        return {
            "file": video_path,
            "platform": "Unknown",
            "username": None,
            "probability": {
                "TikTok": tiktok_probability,
                "Kwai": kwai_probability
            }
        }
    except Exception as e:
        print(f"Erro ao processar {video_path}: {str(e)}")
        return None

def process_batch(batch, TIKTOK_LAST_FRAME_P_HASH, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH, username_file="tiktok_usernames.txt"):
    results = []
    for video_path in batch:
        result = process_video(video_path, TIKTOK_LAST_FRAME_P_HASH, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH, username_file)
        if result:
            results.append(result)
    return results

def main(input_directory):
    # Inicialização
    tiktok_last_frame = get_last_vertical_frame("./classified_videos/tiktok_example_2.mp4")
    TIKTOK_LAST_FRAME_P_HASH = get_p_hash(tiktok_last_frame)

    older_kwai = get_last_vertical_frame("./classified_videos/kwai_example_4.mp4")
    newer_kwai = get_last_vertical_frame("./classified_videos/kwai_example_1.mp4")
    treated_newer_kwai = change_nonwhite_pixels_to_black(newer_kwai)

    OLDER_KWAI_LAST_FRAME_P_HASH = get_p_hash(older_kwai)
    NEWER_KWAI_LAST_FRAME_P_HASH = get_p_hash(treated_newer_kwai)

    # Encontrar todos os arquivos .mp4
    all_videos = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.mp4'):
                all_videos.append(os.path.join(root, file))

    # Calcular o tamanho do lote com base na memória disponível
    available_memory = psutil.virtual_memory().available
    estimated_memory_per_video = 100 * 1024 * 1024  # Estimativa: 100 MB por vídeo
    batch_size = max(1, int(available_memory / (estimated_memory_per_video * 2)))  # Usa metade da memória disponível

    # Número de workers baseado nos núcleos de CPU disponíveis
    num_workers = multiprocessing.cpu_count()

    results = []

    # Processamento em lotes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, len(all_videos), batch_size):
            batch = all_videos[i:i+batch_size]
            future = executor.submit(process_batch, batch, TIKTOK_LAST_FRAME_P_HASH, OLDER_KWAI_LAST_FRAME_P_HASH, NEWER_KWAI_LAST_FRAME_P_HASH, "tiktok_usernames.txt")
            results.extend(future.result())

            # Salvar resultados intermediários
            with open('video_classification_results_partial.json', 'w') as f:
                json.dump(results, f, indent=2)

    # Salvar resultados finais
    with open('video_classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Classificação concluída. Resultados salvos em 'video_classification_results.json'.")
    print("Usernames extraídos foram salvos em 'tiktok_usernames.txt'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifica vídeos TikTok e Kwai em um diretório com probabilidades.")
    parser.add_argument("input_directory", help="Caminho para o diretório contendo os vídeos .mp4")
    args = parser.parse_args()

    main(args.input_directory)
