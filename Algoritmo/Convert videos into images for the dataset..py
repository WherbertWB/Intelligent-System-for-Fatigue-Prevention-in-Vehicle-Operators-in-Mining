import cv2
import os
from PIL import Image


def video_to_images(video_path, output_folder, frame_rate=1, image_size=(640, 640)):
    """
    Extrai frames de um vídeo a cada segundo, redimensiona-os e salva a imagem original e uma cópia invertida horizontalmente.

    :param video_path: Caminho para o arquivo de vídeo.
    :param output_folder: Pasta onde as imagens serão salvas.
    :param frame_rate: Quantidade de frames por segundo (padrão: 1 frame por segundo).
    :param image_size: Tamanho para redimensionar as imagens (padrão: 640x640).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        print(f"Não foi possível obter o FPS do vídeo: {video_path}")
        return

    frame_count = 0
    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Captura um frame por segundo (ou conforme o frame_rate configurado)
        if frame_count % fps == 0:
            # Redimensiona a imagem
            frame_resized = cv2.resize(frame, image_size)
            # Converte para RGB (para usar com PIL)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Salva imagem original
            original_output_path = os.path.join(output_folder, f"{video_name}_frame_{image_count:04d}.jpg")
            image.save(original_output_path)

            # Cria e salva versão espelhada (flip horizontal)
            flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_output_path = os.path.join(output_folder, f"{video_name}_frame_{image_count:04d}_flip.jpg")
            flipped_image.save(flipped_output_path)

            image_count += 1

        frame_count += 1

    cap.release()
    print(f"Imagens salvas na pasta: {output_folder}")


def process_folder(input_folder, output_folder):
    """
    Processa todos os vídeos em uma pasta e extrai frames + duplicata espelhada.

    :param input_folder: Pasta contendo os vídeos.
    :param output_folder: Pasta onde as imagens serão salvas.
    """
    if not os.path.exists(input_folder):
        print(f"Pasta de entrada '{input_folder}' não encontrada.")
        return

    for file_name in os.listdir(input_folder):
        video_path = os.path.join(input_folder, file_name)
        if os.path.isfile(video_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processando vídeo: {file_name}")
            video_output_folder = os.path.join(output_folder, os.path.splitext(file_name)[0])
            video_to_images(video_path, video_output_folder)


if __name__ == "__main__":
    input_folder = input("Digite o caminho da pasta contendo os vídeos: ").strip()
    output_folder = input("Digite o caminho da pasta para salvar as imagens: ").strip()
    process_folder(input_folder, output_folder)
