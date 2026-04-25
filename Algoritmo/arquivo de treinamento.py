
from ultralytics import YOLO


def main():
    # Caminho do modelo pré-treinado (você pode alterar para yolov8m.pt, yolov8l.pt etc.)
    modelo = YOLO('yolo11s.pt') # ou caminho local do seu modelo personalizado

    # Treinamento do modelo com seus parâmetros
    modelo.train(
        data='datasetpuro/data.yaml',  # caminho para o arquivo de configuração do seu dataset
        epochs=100,
        imgsz=640,
        batch=16,
        device= 'cuda',  # 0 para usar GPU, 'cpu' se quiser usar CPU
        pretrained=True,
        workers=8,
        verbose=True
    )

if __name__ == '__main__':
    main()
