import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO
import pygame
import mediapipe as mp
from scipy.spatial import distance as dist
import logging

import uuid
from datetime import datetime
from pathlib import Path
from skfuzzy import interp_membership
from matplotlib.patches import Wedge, Circle  # <- para o Fadigômetro

# ----------------------------
# Logging / estilo
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
plt.style.use('ggplot')

# ------------------
# Áudio
# ------------------
pygame.mixer.init()

def play_alert(sound_name):
    """Toca alerta sem bloquear o loop de vídeo."""
    try:
        sound_path = f'Mensagens de Voz/{sound_name}'
        if os.path.exists(sound_path):
            pygame.mixer.Sound(sound_path).play()
        else:
            logger.warning(f"Arquivo de áudio não encontrado: {sound_path}")
    except Exception as e:
        logger.error(f"Erro ao reproduzir áudio: {str(e)}")

# ---------------------------
# I/O
# ---------------------------
output_dir = "_DataFrame"
Path(output_dir).mkdir(exist_ok=True)
output_dir_path = Path(output_dir)

run_suffix = uuid.uuid4().hex[:8]
excel_path = output_dir_path / f"ResultadosFuzzy_{run_suffix}.xlsx"
csv_path   = output_dir_path / f"ResultadosFuzzy_{run_suffix}.csv"

janela_rows = []
janela_idx = 0

# ---------------------------
# Vídeo / Telemetria
# ---------------------------
video_path = 0#"ImagensTestes/amostra01/video.mp4"
if not os.path.exists(str(video_path)) and not isinstance(video_path, int):
    raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

telemetria_path = 'ImagensTestes/amostra01/telemetria.csv'
try:
    telemetria_df = pd.read_csv(telemetria_path, sep=';', encoding='utf-8')
    telemetria_df['Data'] = pd.to_datetime(telemetria_df['Data'], format='%d/%m/%Y %H:%M')
    telemetria_df['Velocidade'] = telemetria_df['Velocidade'].astype(str).str.replace(',', '.').astype(float)
except Exception as e:
    logger.error(f"Erro ao carregar telemetria: {e}")
    telemetria_df = pd.DataFrame()

# -------------
# YOLO
# -------------
try:
    model_path = 'G:/Meu Drive/_Mestrado/Projeto Mestrado/treinamentodeteccaofadiga/rec_facial_yolo/dataset_20_06_25 - ver1/modelo treinado - yolo11s/detect/train/weights/last.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo YOLO não encontrado: {model_path}")
    model = YOLO(model_path)
    USE_HALF = False
    try:
        model.to("cuda")
        USE_HALF = True
        logger.info("YOLO na GPU (half).")
    except Exception as e:
        logger.warning(f"Rodando em CPU/FP32. Detalhe: {e}")
except Exception as e:
    logger.error(f"Erro ao carregar YOLO: {e}")
    raise

# ------------------------
# Captura
# ------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir o vídeo.")

# -------------------------
# FaceMesh
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ------------------------
# Índices / thresholds
# ------------------------
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES  = [362, 387, 385, 263, 380, 373]
MOUTH_INDICES     = [13, 14]
NOSE_TIP_INDEX    = 1
CHIN_INDEX        = 199
FOREHEAD_INDEX    = 10

EAR_THRESHOLD      = 0.15
EAR_SEMI_THRESHOLD = 0.20
MAR_THRESHOLD      = 0.7
MIN_YAWN_DURATION  = 2
YAWN_COOLDOWN      = 2

HEAD_TILT_LOW      = 2
HEAD_TILT_MODERATE = 3
HEAD_TILT_HIGH     = 4

# Limiar parametrizável para microssono (>= fecha olhos)
CRITICAL_EYES_CLOSED_SEC = 2.0  # ajuste para 3.0 se desejar

# --------------------
# Telemetria fixa
# --------------------
TEMPO_CONDUCAO_FIXO = 0
OSCILACAO_VELOCIDADE_FIXO = 0

def calculate_telemetria_fixed_values():
    """Calcula os valores fixos de telemetria uma vez no início."""
    global TEMPO_CONDUCAO_FIXO, OSCILACAO_VELOCIDADE_FIXO
    if telemetria_df.empty:
        TEMPO_CONDUCAO_FIXO = 0
        OSCILACAO_VELOCIDADE_FIXO = 0
        return
    try:
        contato_on_events = telemetria_df[telemetria_df['Evento'] == 'Contato ON']
        if contato_on_events.empty:
            TEMPO_CONDUCAO_FIXO = 0
            OSCILACAO_VELOCIDADE_FIXO = 0
            return
        contato_on_time = contato_on_events.iloc[-1]['Data']
        dados = telemetria_df[telemetria_df['Data'] >= contato_on_time]
        if dados.empty:
            TEMPO_CONDUCAO_FIXO = 0
            OSCILACAO_VELOCIDADE_FIXO = 0
            return
        ultimo_t = dados['Data'].iloc[-1]
        TEMPO_CONDUCAO_FIXO = min((ultimo_t - contato_on_time).total_seconds()/60, 180)
        velocidades = dados['Velocidade'].dropna()
        if len(velocidades) > 1:
            OSCILACAO_VELOCIDADE_FIXO = min(np.max(np.abs(np.diff(velocidades))), 50)
        else:
            OSCILACAO_VELOCIDADE_FIXO = 0
        logger.info(f"Telemetria fixa: tempo={TEMPO_CONDUCAO_FIXO:.1f}min, oscilacao={OSCILACAO_VELOCIDADE_FIXO:.1f}km/h")
    except Exception as e:
        logger.error(f"Erro telemetria: {e}")
        TEMPO_CONDUCAO_FIXO = 0
        OSCILACAO_VELOCIDADE_FIXO = 0

def get_telemetria_data(_elapsed):
    return TEMPO_CONDUCAO_FIXO, OSCILACAO_VELOCIDADE_FIXO

# --------------------
# Estado
# --------------------
class DriverState:
    def __init__(self):
        # contadores/tempos por janela
        self.blink_count = 0
        self.yawn_count = 0
        self.eyes_closed_total = 0.0
        self.max_eyes_closed_streak = 0.0
        self.eyes_semiclosed_total = 0.0
        self.eyes_closed_start = 0.0
        self.eyes_semiclosed_start = 0.0

        # ângulo vertical
        self.head_tilt_angle_sum = 0.0
        self.head_tilt_angle_count = 0
        self.head_tilt_category = "Normal"
        self.vertical_angle_offset = None

        # bocejo
        self.is_yawning = False
        self.yawn_start_time = 0.0
        self.last_yawn_time = 0.0

        # bbox
        self.last_bbox = None
        self.missed_frames = 0

        # UI / fuzzy
        self.current_status = "Leve"  # sem "Normal" na UI
        self.fuzzy_level = 0.0

        # Microssono / janelas
        self.microssono_in_window = False            # ocorreu ≥ limiar na janela atual
        self.defer_grave_to_next = False             # próxima janela deve considerar override
        self.show_note_grave_from_microssono = False # mostrar texto na tela nesta janela
        self._note_timer_until = 0.0                 # tempo para expirar nota na tela

# ------------------------------------
# Helpers YOLO / landmarks
# ------------------------------------
def iou(boxA, boxB):
    """Corrigido: usa yB corretamente e calcula interseção (w*h)."""
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def ema_bbox(prev, cur, alpha=0.7):
    if prev is None:
        return cur
    return (
        int(alpha*prev[0] + (1-alpha)*cur[0]),
        int(alpha*prev[1] + (1-alpha)*cur[1]),
        int(alpha*prev[2] + (1-alpha)*cur[2]),
        int(alpha*prev[3] + (1-alpha)*cur[3]),
    )

landmarks_history = []
def smooth_landmarks(current_landmarks, window_size=3):
    """Média móvel simples sobre os landmarks; reinicia se mudar nº de pontos."""
    global landmarks_history
    if not current_landmarks:
        return []
    if landmarks_history and len(current_landmarks) != len(landmarks_history[-1]):
        landmarks_history = []
    landmarks_history.append(current_landmarks)
    if len(landmarks_history) > window_size:
        landmarks_history.pop(0)
    smoothed = []
    for i in range(len(current_landmarks)):
        xs = [lm[i][0] for lm in landmarks_history if i < len(lm)]
        ys = [lm[i][1] for lm in landmarks_history if i < len(lm)]
        smoothed.append((float(np.mean(xs)), float(np.mean(ys))) if xs and ys else current_landmarks[i])
    return smoothed

# ----------------------------
# Cálculos geométricos
# ----------------------------
def calculate_ear(landmarks, eye_indices):
    try:
        p2, p6 = landmarks[eye_indices[1]], landmarks[eye_indices[5]]
        p3, p5 = landmarks[eye_indices[2]], landmarks[eye_indices[4]]
        p1, p4 = landmarks[eye_indices[0]], landmarks[eye_indices[3]]
        v1 = dist.euclidean(p2, p6)
        v2 = dist.euclidean(p3, p5)
        h  = dist.euclidean(p1, p4)
        return (v1 + v2) / (2.0 * h)
    except (IndexError, TypeError):
        return 0.3

def calculate_mar(landmarks):
    try:
        return dist.euclidean(landmarks[MOUTH_INDICES[0]], landmarks[MOUTH_INDICES[1]])
    except (IndexError, TypeError):
        return 0.0

def calculate_head_tilt(landmarks, ds: DriverState):
    try:
        nose_tip = landmarks[NOSE_TIP_INDEX]
        chin     = landmarks[CHIN_INDEX]
        forehead = landmarks[FOREHEAD_INDEX]
        vertical_length = dist.euclidean(forehead, chin)
        if vertical_length == 0: return 0, 0, "Normal"
        delta_y = chin[1] - nose_tip[1]
        vertical_angle = np.degrees(np.arctan2(delta_y, vertical_length))
        if ds.vertical_angle_offset is None:
            ds.vertical_angle_offset = vertical_angle
        vertical_angle -= ds.vertical_angle_offset
        abs_vertical = abs(vertical_angle)
        if abs_vertical < HEAD_TILT_LOW:
            tilt_category = "Normal"
        elif abs_vertical < HEAD_TILT_MODERATE:
            tilt_category = "Baixa Vertical"
        elif abs_vertical < HEAD_TILT_HIGH:
            tilt_category = "Moderada Vertical"
        else:
            tilt_category = "Alta Vertical"
        return 0, vertical_angle, tilt_category
    except (IndexError, TypeError):
        return 0, 0, "Normal"

# -------------------------
# YOLO face detector (GPU)
# -------------------------
def detect_face_with_yolo(frame, model, conf_threshold=0.35):
    face_rois = []
    try:
        results = model.predict(frame, conf=conf_threshold, iou=0.5, imgsz=320, half=USE_HALF, verbose=False)
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls[0])].lower() == 'face':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        face_rois.append((frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)))
    except Exception as e:
        logger.error(f"Erro YOLO: {e}")
    return face_rois

# ----------------------------
# UI helpers
# ----------------------------
def draw_transparent_rect(frame, tl, br, color, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, tl, br, color, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def draw_head_tilt_indicator(frame, vertical_angle, tilt_category):
    h, w = frame.shape[:2]
    cx, cy = w - 100, 100
    r = 50
    cv2.circle(frame, (cx, cy), r, (200, 200, 200), 2)
    max_angle = 30
    ang = np.clip(vertical_angle, -max_angle, max_angle)
    offy = int((-ang / max_angle) * r)
    if "Alta" in tilt_category:
        color = (0, 0, 255)
    elif "Moderada" in tilt_category:
        color = (0, 100, 255)
    elif "Baixa" in tilt_category:
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)
    cv2.circle(frame, (cx, cy + offy), 6, color, -1)
    cv2.putText(frame, f"{vertical_angle:.1f} graus", (cx - 40, cy + r + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

def draw_header_info(frame, status_text, nivel_fadiga, elapsed_seconds, note_text: str = None):
    w = frame.shape[1]
    frame = draw_transparent_rect(frame, (0, 0), (w, 40), (0, 0, 0), alpha=0.6)
    status_colors = {'Leve': (0, 255, 0), 'Moderada': (0, 165, 255), 'Grave': (0, 0, 255)}
    color = status_colors.get(status_text, (0, 255, 0))
    cv2.putText(frame, f"ESTADO ATUAL: {status_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"NIVEL: {nivel_fadiga:.1f}", (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    mmss = f"{elapsed_seconds // 60:02}:{elapsed_seconds % 60:02}"
    cv2.putText(frame, f"TEMPO: {mmss}", (w - 160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    if note_text:
        frame = draw_transparent_rect(frame, (20, 50), (w-20, 85), (0, 0, 0), alpha=0.5)
        cv2.putText(frame, note_text, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def draw_metrics_panel(frame, ds: DriverState):
    w, h = 230, 160
    margin = 16
    H, W = frame.shape[:2]
    x, y = margin, H - h - margin
    frame = draw_transparent_rect(frame, (x, y), (x + w, y + h), (0, 0, 0), alpha=0.6)
    cv2.putText(frame, "METRICAS", (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    ang_avg = (ds.head_tilt_angle_sum / max(1, ds.head_tilt_angle_count))
    lines = [
        f"Bocejos: {ds.yawn_count}",
               f"Piscadas(30s): {ds.blink_count}",
        f"Total fech.: {ds.eyes_closed_total:.1f}s",
        f"Pico fech.: {ds.max_eyes_closed_streak:.1f}s",
        f"Semicerr.: {ds.eyes_semiclosed_total:.1f}s",
        f"Ang. vert. avg: {ang_avg:.1f}°",
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x + 10, y + 45 + i * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return frame

def draw_footer_info(frame):
    H, W = frame.shape[:2]
    frame = draw_transparent_rect(frame, (0, H - 35), (W, H), (0, 0, 0), alpha=0.6)
    cv2.putText(frame, "Versao 1.0 | Sistema de Monitoramento de Fadiga | Desenvolvido por: Wherbert Silva",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame

def show_start_screen():
    width, height = 640, 480
    bar_x, bar_y = 100, 360
    bar_width, bar_height = 440, 40
    try:
        pygame.mixer.music.load("Mensagens de Voz/Saudacao.mp3")
        pygame.mixer.music.play()
    except Exception:
        logger.warning("Mensagem de saudação não encontrada")
    start = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(start, "Deteccao", (120, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 240, 240), 5, cv2.LINE_AA)
    cv2.putText(start, "de Fadiga", (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 240, 240), 5, cv2.LINE_AA)
    cv2.putText(start, "Carregando sistema...", (140, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    cv2.rectangle(start, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    cv2.imshow("Tela Inicial", start); cv2.waitKey(400)
    for i in range(0, 101, 5):
        frame = start.copy()
        fill = int(bar_width * (i / 100.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_height), (0, 200, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
        cv2.imshow("Tela Inicial", frame); cv2.waitKey(120)
    time.sleep(0.6); cv2.destroyAllWindows()

# =========================
# Fuzzy
# =========================
def setup_fuzzy_system():
    freq_bocejos               = ctrl.Antecedent(np.arange(0, 10, 1),        'freq_bocejos')
    piscadas                   = ctrl.Antecedent(np.arange(0, 31, 1),        'piscadas')
    olhos_semicerrados         = ctrl.Antecedent(np.arange(0, 30.5, 0.5),    'olhos_semicerrados')
    tempo_por_piscada_max      = ctrl.Antecedent(np.arange(0, 5.1, 0.1),     'tempo_por_piscada_max')
    total_fechados             = ctrl.Antecedent(np.arange(0, 12.1, 0.1),    'total_fechados')
    inclinacao_vertical_graus  = ctrl.Antecedent(np.arange(0, 40, 0.5),      'inclinacao_vertical')
    tempo_conducao             = ctrl.Antecedent(np.arange(0, 180, 1),       'tempo_conducao')
    oscilacao_velocidade       = ctrl.Antecedent(np.arange(0, 50, 1),        'oscilacao_velocidade')

    nivel_fadiga               = ctrl.Consequent(np.arange(0, 10.1, 0.1),    'nivel_fadiga')

    # mfs
    freq_bocejos['nao_frequente'] = fuzz.trapmf(freq_bocejos.universe, [0, 0, 0, 1])
    freq_bocejos['frequente']     = fuzz.trapmf(freq_bocejos.universe, [1, 1, 10, 10])

    piscadas['pouca']     = fuzz.gaussmf(piscadas.universe, 4, 2)
    piscadas['frequente'] = fuzz.gaussmf(piscadas.universe, 8, 2)
    piscadas['muito']     = fuzz.gaussmf(piscadas.universe, 12, 3)

    olhos_semicerrados['baixo'] = fuzz.gaussmf(olhos_semicerrados.universe, 3, 2.0)
    olhos_semicerrados['medio'] = fuzz.gaussmf(olhos_semicerrados.universe, 9, 2.0)
    olhos_semicerrados['alto']  = fuzz.gaussmf(olhos_semicerrados.universe, 15, 2.5)

    tempo_por_piscada_max['pouco_tempo']         = fuzz.gaussmf(tempo_por_piscada_max.universe, 0.8, 0.3)
    tempo_por_piscada_max['tempo_intermediario'] = fuzz.gaussmf(tempo_por_piscada_max.universe, 1.7, 0.2)
    tempo_por_piscada_max['muito_tempo']         = fuzz.smf    (tempo_por_piscada_max.universe, 2.0, 2.3)

    total_fechados['baixo'] = fuzz.gaussmf(total_fechados.universe, 1.5, 0.8)
    total_fechados['medio'] = fuzz.gaussmf(total_fechados.universe, 3.9, 0.6)
    total_fechados['alto']  = fuzz.smf    (total_fechados.universe, 4.5, 7.5)

    inclinacao_vertical_graus['baixa']    = fuzz.gaussmf(inclinacao_vertical_graus.universe, 7.5, 3.0)
    inclinacao_vertical_graus['moderada'] = fuzz.gaussmf(inclinacao_vertical_graus.universe, 15.0, 4.0)
    inclinacao_vertical_graus['alta']     = fuzz.gaussmf(inclinacao_vertical_graus.universe, 25.0, 5.0)

    tempo_conducao['baixo'] = fuzz.trapmf(tempo_conducao.universe, [0, 0, 40, 50])
    tempo_conducao['medio'] = fuzz.trapmf(tempo_conducao.universe, [40, 80, 100, 120])
    tempo_conducao['alto']  = fuzz.trapmf(tempo_conducao.universe, [100, 120, 180, 180])

    oscilacao_velocidade['constante']     = fuzz.sigmf(oscilacao_velocidade.universe, 20, -1)
    oscilacao_velocidade['nao_constante'] = fuzz.sigmf(oscilacao_velocidade.universe, 20,  1)

    nivel_fadiga['leve']     = fuzz.trapmf(nivel_fadiga.universe, [2, 2, 4, 5])
    nivel_fadiga['moderada'] = fuzz.trimf (nivel_fadiga.universe, [4, 5.5, 7])
    nivel_fadiga['grave']    = fuzz.trapmf(nivel_fadiga.universe, [6, 7, 10, 10])

    # Regras (conjunto original mantido)
    rules = [
        # LEVE
        ctrl.Rule(olhos_semicerrados['baixo'] & tempo_por_piscada_max['pouco_tempo'] &
                  total_fechados['baixo'] & inclinacao_vertical_graus['baixa'] &
                  oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),
        ctrl.Rule(piscadas['frequente'] & olhos_semicerrados['baixo'], nivel_fadiga['leve']),
        ctrl.Rule(freq_bocejos['frequente'] & tempo_por_piscada_max['pouco_tempo'], nivel_fadiga['leve']),
        ctrl.Rule(tempo_conducao['baixo'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),
        ctrl.Rule(olhos_semicerrados['baixo'] & oscilacao_velocidade['constante'], nivel_fadiga['leve']),
        ctrl.Rule(piscadas['pouca'] & inclinacao_vertical_graus['baixa'] & tempo_conducao['medio'], nivel_fadiga['leve']),
        ctrl.Rule(piscadas['pouca'] & olhos_semicerrados['baixo'] & tempo_por_piscada_max['pouco_tempo'], nivel_fadiga['leve']),
        ctrl.Rule(freq_bocejos['frequente'] & piscadas['pouca'] & olhos_semicerrados['baixo'], nivel_fadiga['leve']),
        ctrl.Rule(oscilacao_velocidade['nao_constante'] & piscadas['pouca'] &
                  olhos_semicerrados['baixo'] & tempo_por_piscada_max['pouco_tempo'], nivel_fadiga['leve']),
        ctrl.Rule(total_fechados['baixo'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),
        ctrl.Rule(inclinacao_vertical_graus['moderada'] & tempo_por_piscada_max['pouco_tempo'] & olhos_semicerrados['baixo'], nivel_fadiga['leve']),
        ctrl.Rule(piscadas['frequente'] & total_fechados['baixo'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),
        ctrl.Rule(tempo_conducao['baixo'] & oscilacao_velocidade['constante'] & total_fechados['baixo'], nivel_fadiga['leve']),
        ctrl.Rule(inclinacao_vertical_graus['baixa'] & olhos_semicerrados['baixo'] & tempo_por_piscada_max['pouco_tempo'] & total_fechados['baixo'], nivel_fadiga['leve']),
        ctrl.Rule(piscadas['pouca'] & freq_bocejos['nao_frequente'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),
        ctrl.Rule(inclinacao_vertical_graus['baixa'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),
        ctrl.Rule(tempo_conducao['baixo'] & olhos_semicerrados['baixo'], nivel_fadiga['leve']),
        ctrl.Rule(inclinacao_vertical_graus['moderada'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),
        ctrl.Rule(piscadas['frequente'], nivel_fadiga['leve']),
        ctrl.Rule(freq_bocejos['nao_frequente'] & oscilacao_velocidade['nao_constante'] & tempo_conducao['medio'], nivel_fadiga['leve']),
        ctrl.Rule(total_fechados['baixo'] & tempo_por_piscada_max['pouco_tempo'], nivel_fadiga['leve']),
        ctrl.Rule(olhos_semicerrados['baixo'] & inclinacao_vertical_graus['moderada'], nivel_fadiga['leve']),
        ctrl.Rule(piscadas['pouca'] & olhos_semicerrados['baixo'] & oscilacao_velocidade['constante'], nivel_fadiga['leve']),
        ctrl.Rule(inclinacao_vertical_graus['baixa'] & tempo_conducao['medio'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['leve']),

        # MODERADA
        ctrl.Rule(olhos_semicerrados['medio'] & piscadas['frequente'], nivel_fadiga['moderada']),
        ctrl.Rule(tempo_por_piscada_max['tempo_intermediario'] & olhos_semicerrados['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(tempo_conducao['medio'] & olhos_semicerrados['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(tempo_conducao['alto'] & oscilacao_velocidade['constante'], nivel_fadiga['moderada']),
        ctrl.Rule(piscadas['muito'], nivel_fadiga['moderada']),
        ctrl.Rule(total_fechados['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(inclinacao_vertical_graus['moderada'] & oscilacao_velocidade['constante'], nivel_fadiga['moderada']),
        ctrl.Rule(oscilacao_velocidade['nao_constante'] & (piscadas['frequente'] | olhos_semicerrados['medio']), nivel_fadiga['moderada']),
        ctrl.Rule(oscilacao_velocidade['constante'] & (piscadas['frequente'] | olhos_semicerrados['medio']), nivel_fadiga['moderada']),
        ctrl.Rule(piscadas['frequente'] & olhos_semicerrados['medio'] & oscilacao_velocidade['constante'], nivel_fadiga['moderada']),
        ctrl.Rule(piscadas['muito'] & oscilacao_velocidade['constante'], nivel_fadiga['moderada']),
        ctrl.Rule(olhos_semicerrados['medio'] & tempo_conducao['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(olhos_semicerrados['baixo'] & oscilacao_velocidade['constante'] & tempo_conducao['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(piscadas['muito'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['moderada']),
        ctrl.Rule(piscadas['pouca'] & tempo_por_piscada_max['tempo_intermediario'] & tempo_conducao['alto'], nivel_fadiga['moderada']),
        ctrl.Rule(inclinacao_vertical_graus['baixa'] & olhos_semicerrados['medio'] & total_fechados['baixo'], nivel_fadiga['moderada']),
        ctrl.Rule(freq_bocejos['frequente'] & olhos_semicerrados['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(inclinacao_vertical_graus['moderada'] & tempo_conducao['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(piscadas['frequente'] & tempo_por_piscada_max['tempo_intermediario'], nivel_fadiga['moderada']),
        ctrl.Rule(total_fechados['medio'] & oscilacao_velocidade['constante'], nivel_fadiga['moderada']),
        ctrl.Rule(inclinacao_vertical_graus['moderada'] & total_fechados['medio'], nivel_fadiga['moderada']),
        ctrl.Rule(olhos_semicerrados['medio'] & (freq_bocejos['frequente'] | piscadas['muito']), nivel_fadiga['moderada']),
        ctrl.Rule(oscilacao_velocidade['constante'] & tempo_conducao['medio'] & inclinacao_vertical_graus['moderada'], nivel_fadiga['moderada']),
        ctrl.Rule(inclinacao_vertical_graus['baixa'] & total_fechados['medio'] & oscilacao_velocidade['nao_constante'], nivel_fadiga['moderada']),
        ctrl.Rule(tempo_por_piscada_max['tempo_intermediario'] & total_fechados['medio'] & inclinacao_vertical_graus['moderada'], nivel_fadiga['moderada']),
        ctrl.Rule(freq_bocejos['frequente'] & total_fechados['medio'] & olhos_semicerrados['medio'], nivel_fadiga['moderada']),

        # GRAVE
        ctrl.Rule(tempo_por_piscada_max['muito_tempo'], nivel_fadiga['grave']),
        ctrl.Rule(total_fechados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(olhos_semicerrados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_por_piscada_max['tempo_intermediario'] & olhos_semicerrados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(inclinacao_vertical_graus['alta'] & olhos_semicerrados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(inclinacao_vertical_graus['alta'] & tempo_por_piscada_max['tempo_intermediario'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_conducao['alto'] & oscilacao_velocidade['constante'] & olhos_semicerrados['medio'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_conducao['alto'] & piscadas['muito'], nivel_fadiga['grave']),
        ctrl.Rule(freq_bocejos['frequente'] & tempo_por_piscada_max['tempo_intermediario'] & inclinacao_vertical_graus['moderada'], nivel_fadiga['grave']),
        ctrl.Rule(piscadas['muito'] & olhos_semicerrados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(oscilacao_velocidade['constante'] & inclinacao_vertical_graus['alta'], nivel_fadiga['grave']),
        ctrl.Rule(oscilacao_velocidade['constante'] & (tempo_por_piscada_max['tempo_intermediario'] | olhos_semicerrados['alto']), nivel_fadiga['grave']),
        ctrl.Rule(tempo_conducao['alto'] & inclinacao_vertical_graus['alta'], nivel_fadiga['grave']),
        ctrl.Rule(piscadas['pouca'] & olhos_semicerrados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(oscilacao_velocidade['constante'] & tempo_conducao['alto'] & inclinacao_vertical_graus['moderada'] & olhos_semicerrados['medio'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_por_piscada_max['tempo_intermediario'] & (olhos_semicerrados['alto'] | inclinacao_vertical_graus['alta']), nivel_fadiga['grave']),
        ctrl.Rule(oscilacao_velocidade['constante'] & tempo_conducao['medio'] & (piscadas['frequente'] | olhos_semicerrados['medio']), nivel_fadiga['grave']),
        ctrl.Rule(olhos_semicerrados['alto'] & tempo_conducao['baixo'], nivel_fadiga['grave']),
        ctrl.Rule(inclinacao_vertical_graus['moderada'] & tempo_conducao['alto'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_por_piscada_max['tempo_intermediario'] & oscilacao_velocidade['constante'] & tempo_conducao['alto'], nivel_fadiga['grave']),
        ctrl.Rule(total_fechados['alto'] & oscilacao_velocidade['constante'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_por_piscada_max['muito_tempo'] & total_fechados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_por_piscada_max['muito_tempo'] & inclinacao_vertical_graus['alta'], nivel_fadiga['grave']),
        ctrl.Rule(total_fechados['alto'] & inclinacao_vertical_graus['alta'], nivel_fadiga['grave']),
        ctrl.Rule(olhos_semicerrados['medio'] & total_fechados['medio'] & oscilacao_velocidade['constante'], nivel_fadiga['grave']),
        ctrl.Rule(olhos_semicerrados['medio'] & total_fechados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(tempo_por_piscada_max['tempo_intermediario'] & total_fechados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(freq_bocejos['frequente'] & total_fechados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(piscadas['pouca'] & total_fechados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(piscadas['muito'] & total_fechados['alto'], nivel_fadiga['grave']),
        ctrl.Rule(inclinacao_vertical_graus['alta'] & total_fechados['medio'], nivel_fadiga['grave']),
        ctrl.Rule(inclinacao_vertical_graus['moderada'] & olhos_semicerrados['alto'], nivel_fadiga['grave']),
    ]

    sistema_controle = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(sistema_controle)

    nivel_universe = nivel_fadiga.universe.copy()
    nivel_mfs = {
        'leve': nivel_fadiga['leve'].mf.copy(),
        'moderada': nivel_fadiga['moderada'].mf.copy(),
        'grave': nivel_fadiga['grave'].mf.copy()
    }
    return sim, nivel_universe, nivel_mfs

# --------------------
# SALVAMENTO (gráficos e linha)
# --------------------
def save_fuzzy_plot(window_idx, nivel_para_plot, universe, mfs_dict, mu_dict, outdir: Path, suffix: str = ""):
    """
    Salva o gráfico das saídas fuzzy (funções de pertinência) usando cores:
      Leve [2–5]     = verde
      Moderada [4–7] = laranja
      Grave [6–10]   = vermelho
    O preenchimento também segue essas cores.
    """
    try:
        # Cores solicitadas
        c_leve     = "#25A55F"  # verde
        c_moderada = "#F28C28"  # laranja
        c_grave    = "#D7263D"  # vermelho

        fig, ax = plt.subplots(figsize=(7.8, 3.2), dpi=120)

        # Linhas das MFs
        ax.plot(universe, mfs_dict['leve'],     linewidth=2.4, label='Leve [2–5]',     color=c_leve)
        ax.plot(universe, mfs_dict['moderada'], linewidth=2.4, label='Moderada [4–7]', color=c_moderada)
        ax.plot(universe, mfs_dict['grave'],    linewidth=2.4, label='Grave [6–10]',   color=c_grave)

        # Preenchimento até o nível de pertinência calculado
        ax.fill_between(universe, 0, np.minimum(mu_dict.get('leve', 0.0),     mfs_dict['leve']),
                        alpha=0.25, color=c_leve)
        ax.fill_between(universe, 0, np.minimum(mu_dict.get('moderada', 0.0), mfs_dict['moderada']),
                        alpha=0.25, color=c_moderada)
        ax.fill_between(universe, 0, np.minimum(mu_dict.get('grave', 0.0),    mfs_dict['grave']),
                        alpha=0.25, color=c_grave)

        # Marcador vertical no valor exibido/tabelado (mesmo usado na UI e CSV)
        ax.axvline(min(nivel_para_plot, 10.0), linestyle='--', linewidth=2, color="#333333")

        ax.set_title("Saída — Grau de Fadiga (0–10)")
        ax.set_xlabel("Índice (0–10)")
        ax.set_ylabel("Pertinência")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        outdir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outdir / f"saida_fuzzy_janela_{window_idx:03d}{suffix}.png")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Falha ao salvar gráfico fuzzy da janela {window_idx}{suffix}: {e}")



def save_gauge_plot(window_idx: int, value: float, outdir: Path, fname_prefix: str = "fadigometro"):
    """
    Gera um fadigômetro horizontal com 3 faixas sobrepostas:
    - Fadiga Leve (verde): [2-5]
    - Fadiga Moderada (laranja): [4-7]
    - Fadiga Grave (vermelho): [6-10]
    """
    try:
        v = float(np.clip(value, 0.0, 10.0))

        # Criar figura
        fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
        ax.set_aspect('equal')
        ax.axis('off')

        # Configurações do gauge horizontal
        gauge_width = 8.0
        gauge_height = 0.8
        gauge_x = 0.0
        gauge_y = 0.0

        # Fundo base do gauge
        base_rect = plt.Rectangle(
            (gauge_x, gauge_y - gauge_height / 2),
            gauge_width,
            gauge_height,
            facecolor='#F1F5F9',
            edgecolor='#E2E8F0',
            linewidth=1,
            alpha=0.3
        )
        ax.add_patch(base_rect)

        # Faixa Verde: Fadiga Leve [2-5]
        verde_start = (2.0 / 10.0) * gauge_width
        verde_width = ((5.0 - 2.0) / 10.0) * gauge_width
        verde_rect = plt.Rectangle(
            (gauge_x + verde_start, gauge_y - gauge_height / 2),
            verde_width,
            gauge_height,
            facecolor='#22C55E',
            edgecolor='none',
            alpha=0.85
        )
        ax.add_patch(verde_rect)

        # Faixa Laranja: Fadiga Moderada [4-7] (sobreposta)
        laranja_start = (4.0 / 10.0) * gauge_width
        laranja_width = ((7.0 - 4.0) / 10.0) * gauge_width
        laranja_rect = plt.Rectangle(
            (gauge_x + laranja_start, gauge_y - gauge_height / 2),
            laranja_width,
            gauge_height,
            facecolor='#F59E0B',
            edgecolor='none',
            alpha=0.8
        )
        ax.add_patch(laranja_rect)

        # Faixa Vermelha: Fadiga Grave [6-10] (sobreposta)
        vermelho_start = (6.0 / 10.0) * gauge_width
        vermelho_width = ((10.0 - 6.0) / 10.0) * gauge_width
        vermelho_rect = plt.Rectangle(
            (gauge_x + vermelho_start, gauge_y - gauge_height / 2),
            vermelho_width,
            gauge_height,
            facecolor='#EF4444',
            edgecolor='none',
            alpha=0.8
        )
        ax.add_patch(vermelho_rect)

        # Marcações da escala
        tick_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        major_ticks = [0, 2, 4, 5, 6, 7, 10]  # Marcações principais

        for tick in tick_values:
            x_pos = gauge_x + (tick / 10.0) * gauge_width
            is_major = tick in major_ticks

            # Linha da marcação
            tick_height = 0.15 if is_major else 0.1
            ax.plot(
                [x_pos, x_pos],
                [gauge_y - gauge_height / 2 - tick_height, gauge_y - gauge_height / 2],
                color='#475569',
                linewidth=2 if is_major else 1.2,
                alpha=0.8
            )

            # Números da escala (apenas marcações principais)
            if is_major:
                ax.text(
                    x_pos,
                    gauge_y - gauge_height / 2 - tick_height - 0.2,
                    str(tick),
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='#374151',
                    weight='bold'
                )

        # Labels das faixas dentro do gauge
        # Label "LEVE"
        leve_center = gauge_x + verde_start + verde_width / 2
        ax.text(leve_center, gauge_y, "LEVE", ha='center', va='center',
                fontsize=10, color='white', weight='bold', alpha=0.9)

        # Label "MODERADA"
        moderada_center = gauge_x + laranja_start + laranja_width / 2
        ax.text(moderada_center, gauge_y, "MODERADA", ha='center', va='center',
                fontsize=10, color='white', weight='bold', alpha=0.9)

        # Label "GRAVE"
        grave_center = gauge_x + vermelho_start + vermelho_width / 2
        ax.text(grave_center, gauge_y, "GRAVE", ha='center', va='center',
                fontsize=10, color='white', weight='bold', alpha=0.9)

        # Indicador de valor atual
        valor_x = gauge_x + (v / 10.0) * gauge_width

        # Linha indicadora
        ax.plot(
            [valor_x, valor_x],
            [gauge_y - gauge_height / 2 - 0.3, gauge_y + gauge_height / 2 + 0.3],
            color='#1E40AF',
            linewidth=4,
            alpha=0.9,
            solid_capstyle='round'
        )

        # Círculo no topo do indicador
        circle = plt.Circle(
            (valor_x, gauge_y + gauge_height / 2 + 0.15),
            0.12,
            color='#1E40AF',
            alpha=0.9,
            zorder=10
        )
        ax.add_patch(circle)

        # Determinar status baseado no valor
        def get_status_text(val):
            if val <= 2:
                return "Sem Fadiga", "#10B981"
            elif val <= 5:
                return "Fadiga Leve", "#22C55E"
            elif val <= 7:
                return "Fadiga Moderada", "#F59E0B"
            else:
                return "Fadiga Grave", "#EF4444"

        status_text, status_color = get_status_text(v)

        # Título principal
        ax.text(
            gauge_x + gauge_width / 2,
            gauge_y + 1.2,
            "FADIGÔMETRO",
            ha='center',
            va='center',
            fontsize=16,
            weight='bold',
            color='#1f2937'
        )

        # Subtítulo com sistema fuzzy
        ax.text(
            gauge_x + gauge_width / 2,
            gauge_y + 0.9,
            "Sistema de Monitoramento Fuzzy",
            ha='center',
            va='center',
            fontsize=12,
            color='#6b7280',
            style='italic'
        )

        # Valor atual com status
        ax.text(
            gauge_x + gauge_width / 2,
            gauge_y - 1.0,
            f"Índice: {v:.1f}",
            ha='center',
            va='center',
            fontsize=14,
            weight='bold',
            color='#374151'
        )

        # Status da fadiga
        ax.text(
            gauge_x + gauge_width / 2,
            gauge_y - 1.3,
            status_text,
            ha='center',
            va='center',
            fontsize=13,
            weight='bold',
            color=status_color
        )

        # Escala de referência
        ax.text(
            gauge_x + gauge_width / 2,
            gauge_y - 1.6,
            "(Escala: 0 - 10)",
            ha='center',
            va='center',
            fontsize=10,
            color='#9ca3af'
        )

        # Legenda das faixas (canto inferior)
        legend_y = gauge_y - 2.2
        legend_spacing = gauge_width / 3

        # Fadiga Leve
        ax.add_patch(plt.Rectangle((gauge_x + 0.5, legend_y - 0.1), 0.3, 0.2,
                                   facecolor='#22C55E', alpha=0.8))
        ax.text(gauge_x + 1.0, legend_y, "Leve (2-5)", fontsize=9,
                color='#374151', va='center')

        # Fadiga Moderada
        ax.add_patch(plt.Rectangle((gauge_x + 2.8, legend_y - 0.1), 0.3, 0.2,
                                   facecolor='#F59E0B', alpha=0.8))
        ax.text(gauge_x + 3.3, legend_y, "Moderada (4-7)", fontsize=9,
                color='#374151', va='center')

        # Fadiga Grave
        ax.add_patch(plt.Rectangle((gauge_x + 5.1, legend_y - 0.1), 0.3, 0.2,
                                   facecolor='#EF4444', alpha=0.8))
        ax.text(gauge_x + 5.6, legend_y, "Grave (6-10)", fontsize=9,
                color='#374151', va='center')

        # Configurar limites da visualização
        ax.set_xlim(-0.5, gauge_width + 0.5)
        ax.set_ylim(gauge_y - 2.6, gauge_y + 1.5)

        # Salvar figura
        outdir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(
            outdir / f"{fname_prefix}_janela_{window_idx:03d}.png",
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            dpi=140
        )
        plt.close(fig)

    except Exception as e:
        logger.error(f"Falha ao salvar Fadigômetro da janela {window_idx}: {e}")

def append_window_row(window_idx, t_ini, t_fim, ds: DriverState,
                      tempo_conducao, oscilacao_velocidade,
                      nivel_para_tabela, status_text,
                      mu_leve, mu_moderada, mu_grave,
                      microssono_prev=False, microssono_na_janela=False):
    """Empilha em memória e grava CSV incremental para cada janela."""
    global janela_rows, csv_path
    ang_avg = (ds.head_tilt_angle_sum / max(1, ds.head_tilt_angle_count))

    row = {
        "Janela": window_idx,
        "Inicio": t_ini.strftime("%Y-%m-%d %H:%M:%S"),
        "Fim":    t_fim.strftime("%Y-%m-%d %H:%M:%S"),
        "Freq_Bocejos": int(ds.yawn_count),
        "Piscadas_30s": int(ds.blink_count),
        "Olhos_Semicerrados_s": round(ds.eyes_semiclosed_total, 2),
        "Tempo_por_Piscada_MAX_s": round(ds.max_eyes_closed_streak, 2),
        "Total_Olhos_Fechados_s": round(ds.eyes_closed_total, 2),
        "Angulo_Vertical_avg_graus": round(ang_avg, 1),
        "Tempo_Conducao_min": round(tempo_conducao, 1),
        "Oscilacao_Velocidade_kmh": round(oscilacao_velocidade, 1),
        "Nivel_Fadiga": round(float(nivel_para_tabela), 2),  # exatamente o que vai pra UI/gráfico
        "Classificacao": status_text,
        "Houve_Microssono_na_Janela": bool(microssono_na_janela),
        "Microssono_>2s_na_Janela_Anterior": bool(microssono_prev),
        "mu_Leve": round(mu_leve, 3),
        "mu_Moderada": round(mu_moderada, 3),
        "mu_Grave": round(mu_grave, 3),
    }

    janela_rows.append(row)
    try:
        df_tmp = pd.DataFrame([row])
        header = not csv_path.exists()
        df_tmp.to_csv(csv_path, sep=';', index=False, mode='a', header=header, encoding='utf-8')
    except Exception as e:
        logger.error(f"Falha ao gravar CSV incremental: {e}")

# ----------------------------
# Processamento de frame
# ----------------------------
def process_frame(frame, ds: DriverState, start_time, fps):
    if frame is None:
        return frame

    # Reduz resolução
    scale_percent = 80
    w = int(frame.shape[1] * scale_percent / 100)
    h = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    try:
        face_rois = detect_face_with_yolo(frame, model)
        candidate_bbox = None
        if face_rois:
            candidate_bbox = max((b for _, b in face_rois), key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        if candidate_bbox is None:
            if ds.last_bbox is not None and ds.missed_frames < 5:
                candidate_bbox = ds.last_bbox
                ds.missed_frames += 1
        else:
            candidate_bbox = ema_bbox(ds.last_bbox, candidate_bbox,
                                      alpha=0.85 if iou(ds.last_bbox, candidate_bbox) >= 0.3 else 0.7)
            ds.missed_frames = 0

        if candidate_bbox is not None:
            ds.last_bbox = candidate_bbox
            x1, y1, x2, y2 = candidate_bbox
            try:
                roi = frame[y1:y2, x1:x2].copy()
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi_res = cv2.resize(roi_rgb, (480, 360))
                results = face_mesh.process(roi_res)
                scale_x = (x2 - x1) / 480.0
                scale_y = (y2 - y1) / 360.0
                if results.multi_face_landmarks:
                    lms = [(int(lm.x * 480 * scale_x + x1), int(lm.y * 360 * scale_y + y1))
                           for lm in results.multi_face_landmarks[0].landmark]
                    lms = smooth_landmarks(lms)

                    left_ear  = calculate_ear(lms, LEFT_EYE_INDICES)
                    right_ear = calculate_ear(lms, RIGHT_EYE_INDICES)
                    ear_avg   = (left_ear + right_ear) / 2.0
                    mar       = calculate_mar(lms)

                    _, v_ang, tilt_cat = calculate_head_tilt(lms, ds)
                    frame = draw_head_tilt_indicator(frame, v_ang, tilt_cat)
                    ds.head_tilt_angle_sum   += abs(v_ang)
                    ds.head_tilt_angle_count += 1

                    # olhos fechados
                    if ear_avg < EAR_THRESHOLD:
                        if ds.eyes_closed_start == 0:
                            ds.eyes_closed_start = time.time()
                        else:
                            dur = time.time() - ds.eyes_closed_start
                            ds.max_eyes_closed_streak = max(ds.max_eyes_closed_streak, dur)
                    else:
                        if ds.eyes_closed_start > 0:
                            duration = time.time() - ds.eyes_closed_start
                            if 0.1 <= duration <= 1.5:
                                ds.blink_count += 1
                            if duration > 0.2:
                                ds.eyes_closed_total += duration
                                ds.max_eyes_closed_streak = max(ds.max_eyes_closed_streak, duration)
                            ds.eyes_closed_start = 0

                    # semicerrados
                    if EAR_SEMI_THRESHOLD > ear_avg >= EAR_THRESHOLD:
                        if ds.eyes_semiclosed_start == 0:
                            ds.eyes_semiclosed_start = time.time()
                    else:
                        if ds.eyes_semiclosed_start > 0:
                            ds.eyes_semiclosed_total += (time.time() - ds.eyes_semiclosed_start)
                            ds.eyes_semiclosed_start = 0

                    # microssono: SOM imediato e sinalizar carry para próxima janela
                    if ds.eyes_closed_start > 0:
                        eyes_closed_duration = time.time() - ds.eyes_closed_start
                        if eyes_closed_duration >= CRITICAL_EYES_CLOSED_SEC:
                            play_alert("FadigaCritica.mp3")
                            ds.microssono_in_window = True
                            ds.defer_grave_to_next = True
                            # consolida tempo/pico e zera contagem em andamento
                            ds.eyes_closed_total += eyes_closed_duration
                            ds.max_eyes_closed_streak = max(ds.max_eyes_closed_streak, eyes_closed_duration)
                            ds.eyes_closed_start = 0

                    # bocejo
                    if mar > MAR_THRESHOLD:
                        if (not ds.is_yawning) and ((time.time() - ds.last_yawn_time) > YAWN_COOLDOWN):
                            ds.yawn_start_time = time.time(); ds.is_yawning = True
                    elif ds.is_yawning:
                        if (time.time() - ds.yawn_start_time) >= MIN_YAWN_DURATION:
                            ds.yawn_count += 1; ds.last_yawn_time = time.time()
                        ds.is_yawning = False

                    # pontos essenciais
                    pontos = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MOUTH_INDICES + [NOSE_TIP_INDEX, CHIN_INDEX, FOREHEAD_INDEX]
                    for idx in pontos:
                        if 0 <= idx < len(lms):
                            cv2.circle(frame, (int(lms[idx][0]), int(lms[idx][1])), 1, (0, 255, 255), -1)

                # bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(frame, "Face", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            except Exception as e:
                logger.error(f"Erro ROI: {e}")
        else:
            cv2.putText(frame, "Rosto nao detectado", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # header/painéis
        elapsed = int(time.time() - start_time)
        note = None
        if ds.show_note_grave_from_microssono and time.time() < ds._note_timer_until:
            note = f"Grave (microssono ≥ {CRITICAL_EYES_CLOSED_SEC:.0f}s na janela anterior)"
        frame = draw_header_info(frame, ds.current_status, ds.fuzzy_level, elapsed, note_text=note)
        frame = draw_metrics_panel(frame, ds)
        frame = draw_footer_info(frame)

        # expirar nota
        if ds.show_note_grave_from_microssono and time.time() >= ds._note_timer_until:
            ds.show_note_grave_from_microssono = False

    except Exception as e:
        logger.error(f"Erro no frame: {e}")

    return frame

# --------------------
# Loop principal
# --------------------
def main():
    global janela_idx
    ds = DriverState()
    sistema_fuzzy, nivel_universe, nivel_mfs = setup_fuzzy_system()

    calculate_telemetria_fixed_values()
    show_start_screen()

    last_fuzzy_update = time.time()
    fuzzy_update_interval = 30  # s
    frame_count = 0
    start_time = time.time()
    janela_t_ini = datetime.now()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = int(1000 / fps)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.info("Fim do vídeo.")
                break

            frame_count += 1
            now = time.time()

            # pular alguns frames para aliviar
            if frame_count % 2 != 0:
                continue

            frame = process_frame(frame, ds, start_time, fps)

            # ======= A CADA 30s: sempre calcular e salvar =======
            if now - last_fuzzy_update >= fuzzy_update_interval:
                try:
                    tempo_conducao, oscilacao_velocidade = get_telemetria_data(now - start_time)
                    ang_avg = (ds.head_tilt_angle_sum / max(1, ds.head_tilt_angle_count))

                    # (1) DETECÇÃO "SEM ENTRADAS" NA JANELA
                    no_samples = (ds.head_tilt_angle_count == 0)
                    no_inputs = (
                        ds.yawn_count == 0 and
                        ds.blink_count == 0 and
                        ds.eyes_semiclosed_total == 0 and
                        ds.max_eyes_closed_streak == 0 and
                        ds.eyes_closed_total == 0 and
                        abs(ang_avg) == 0
                    )

                    microssono_prev = bool(ds.defer_grave_to_next)

                    if no_samples and no_inputs and not microssono_prev:
                        # Janela sem dados reais: não rodar fuzzy; 0.0 / Leve
                        nivel_crisp_real = 0.0
                        status_text = "Leve"
                        nivel_para_table_e_plot = 0.0
                        mu_leve = mu_moderada = mu_grave = 0.0
                    else:
                        # (2) PREPARO DAS ENTRADAS (com carry de microssono, se houver)
                        in_tempo_pico = np.clip(ds.max_eyes_closed_streak, 0.0, 5.0)
                        in_total_fech = np.clip(ds.eyes_closed_total, 0.0, 12.0)

                        if microssono_prev:
                            # empurra a janela atual para a região GRAVE do conjunto (para coerência)
                            in_tempo_pico = max(in_tempo_pico, 2.1)
                            in_total_fech = max(in_total_fech, 4.6)

                        # (3) RODA O FUZZY
                        sistema_fuzzy.input['freq_bocejos']          = min(ds.yawn_count, 10)
                        sistema_fuzzy.input['piscadas']              = min(ds.blink_count, 30)
                        sistema_fuzzy.input['olhos_semicerrados']    = min(ds.eyes_semiclosed_total, 30)
                        sistema_fuzzy.input['tempo_por_piscada_max'] = in_tempo_pico
                        sistema_fuzzy.input['total_fechados']        = in_total_fech
                        sistema_fuzzy.input['inclinacao_vertical']   = np.clip(ang_avg, 0.0, 40.0)
                        sistema_fuzzy.input['tempo_conducao']        = min(tempo_conducao, 180)
                        sistema_fuzzy.input['oscilacao_velocidade']  = min(oscilacao_velocidade, 50)

                        sistema_fuzzy.compute()
                        nivel_crisp_real = float(sistema_fuzzy.output['nivel_fadiga'])

                        # Classe textual (sem override ainda)
                        if nivel_crisp_real <= 4:
                            status_text = "Leve"
                        elif nivel_crisp_real <= 7:
                            status_text = "Moderada"
                        else:
                            status_text = "Grave"

                        # --- NOVO: override somente quando veio de microssono e o fuzzy NÃO deu Grave
                        override_from_microssono = (microssono_prev and status_text != "Grave")

                        if override_from_microssono:
                            status_text = "Grave"
                            nivel_para_table_e_plot = 10.0
                            mu_leve, mu_moderada, mu_grave = 0.0, 0.0, 1.0
                        else:
                            nivel_para_table_e_plot = nivel_crisp_real
                            mu_leve     = float(interp_membership(nivel_universe, nivel_mfs['leve'],     nivel_para_table_e_plot))
                            mu_moderada = float(interp_membership(nivel_universe, nivel_mfs['moderada'], nivel_para_table_e_plot))
                            mu_grave    = float(interp_membership(nivel_universe, nivel_mfs['grave'],    nivel_para_table_e_plot))

                    # Atualiza UI com o mesmo valor que será plotado/tabelado
                    ds.fuzzy_level = nivel_para_table_e_plot
                    ds.current_status = status_text

                    # Nota na tela quando o "Grave" é herdado de microssono
                    if microssono_prev:
                        ds.show_note_grave_from_microssono = True
                        ds._note_timer_until = time.time() + 6.0  # 6s

                    # (5) GRÁFICOS (fuzzy e fadigômetro)
                    save_fuzzy_plot(
                        window_idx=janela_idx,
                        nivel_para_plot=nivel_para_table_e_plot,
                        universe=nivel_universe,
                        mfs_dict=nivel_mfs,
                        mu_dict={'leve': mu_leve, 'moderada': mu_moderada, 'grave': mu_grave},
                        outdir=output_dir_path,
                        suffix=""
                    )
                    save_gauge_plot(
                        window_idx=janela_idx,
                        value=nivel_para_table_e_plot,
                        outdir=output_dir_path,
                        fname_prefix="fadigometro"
                    )

                    # (6) LINHA NA TABELA — sempre a cada 30s
                    janela_t_fim = datetime.now()
                    append_window_row(
                        window_idx=janela_idx,
                        t_ini=janela_t_ini,
                        t_fim=janela_t_fim,
                        ds=ds,
                        tempo_conducao=tempo_conducao,
                        oscilacao_velocidade=oscilacao_velocidade,
                        nivel_para_tabela=nivel_para_table_e_plot,  # exatamente o mostrado
                        status_text=status_text,
                        mu_leve=mu_leve,
                        mu_moderada=mu_moderada,
                        mu_grave=mu_grave,
                        microssono_prev=microssono_prev,
                        microssono_na_janela=ds.microssono_in_window
                    )

                    # preparar próxima janela
                    janela_idx += 1
                    janela_t_ini = datetime.now()

                    # Reset acumuladores da janela
                    ds.blink_count = 0
                    ds.yawn_count = 0
                    ds.eyes_semiclosed_total = 0
                    ds.eyes_closed_total = 0
                    ds.max_eyes_closed_streak = 0.0
                    ds.head_tilt_angle_sum = 0.0
                    ds.head_tilt_angle_count = 0

                    # Consome “carry” de microssono anterior e limpa flag da janela encerrada
                    if microssono_prev:
                        ds.defer_grave_to_next = False
                    ds.microssono_in_window = False

                    last_fuzzy_update = now

                except Exception as e:
                    logger.error(f"Erro no passo de defuzzificação/salvamento: {e}")

            # Exibição
            cv2.imshow('Monitoramento de Fadiga', frame)
            if cv2.waitKey(frame_delay) == 27:
                break

    except KeyboardInterrupt:
        logger.info("Interrupção manual.")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
    finally:
        try:
            if janela_rows:
                df = pd.DataFrame(janela_rows).sort_values("Janela")
                df.to_excel(excel_path, index=False)
                logger.info(f"Planilha salva em: {excel_path}")
            else:
                logger.info("Nenhuma janela processada.")
        except Exception as e:
            logger.error(f"Falha ao salvar Excel final: {e}")
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        logger.info("Sistema encerrado.")

if __name__ == "__main__":
    main()
