import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

VIDEO_DIR      = "C:\PythonProject\PythonProject3"
VIDEO_FILENAME = "sheesh.mp4"
VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_FILENAME) #Итоговый путь

OUTPUT_DIR      = "C:\Анжуманя"
OUTPUT_FILENAME = "result.mp4"   # None - не сохранять
OUTPUT_PATH = (
    os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if OUTPUT_FILENAME else None
)

ANGLE_DOWN_THRESHOLD = 90    # угол локтя внизу (опустился)
ANGLE_UP_THRESHOLD   = 155   # угол локтя вверху (выпрямился)
MIN_VISIBILITY       = 0.5   # минимальная видимость точки


#Функции

@dataclass
class PushupEvent:
    count: int
    timestamp_sec: float
    timestamp_str: str


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Угол в точке b (в градусах) между лучами ba и bc."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def get_landmark_xy(landmarks, idx: int, w: int, h: int) -> Optional[np.ndarray]:
    lm = landmarks[idx]
    if lm.visibility < MIN_VISIBILITY:
        return None
    return np.array([lm.x * w, lm.y * h])


#Каунтер
class PushupCounter:
    def __init__(self):
        self.count = 0
        self.state = "up"          # "up" | "down"
        self.events: List[PushupEvent] = []
        self.current_angle = 0.0

    def update(self, left_angle: Optional[float],
               right_angle: Optional[float],
               timestamp: float) -> bool:
        """Обновить состояние. Возвращает True, если зафиксировано новое отжимание."""
        angle = None
        if left_angle is not None and right_angle is not None:
            angle = (left_angle + right_angle) / 2
        elif left_angle is not None:
            angle = left_angle
        elif right_angle is not None:
            angle = right_angle

        if angle is None:
            return False

        self.current_angle = angle
        new_rep = False

        if angle < ANGLE_DOWN_THRESHOLD and self.state == "up":
            self.state = "down"
        elif angle > ANGLE_UP_THRESHOLD and self.state == "down":
            self.state = "up"
            self.count += 1
            ts = format_time(timestamp)
            self.events.append(PushupEvent(self.count, timestamp, ts))
            new_rep = True

        return new_rep


#Отрисовка скелета

def draw_overlay(frame: np.ndarray, counter: PushupCounter,
                 fps: float, frame_idx: int) -> np.ndarray:
    h, w = frame.shape[:2]

    #Полупрозрачная панелька слева
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 180), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    #Заголовок
    cv2.putText(frame, "PUSHUPS", (12, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (80, 200, 120), 2, cv2.LINE_AA)

    # Счётчик
    cv2.putText(frame, str(counter.count), (12, 105),
                cv2.FONT_HERSHEY_DUPLEX, 3.2, (255, 255, 255), 4, cv2.LINE_AA)

    # Фаза отжимания
    phase_color = (50, 220, 50) if counter.state == "up" else (50, 100, 255)
    phase_label = "UP  " if counter.state == "up" else "DOWN"
    cv2.putText(frame, phase_label, (12, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, phase_color, 2, cv2.LINE_AA)

    # Угол локтя
    angle_txt = f"elbow: {counter.current_angle:.0f}°"
    cv2.putText(frame, angle_txt, (12, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    #Таймкод
    ts = format_time(frame_idx / fps) if fps > 0 else "00:00"
    cv2.putText(frame, ts, (w - 130, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Последние события (справа)
    recent = counter.events[-5:]
    for i, ev in enumerate(reversed(recent)):
        alpha = 1.0 - i * 0.18
        color = tuple(int(c * alpha) for c in (120, 230, 120))
        txt = f"#{ev.count}  {ev.timestamp_str}"
        cv2.putText(frame, txt, (w - 200, 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    return frame


def draw_skeleton(frame, landmarks, mp_pose, w, h):
    """Рисует ключевые точки и соединения для рук/корпуса."""
    connections = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW,     mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW,    mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    ]

    lm_list = landmarks.landmark
    for start_lm, end_lm in connections:
        s = lm_list[start_lm.value]
        e = lm_list[end_lm.value]
        if s.visibility > MIN_VISIBILITY and e.visibility > MIN_VISIBILITY:
            sx, sy = int(s.x * w), int(s.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            cv2.line(frame, (sx, sy), (ex, ey), (100, 200, 255), 2, cv2.LINE_AA)

    key_points = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,    mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,    mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,      mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    for kp in key_points:
        lm = lm_list[kp.value]
        if lm.visibility > MIN_VISIBILITY:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, (100, 200, 255),  1, cv2.LINE_AA)


#Main
def process_video(video_path: str, output_path: Optional[str] = None,
                  show: bool = True) -> List[PushupEvent]:

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counter   = PushupCounter()
    frame_idx = 0
    start_real = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp  = frame_idx / fps

            #ОБнаружение позы
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            left_angle = right_angle = None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                def xy(idx):
                    return get_landmark_xy(lms, idx, width, height)

                #Левая рука
                ls = xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                le = xy(mp_pose.PoseLandmark.LEFT_ELBOW.value)
                lw = xy(mp_pose.PoseLandmark.LEFT_WRIST.value)
                if ls is not None and le is not None and lw is not None:
                    left_angle = calc_angle(ls, le, lw)

                #Правая рука
                rs = xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                re = xy(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                rw = xy(mp_pose.PoseLandmark.RIGHT_WRIST.value)
                if rs is not None and re is not None and rw is not None:
                    right_angle = calc_angle(rs, re, rw)

                new_rep = counter.update(left_angle, right_angle, timestamp)

                draw_skeleton(frame, results.pose_landmarks, mp_pose, width, height)

                if new_rep:
                    #Эффект вспышки
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (width, height), (50, 255, 50), -1)
                    cv2.addWeighted(flash, 0.15, frame, 0.85, 0, frame)
            else:
                counter.update(None, None, timestamp)

            frame = draw_overlay(frame, counter, fps, frame_idx)

            #Прогресс в терминале
            if frame_idx % 30 == 0:
                pct = frame_idx / total * 100 if total > 0 else 0
                elapsed = time.time() - start_real
                print(f"\r  Обработано: {frame_idx}/{total} ({pct:.1f}%) | "
                      f"Отжиманий: {counter.count} | "
                      f"Время: {elapsed:.1f}s", end="", flush=True)

            if writer:
                writer.write(frame)

            if show:
                cv2.imshow("Pushup Counter — нажми Q для выхода", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    print()  #Новая строка после прогресс-бара
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    return counter.events


def print_report(events: List[PushupEvent], video_path: str):
    print("\n" + "═" * 50)
    print(f"  РЕЗУЛЬТАТЫ:  {video_path}")
    print("═" * 50)
    print(f"  Всего отжиманий: {len(events)}")

    if len(events) >= 2:
        durations = [
            events[i].timestamp_sec - events[i-1].timestamp_sec
            for i in range(1, len(events))
        ]
        avg = np.mean(durations)
        print(f"  Среднее время между отжиманиями: {avg:.2f}s")
        print(f"  Темп (отж/мин): {60/avg:.1f}")

    print("\n  Временна́я метка каждого отжимания:")
    print("  " + "─" * 30)
    for ev in events:
        print(f"    #{ev.count:>3}  →  {ev.timestamp_str}")
    print("═" * 50 + "\n")

    pushups_total = len(events)
# Для Терминала CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Счётчик отжиманий через MediaPipe Pose"
    )
    # Если аргумент не передан — берём путь из переменных VIDEO_DIR / VIDEO_FILENAME
    parser.add_argument("--video",  default=VIDEO_PATH,
                        help=f"Путь к входному видео (по умолчанию: {VIDEO_PATH})")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help="Путь для сохранения результата (опционально)")
    parser.add_argument("--no-show", action="store_true",
                        help="Не показывать окно предпросмотра (быстрее)")
    parser.add_argument("--angle-down", type=float, default=ANGLE_DOWN_THRESHOLD,
                        help=f"Порог угла «вниз» (по умолчанию {ANGLE_DOWN_THRESHOLD}°)")
    parser.add_argument("--angle-up",   type=float, default=ANGLE_UP_THRESHOLD,
                        help=f"Порог угла «вверх» (по умолчанию {ANGLE_UP_THRESHOLD}°)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    #Применяем пользовательские пороги
    ANGLE_DOWN_THRESHOLD = args.angle_down
    ANGLE_UP_THRESHOLD   = args.angle_up

    print(f"\n  Видео:   {args.video}")
    print(f"  Порог ↓: {ANGLE_DOWN_THRESHOLD}°   Порог ↑: {ANGLE_UP_THRESHOLD}°")
    if args.output:
        print(f"  Вывод:   {args.output}")
    print()

    events = process_video(
        video_path=args.video,
        output_path=args.output,
        show=not args.no_show,
    )

    print_report(events, args.video)
