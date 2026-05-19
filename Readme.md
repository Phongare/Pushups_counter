# PYTHON 3.10 
# MEDIAPIPE 0.10.14
# ПУТИ К ВИДЕО
     VIDEO_DIR      = "/home/user/videos" - Папка к видео        
                    = r"C:\Users\user\Videos" - Виндовс, выше Мак и Линукс
     VIDEO_FILENAME      = "video.mp4" - Название видео     
     OUTPUT_DIR      = "/home/outputs" - Папка с обработанными видео                   
     OUTPUT_FILENAME = "workout.mp4" - Обр. видео 
     VIDEO_DIR = os.path.dirname(os.path.abspath(__file__)) - папка рядом со скриптом
     VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads") - подпапка относительно скрипта
# ЗАПУСК С ТЕРМИНАЛА
     python pushup_counter.py (если пути заданы)
     python pushup_counter.py --video /tmp/user123/clip.mp4 --output result.mp4
# ПЕРЕМЕННАЯ КОЛ-ВА ОТЖИМАНИЙ
    pushups_total или же len(events)
