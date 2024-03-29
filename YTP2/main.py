import cv2
import numpy as np

# Загрузка видео
video_capture = cv2.VideoCapture('video_preview.mp4')

# Инициализация метода выделения фона
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Инициализация параметров для подсчета автомобилей
min_contour_width = 20
min_contour_height = 20
offset = 2
crossing_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
matches = []
with_traffic_count = 0
against_traffic_count = 0

while True:
    # Чтение следующего кадра
    ret, frame = video_capture.read()

    if not ret:
        break  # Прервать цикл при достижении конца видео

    # Применение метода выделения фона
    fg_mask = background_subtractor.apply(frame)

    # Удаление шумов с помощью морфологических операций
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Нахождение контуров объектов
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка прямоугольников вокруг движущихся объектов и подсчет автомобилей, пересекших середину
    for contour in contours:
        if cv2.contourArea(contour) > 275:  # Фильтрация маленьких объектов
            x, y, w, h = cv2.boundingRect(contour)

            # Регулировка параметров
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 5:  # Пример фильтрации по соотношению сторон
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Подсчет количества автомобилей, пересекших середину изображения
                if x < crossing_width + offset and x + w > crossing_width - offset:
                    with_traffic_count += 1
                elif x > crossing_width:
                    against_traffic_count += 1

    # Добавление текстовой метки с количеством автомобилей, пересекших середину
    cv2.putText(frame, f'With Traffic: {with_traffic_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Against Traffic: {against_traffic_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Показ кадра с выделенными объектами и счетчиком
    cv2.imshow('Motion Detection', frame)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
