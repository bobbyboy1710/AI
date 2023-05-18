import cv2
import numpy as np

# Завантаження зображення
image = cv2.imread('coins_2.JPG')

# Перетворення зображення на відтінки сірого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Виконання порогової сегментації
_, threshold = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Застосування морфологічних операцій для покращення результату
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Знаходження контурів монет
contours, _ = cv2.findContours(
    sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Задання кольорів для монет
# Приклад кольорів (червоний, зелений, синій)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

# Проходження по контурам і розмальовування монет
for i, contour in enumerate(contours):
    color = colors[i % len(colors)]  # Вибір кольору для монети
    cv2.drawContours(image, [contour], 0, color, 2)

# Відображення результату
cv2.imshow('Segmented Coins', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
