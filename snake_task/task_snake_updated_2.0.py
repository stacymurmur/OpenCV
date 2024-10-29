import numpy as np
import cv2
import random

fontFace = cv2.FONT_HERSHEY_DUPLEX
height, width = 400, 400
size = 20
speed = size
score = 0


background = cv2.imread('background.jpg')
background = cv2.resize(background, (width, height))

apple = cv2.imread('apple_nobg.png', cv2.IMREAD_UNCHANGED)
apple = cv2.resize(apple, (size, size))

# обработка альфа-канала(мы его достаем и нормируем относиельно цвета)
if apple.shape[2] == 4:  
    bgr = apple[:, :, :3]
    alpha = apple[:, :, 3]
    alpha_mask = alpha / 256.0
else:
    bgr = apple
    alpha_mask = np.ones((size, size), dtype=np.float32)

snake = [(100, 100), (80, 100), (60, 100)]
direction = (speed, 0)
food = (random.randint(0, (width // size) - 1) * size, random.randint(0, (height // size) - 1) * size)

#рисую змейку, яблоко и наложение на него альфа-маски
def draw_objects(field):
    head = snake[0]
    cv2.circle(field, (head[0] + size // 2, head[1] + size // 2), size // 2, (255, 255, 0), thickness=-1)

    for idx, segment in enumerate(snake[1:], start=0):
        reduction_factor = 1 - (idx / len(snake) * 0.2)
        segment_size = int(size * reduction_factor)
        if segment_size < 5:
            segment_size = 5

        x, y = segment
        offset = (size - segment_size) // 2

        cv2.rectangle(field,
                      (x , y),
                      (x + offset + segment_size, y + segment_size),
                      (256, 0, 0), thickness=-1)

    x, y = food
    for i in range(size):
        for j in range(size):
            if 0 <= y + i < height and 0 <= x + j < width:
                for c in range(3):
                    field[y + i, x + j, c] = alpha_mask[i, j] * bgr[i, j, c] + (1 - alpha_mask[i, j]) * field[
                        y + i, x + j, c]


#движение змейки, проверка граничного случая, когда она выходит за экран и когда ест яблоко
def move_snake():
    global food, score
    new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

    if new_head[0] < 0:
        new_head = (width - size, new_head[1])
    elif new_head[0] >= width:
        new_head = (0, new_head[1])
    elif new_head[1] < 0:
        new_head = (new_head[0], height - size)
    elif new_head[1] >= height:
        new_head = (new_head[0], 0)

    snake.insert(0, new_head)


    if snake[0] == food:
        score += 1
        food = (random.randint(0, (width // size) - 1) * size,
                random.randint(0, (height // size) - 1) * size)
    else:
        snake.pop()

def check_snake():
    head = snake[0]
    if head in snake[1:]:
        return True
    return False

cv2.namedWindow('Snake')

#Основной игровой цикл (q - выход из игры, управление клавишами w,a,s,d на английской раскладке)
while True:
    field = background.copy()
    draw_objects(field)

    cv2.putText(field, "Snake game", (width // 4, 50), fontFace, 1, (255, 255, 255), 2)
    cv2.putText(field, f"Score: {score}", (width - 150, height - 20), fontFace, 0.8, (255, 255, 255), 2)

    cv2.imshow('Snake', field)

    key = cv2.waitKey(200)

    if key == ord('w') and direction != (0, speed):
        direction = (0, -speed)
    elif key == ord('s') and direction != (0, -speed):
        direction = (0, speed)
    elif key == ord('a') and direction != (speed, 0):
        direction = (-speed, 0)
    elif key == ord('d') and direction != (-speed, 0):
        direction = (speed, 0)

    move_snake()

    if check_snake():
        img = np.zeros((200, 400))
        cv2.putText(img, "Game over!", (width // 4, 100), fontFace, 1, (255, 255 , 0), 2)
        cv2.imshow("Window", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Window")
        print("Game over!")
        break

    if key == ord('q'):
        break

cv2.destroyAllWindows()
