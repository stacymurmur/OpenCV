import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider


img = cv.imread('7687.jpg', cv.IMREAD_COLOR)

def compute_sobel_gradient(image):
    # Применяем градиент Собеля по обоим направлениям и далее считаем сумму абсолютных значений
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    sobel_sum = np.mean(np.abs(sobelx)) + np.mean(np.abs(sobely))
    return sobel_sum

def optimize_sharpening_params(img, alpha_range, beta_range, step):
    max_sobel_sum = 0
    best_params = (0, 0)
    best_image = None

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray_img, (3, 3), 0)

    param_values = []
    sobel_sums = []

    for alpha in np.arange(*alpha_range, step):
        for beta in np.arange(*beta_range, step):
            sharp = cv.addWeighted(gray_img, alpha, blurred, beta, 0)

            sobel_sum = compute_sobel_gradient(sharp)

            param_values.append((alpha, beta))
            sobel_sums.append(sobel_sum)

            if sobel_sum > max_sobel_sum:
                max_sobel_sum = sobel_sum
                best_params = (alpha, beta)
                best_image = sharp

    return best_params, best_image, max_sobel_sum, param_values, sobel_sums

# Определяем диапазоны
alpha_range = (1.0, 2.5)
beta_range = (-1.5, -0.1)
step = 0.1

best_params, best_image, max_sobel_sum, param_values, sobel_sums = optimize_sharpening_params(img, alpha_range, beta_range, step)

print(f"Лучшие параметры: alpha = {best_params[0]:.2f}, beta = {best_params[1]:.2f}")
print(f"Максимальная сумма градиентов Собеля: {max_sobel_sum:.2f}")

normalized_img = best_image / 255.0  # Переводим значения в диапазон [0, 1]


fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)


image_display = ax.imshow(normalized_img, vmin=0, vmax=1)
ax.axis('off')

# Добавление ползунка для интенсивности
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, "Intensity", 0.5, 2.0, valinit=1.0)

# Функция для обновления изображения
def update(val):
    intensity = slider.val
    adjusted_img = np.clip(normalized_img * intensity, 0, 1)
    image_display.set_data(adjusted_img)
    fig.canvas.draw_idle()


alpha_vals = [params[0] for params in param_values]
beta_vals = [params[1] for params in param_values]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(alpha_vals, beta_vals, sobel_sums, c=sobel_sums, cmap='viridis', s=50)
ax.scatter(best_params[0], best_params[1], max_sobel_sum, color='red', s=100, edgecolors='black', label=f'Best: alpha={best_params[0]}, beta={best_params[1]}')
fig.colorbar(sc, label='Сумма градиентов Собеля')
ax.set_xlabel('alpha')
ax.set_ylabel('beta')
ax.set_zlabel('Сумма градиентов')
ax.set_title('3D график зависимости суммы градиентов от параметров alpha и beta')
ax.view_init(elev=20., azim=-60)
ax.legend()


slider.on_changed(update)
plt.show()
