import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np 
import os
from scipy.ndimage import distance_transform_edt
import cv2
from sklearn.metrics import roc_auc_score

images = [] #обрані зображення
paths = [] #шляхи до зображень
current_index = 0
image_label = None   #мітка для відображення зображення
last_processed_image = None #останнє показане зображення
ROOT_WIDTH = 1000 #розміри відображення
ROOT_HEIGHT = 600
IMAGE_DISPLAY_SIZE = 550 #фіксований розмір для відображення
preprocess_combobox = None  # Глобальна змінна для combobox препроцесингу
metrics_label = None
ground_truth_path = ""

def load_images():
    """
    Функція для завантаження бази зображень для роботи
    """
    global images, current_index, image_label, paths #глобальне використання змінних - не тільки читання, а й зміна
    paths = filedialog.askopenfilenames(title="Choose an image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp;*.gif;*.webp")]) #вибір зображень - збереження кортежом усіх

    images = [] #ініціалізація масиву
    for path in paths: #для кожного шляху у кортежі відкривається і показується зображення
        img = Image.open(path) 
        images.append(img)
    
    current_index = 0 #передача індексу
    show_image(images[current_index])

def show_image(img: Image.Image):
    """
    Функція для відображення картинки на інтерфейсі
    """
    global image_label
    if img is None:
        return
    
    img_resized = img.resize((IMAGE_DISPLAY_SIZE, IMAGE_DISPLAY_SIZE), Image.Resampling.LANCZOS) #зміна розміру під параметри вікна
    img_tk = ImageTk.PhotoImage(img_resized)  #перетворення у формат, з яким може працювати tkinter

    #відображення зображення і збереження його посилання
    image_label.config(image=img_tk) #перетворення у формат, з яким може працювати tkinter
    image_label.image = img_tk 

def next_image():
    """
    Функція для переходу до наступної картинки на інтерфейсі
    """
    global current_index
    if images:
        current_index = (current_index + 1) % len(images) #перехід до наступного індексу - з обробкою логіки останнього елемента і переходу до першого
        show_image(images[current_index]) #відображення

def prev_image():
    """
    Функція для повернення до попереднього зображення
    """
    global current_index
    if images:
        current_index = (current_index - 1) % len(images) #перехід до попереднього
        show_image(images[current_index]) #відображення

def save_image(img, original_path, suffix="_processed"):
    """
    Функція для збереження зображення у поточну папку
    """
    base, ext = os.path.splitext(original_path) #отримання поточного шляху
    save_path = f"{base}{suffix}{ext}" #створення новго шляху з суфіксом
    img.save(save_path) #збереження
    return save_path  # Повертаємо шлях для метрик

def update_metrics_display(mse, ssim=None, fom=None, psnr=None, roc=None):
    global metrics_label
    if not metrics_label:
        return

    text = f"MSE: {mse:.4f}\n"
    if ssim is not None:
        text += f"SSIM: {ssim:.4f} (0–1)\n"
    if fom is not None:
        text += f"FOM: {fom:.4f} (0–1)\n"
    if psnr is not None:
        text += f"PSNR: {psnr:.4f}\n"
    if roc is not None:
        text += f"ROC: {roc:.4f}"

    metrics_label.config(text=text)

def apply_preprocessing(original_img):
    global last_processed_image, preprocess_combobox, current_preprocess_suffix
    current_preprocess_suffix = ""  # Скидаємо перед кожним викликом

    if not preprocess_combobox:
        return original_img

    option = preprocess_combobox.get()
    if option == "Default":
        return original_img

    # Тимчасово підставляємо зображення
    original_current = images[current_index]
    original_last = last_processed_image
    images[current_index] = original_img.copy()

    if option == "White Balance":
        white_balance()
        current_preprocess_suffix = "_pre_white_balance"
    elif option == "Histogram Equalization":
        histogram_equalization()
        current_preprocess_suffix = "_pre_histogram"
    elif option == "Sharpen Filter":
        sharpen_filter()
        current_preprocess_suffix = "_pre_sharpen"
    elif option == "CLAHE":
        clahe()
        current_preprocess_suffix = "_pre_clahe"
    elif option == "Dark channel":
        dark_channel()
        current_preprocess_suffix = "_pre_dark_channel"

    # Беремо результат
    processed = last_processed_image if last_processed_image else images[current_index].copy()

    # Повертаємо стан
    images[current_index] = original_current
    last_processed_image = original_last

    if current_preprocess_suffix:
        pre_path = save_image(processed, paths[current_index], suffix=current_preprocess_suffix)

    return processed

def canny_filter():
    """
    Функція для фільтру Кенні
    """
    global images, current_index, last_processed_image, current_preprocess_suffix
    if not images:
        return

    img = apply_preprocessing(images[current_index])  # Отримуємо препроцесоване або оригінал

    img = img.convert("RGB") #відкриття та конвертація у rgb
    arr = np.array(img, dtype=float) #перетворення у масив

    #перетворення у відтінки сірого - перший етап фільтрації
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

    #гаусове згладжування 5x5 - другий етап
    gauss_kernel = np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2]
    ]) / 159.0 #маска гауса
    padded = np.pad(gray, 2, mode='edge') #рамка 
    blurred = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            blurred[i,j] = np.sum(gauss_kernel * padded[i:i+5, j:j+5]) #заміна пікселя середньозваженим з маски 5*5

    #градієнти Соболя - третій етап
    #масики для обчислень - по вертикалі та горизонталі
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    Ix = np.zeros_like(blurred) #масиви для результатів
    Iy = np.zeros_like(blurred)
    padded_blur = np.pad(blurred, 1, mode='edge')

    for i in range(blurred.shape[0]):
        for j in range(blurred.shape[1]):
            Ix[i,j] = np.sum(Kx * padded_blur[i:i+3, j:j+3]) #зміна яскравості у кожній точці
            Iy[i,j] = np.sum(Ky * padded_blur[i:i+3, j:j+3])

    #модуль та напрямок градієнта
    G = np.hypot(Ix, Iy) #наскільки різко змінюється яскравість
    theta = np.arctan2(Iy, Ix) * (180.0 / np.pi) #кут напрямку градієнта
    theta[theta < 0] += 180  #кути від 0 до 180 - перетворення у позитивні значення

    #Non-maximum suppression - четвертий етап
    #фільтрування лише найчіткіших країв - інші занулити
    nms = np.zeros_like(G) #масив для найсильніших
    for i in range(1, G.shape[0]-1):
        for j in range(1, G.shape[1]-1):
            angle = theta[i,j] #напрям краю 
            q = r = 255 #сусіди з обох боків
            #сусіди по напрямках
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = G[i, j+1]
                r = G[i, j-1]
            elif 22.5 <= angle < 67.5:
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            elif 67.5 <= angle < 112.5:
                q = G[i+1, j]
                r = G[i-1, j]
            elif 112.5 <= angle < 157.5:
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            #якщо поточний піксель не є найбільшим серед сусідів - занулити
            if (G[i,j] >= q) and (G[i,j] >= r):
                nms[i,j] = G[i,j]
            else:
                nms[i,j] = 0

    #подвійний поріг - п'ятий етап
    #2 пороги - сильні та слабкі пороги
    highThreshold = nms.max() * 0.2
    lowThreshold = highThreshold * 0.5

    res = np.zeros_like(nms)
    strong = 255 #сильні та слабкі краї
    weak = 50

    #індекси, які підпадають під умову сильних/слабких країв
    strong_i, strong_j = np.where(nms >= highThreshold)
    weak_i, weak_j = np.where((nms <= highThreshold) & (nms >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    #трасування слабких країв - шостий етап
    #слабкий піксель стає сильним біля справжнього сильно контуру
    for i in range(1, res.shape[0]-1):
        for j in range(1, res.shape[1]-1):
            if res[i,j] == weak:
                if ((res[i+1,j-1] == strong) or (res[i+1,j] == strong) or (res[i+1,j+1] == strong) or (res[i,j-1] == strong) or (res[i,j+1] == strong) or (res[i-1,j-1] == strong) or (res[i-1,j] == strong) or (res[i-1,j+1] == strong)):
                    res[i,j] = strong
                else:
                    res[i,j] = 0

    img_edges = Image.fromarray(res.astype(np.uint8)) #перетворення результату
    show_image(img_edges) #відображення на екрані
    last_processed_image = img_edges #збереження індекса
    filter_suffix = "_canny"
    full_suffix = current_preprocess_suffix + filter_suffix if current_preprocess_suffix else filter_suffix
    save_path = save_image(img_edges, paths[current_index], suffix=full_suffix)
    mse = distortion_rate(paths[current_index], save_path)
    ssim = ssim_rate(paths[current_index], save_path)
    fom = fom_rate(save_path)
    psnr = psnr_rate(mse)
    roc = auc_roc_rate(save_path)
    update_metrics_display(mse, ssim, fom, psnr, roc)

def directional():
    """
    Напрямний фільтр - по куту 135
    """
    global images, current_index, last_processed_image, current_preprocess_suffix
    if not images:
        return

    img = apply_preprocessing(images[current_index])  # Отримуємо препроцесоване або оригінал

    img = img.convert("L") #відкриття зображення у відтінках сірого 
    array = np.array(img, dtype=float) #перетворення у матрицю 
    height, width = array.shape #розміри масиву 
    result = np.zeros_like(array) #масив для результату

    k = np.array([
        [0, 1, 2],
        [-1, 0, 1],
        [-2, -1, 0]
    ]) #фільтр напрямку 135 градусів

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = array[i-1:i+2, j-1:j+2]  #маска 3x3
            value = np.sum(region * k) #згортка
            result[i, j] = value

    result = np.clip(result, 0, 255).astype(np.uint8) #обмеження результату
    img_result = Image.fromarray(result) 
    show_image(img_result) #відображення на екрані
    last_processed_image = img_result #збереження індекса
    filter_suffix = "_directional"
    full_suffix = current_preprocess_suffix + filter_suffix if current_preprocess_suffix else filter_suffix
    save_path = save_image(img_result, paths[current_index], suffix=full_suffix)
    mse = distortion_rate(paths[current_index], save_path)
    ssim = ssim_rate(paths[current_index], save_path)
    fom = fom_rate(save_path)
    psnr = psnr_rate(mse)
    roc = auc_roc_rate(save_path)
    update_metrics_display(mse, ssim, fom, psnr, roc)

def pruitt():
    """
    Фільтр Прюітта
    """
    global images, current_index, last_processed_image, current_preprocess_suffix
    if not images:
        return

    img = apply_preprocessing(images[current_index])  # Отримуємо препроцесоване або оригінал

    img = img.convert("L") #відкриття у відтінках сірого 
    array = np.array(img, dtype=float) 
    height, width = array.shape 
    result = np.zeros_like(array)

    g_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]) #ядро по горизонталі
    g_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ]) #ядро по вертикалі 
 
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = array[i-1:i+2, j-1:j+2]  #маска 3x3 
            Gx = np.sum(region * g_x) #згортка по кожному напрямку
            Gy = np.sum(region * g_y)
            result[i, j] = np.sqrt(Gx**2 + Gy**2) #комбінована величина градієнта

    result = np.clip(result, 0, 255).astype(np.uint8) #обмеження результату
    img_result = Image.fromarray(result)
    show_image(img_result) #відображення на екрані
    last_processed_image = img_result #збереження індекса
    filter_suffix = "_pruitt"
    full_suffix = current_preprocess_suffix + filter_suffix if current_preprocess_suffix else filter_suffix
    save_path = save_image(img_result, paths[current_index], suffix=full_suffix)
    mse = distortion_rate(paths[current_index], save_path)
    ssim = ssim_rate(paths[current_index], save_path)
    fom = fom_rate(save_path)
    psnr = psnr_rate(mse)
    roc = auc_roc_rate(save_path)
    update_metrics_display(mse, ssim, fom, psnr, roc)

def sharpen_filter():
    global images, current_index, last_processed_image
    if not images:
        return
    
    img = images[current_index] #поточне зображення 

    img = img.convert("L") #відкриття зображення у відтінках сірого 
    array = np.array(img, dtype=float) #перетворення у матрицю 
    height, width = array.shape #розміри масиву 
    result = np.zeros_like(array) #масив для результату

    k = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]) #фільтр 

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = array[i-1:i+2, j-1:j+2]  #маска 3x3
            value = np.sum(region * k) #згортка
            result[i, j] = value

    result = np.clip(result, 0, 255).astype(np.uint8) #обмеження результату
    img_result = Image.fromarray(result) 
    show_image(img_result) #відображення на екрані
    last_processed_image = img_result  # Додаємо це, щоб last_processed_image встановлювався

def clahe():
    global images, current_index, last_processed_image
    if not images:
        return
    
    img = images[current_index]
    array = np.array(img, dtype=np.uint8)
    img_bw = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    clahe_img = np.clip(clahe.apply(img_bw) + 30, 0, 255).astype(np.uint8)

    img_result = Image.fromarray(clahe_img) #формування назад у масив

    show_image(img_result) #відображення на екрані
    last_processed_image = img_result  # Додаємо це, щоб last_processed_image встановлювався

def dark_channel():
    """
    Реалізація Dark Channel Prior для усунення туману (dehazing).
    Працює на оригінальному розмірі, без resize.
    """
    global images, current_index, last_processed_image
    if not images:
        return
    
    img = images[current_index].convert("RGB")
    array = np.array(img, dtype=float) / 255.0  
    height, width, channels = array.shape

    patch_size = 15  # вікно 15×15
    half_patch = patch_size // 2  # 7
    omega = 0.95
    t0 = 0.1

    # Крок 1: Dark Channel
    dark = np.ones((height, width))  # ініціалізуємо максимумом
    for i in range(half_patch, height - half_patch):
        for j in range(half_patch, width - half_patch):
            patch = array[i-half_patch:i+half_patch+1, j-half_patch:j+half_patch+1, :]
            min_per_channel = np.min(patch, axis=(0,1))  # мінімум по кожному каналу в патчі
            dark[i, j] = np.min(min_per_channel)         # мінімум з трьох мінімумів

    # Крок 2: Оцінка повітряного світла A
    flat_dark = dark.flatten()
    flat_img = array.reshape(-1, 3)
    
    # Беремо 0.1% найяскравіших пікселів у dark channel
    num_pixels = flat_dark.size
    num_brightest = max(1, int(num_pixels * 0.001))  # 0.1%
    brightest_indices = np.argsort(flat_dark)[-num_brightest:]
    
    # Середній колір цих пікселів в оригінальному зображенні
    A = np.mean(flat_img[brightest_indices], axis=0)

    # Крок 3: Оцінка transmission map t(x)
    t = np.ones((height, width))
    for i in range(half_patch, height - half_patch):
        for j in range(half_patch, width - half_patch):
            patch = array[i-half_patch:i+half_patch+1, j-half_patch:j+half_patch+1, :]
            min_patch = np.min(patch / A, axis=2)  # мінімум по каналах після ділення на A
            t[i, j] = 1 - omega * np.min(min_patch)

    # Крок 4: Відновлення J(x)
    t = np.maximum(t, t0)  # нижня межа
    J = np.zeros_like(array)
    for c in range(3):
        J[:,:,c] = (array[:,:,c] - A[c]) / t + A[c]

    # Повертаємо в діапазон [0, 255]
    J = np.clip(J * 255, 0, 255).astype(np.uint8)

    img_result = Image.fromarray(J)
    show_image(img_result)
    last_processed_image = img_result
    save_image(img_result, paths[current_index], suffix="_dehazed")

def white_balance():
    """
    Баланс білого 
    """
    global images, current_index, last_processed_image
    if not images:
        return

    img = images[current_index] #поточне зображення 
    array = np.array(img, dtype=float)
    R, G, B = array[:,:,0], array[:,:,1], array[:,:,2] #канали з зображення

    #середні значення яскравості для кожного каналу
    r_mean = np.mean(R)
    g_mean = np.mean(G)
    b_mean = np.mean(B)

    #коефіцієнти вирівнювання
    kR = g_mean / r_mean
    kG = 1.0
    kB = g_mean / b_mean

    #застосовання корекції
    R = np.clip(R * kR, 0, 255)
    G = np.clip(G * kG, 0, 255)
    B = np.clip(B * kB, 0, 255)

    balanced = np.stack([R, G, B], axis=2).astype(np.uint8) #об'єднання каналів
    img_result = Image.fromarray(balanced) #формування назад у масив

    show_image(img_result) #відображення на екрані
    last_processed_image = img_result  # Додаємо це, щоб last_processed_image встановлювався

def tracehold_segmentation(tracehold):
    """
    Колірна сегментація
    """
    global images, current_index, last_processed_image, current_preprocess_suffix
    if not images:
        return

    img = apply_preprocessing(images[current_index])  # Отримуємо препроцесоване або оригінал

    array = np.array(img, dtype=float)
    height, width, _ = array.shape
    R, G, B = array[:,:,0], array[:,:,1], array[:,:,2] #канали з зображення
    result = np.zeros_like(array) #масив для результату

    for i in range(1, height- 1):
        for j in range (1, width - 1):
            if R[i, j] > tracehold and G[i, j] > tracehold and B[i, j] > tracehold: #якщо усі канали мають значення більше за поріг
                result[i, j] = [255, 255, 255] #піксель вважаю світлим
            else:
                result[i, j] = [0, 0, 0] #інакше - темним

    result = np.clip(result, 0, 255).astype(np.uint8) #обмеження результату
    img_result = Image.fromarray(result)
    show_image(img_result) #відображення на екрані
    last_processed_image = img_result #збереження індекса
    filter_suffix = "_tracehold"
    full_suffix = current_preprocess_suffix + filter_suffix if current_preprocess_suffix else filter_suffix
    save_path = save_image(img_result, paths[current_index], suffix=full_suffix)
    mse = distortion_rate(paths[current_index], save_path)
    ssim = ssim_rate(paths[current_index], save_path)
    fom = fom_rate(save_path)
    psnr = psnr_rate(mse)
    roc = auc_roc_rate(save_path)
    update_metrics_display(mse, ssim, fom, psnr, roc)

def histogram_equalization():
    """
    Гістограмне вирівнювання
    """
    global images, current_index, last_processed_image
    if not images:
        return

    img = images[current_index]
    array = np.array(img, dtype=float)
    height, width, _ = array.shape
    #відтінки сірого
    Y = 0.299 * array[:,:,0] + 0.587 * array[:,:,1] + 0.114 * array[:,:,2]

    hist, _ = np.histogram(Y, bins=256, range=(0, 255)) #створення гістограми
    p = hist / Y.size #ймовірність появи кожного рівня яскравості

    #кумулятивна сума
    cdf = np.zeros(256)
    cdf[0] = p[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + p[i]

    result = np.zeros_like(Y, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i,j] = round(cdf[int(Y[i,j])] * 255) #нова яскравість пікселя через масив

    img_result = Image.fromarray(result) #збереження у масив
    show_image(img_result) #відображення на екрані
    last_processed_image = img_result  # Додаємо це, щоб last_processed_image встановлювався

def harris_angle_detectors(threshold):
    """
    Функція виділення кутів Гарісса
    """
    global images, current_index, last_processed_image, current_preprocess_suffix
    if not images:
        return

    img = apply_preprocessing(images[current_index])  # Отримуємо препроцесоване або оригінал

    img = img.convert("L")  #зображення у відтінках сірого
    array = np.array(img, dtype=float)
    height, width = array.shape
    result = np.zeros_like(array)

    i_x = np.zeros_like(array) #горизонтальні градієнти - зміна по осі х
    i_y = np.zeros_like(array) #вертикальні
    R = np.zeros_like(array)  #карта Гарісса
    
    for i in range(1, height - 1):       
        for j in range(1, width - 1):   
            #градієнти яскравості 
            i_x[i, j] = array[i, j+1] - array[i, j]  #різниця між пікселем справа і поточним
            i_y[i, j] = array[i+1, j] - array[i, j]  #різниця між пікселем знизу і поточним

    for i in range(1, height - 1):
            for j in range(1, width - 1):
                #маска 3*3
                ix_block = i_x[i-1:i+2, j-1:j+2]
                iy_block = i_y[i-1:i+2, j-1:j+2]

                #компоненти матриці
                Sxx = np.sum(ix_block**2)
                Syy = np.sum(iy_block**2)
                Sxy = np.sum(ix_block * iy_block)

                #матриця 2*2
                M = np.array([[Sxx, Sxy],
                              [Sxy, Syy]])
                R[i, j] = np.linalg.det(M) - 0.04 * (Sxx + Syy)**2  #формула Гарісса

    t = threshold * np.max(R)  #визначення порогового значення
    for i in range(1, height - 1):
            for j in range(1, width - 1):
                if R[i, j] > t:  #якщо значення з масиву для пікселя перевищує поріг - це кут
                    result[i, j] = 255  #стає білим

    img_result = Image.fromarray(result.astype(np.uint8))  #конвертація 
    show_image(img_result)  #відображення на екані
    last_processed_image = img_result #збереження індекса
    filter_suffix = "_harris"
    full_suffix = current_preprocess_suffix + filter_suffix if current_preprocess_suffix else filter_suffix
    save_path = save_image(img_result, paths[current_index], suffix=full_suffix)
    mse = distortion_rate(paths[current_index], save_path)
    ssim = ssim_rate(paths[current_index], save_path)
    fom = fom_rate(save_path)
    psnr = psnr_rate(mse)
    roc = auc_roc_rate(save_path)
    update_metrics_display(mse, ssim, fom, psnr, roc)

def fast(t):
    """
    Виділення ключових точок
    """
    global images, current_index, last_processed_image, current_preprocess_suffix
    if not images:
        return

    img = apply_preprocessing(images[current_index])  # Отримуємо препроцесоване або оригінал

    img = img.convert("L") #зображення у відтінках сірого
    array = np.array(img, dtype=float)
    height, width = array.shape
    result = np.zeros_like(array) #масив на результат

    #координати 16 пікселів для кола радіусом 3 
    c = [(-3,0), (-3,1), (-2,2), (-1,3), (0,3), (1,3), (2,2), (3,1), (3,0), (3,-1), (2,-2), (1,-3), (0,-3), (-1,-3), (-2,-2), (-3,-1)]

    N = 12 #мінімальна кількість яскравих пікселів

    for i in range(3, height - 3):
            for j in range(3, width - 3):
                p = array[i,j] #інтенсивінсть центрального
                circle = [array[i+di, j+dj] for (di,dj) in c] #масив точок по колу

                #поділ їх на масив яскравих і темних
                bright = [c > p + t for c in circle]
                dark   = [c < p - t for c in circle]

                #подвоєна сума, щоб повернутися у початкову точку
                bright2 = bright + bright
                dark2   = dark + dark

                keypoint = False #булева для позначки чи є піксель кутом
                count_bright = count_dark = 0 #лічильники послідовності

                for k in range(32):
                    if bright2[k]:
                        count_bright += 1
                        if count_bright >= N: #якщо накопичилось більше н яскравих
                            keypoint = True #це кут
                            break
                    else:
                        count_bright = 0

                    if dark2[k]:
                        count_dark += 1
                        if count_dark >= N:
                            keypoint = True
                            break
                    else:
                        count_dark = 0

                if keypoint: #якщо це кут - заміна його на біллий
                    result[i,j] = 255

    img_result = Image.fromarray(result.astype(np.uint8)) #конвертація результату
    show_image(img_result) #відображення на екрані
    last_processed_image = img_result #збереження індекса
    filter_suffix = "_fast"
    full_suffix = current_preprocess_suffix + filter_suffix if current_preprocess_suffix else filter_suffix
    save_path = save_image(img_result, paths[current_index], suffix=full_suffix)
    mse = distortion_rate(paths[current_index], save_path)
    ssim = ssim_rate(paths[current_index], save_path)
    fom = fom_rate(save_path)
    psnr = psnr_rate(mse)
    roc = auc_roc_rate(save_path)
    update_metrics_display(mse, ssim, fom, psnr, roc)

def distortion_rate(original_path, filtered_path):
    """
    Обчислення спотворення між оригіналом і зміненим зображенням
    """
    orig = Image.open(original_path).convert("RGB") #оригінал
    filt = Image.open(filtered_path).convert("RGB") #оброблене

    orig_array = np.array(orig, dtype=float)
    filt_array = np.array(filt, dtype=float)

    #середньоквадратична помилка 
    mse = np.mean((filt_array - orig_array) ** 2)

    return mse

def psnr_rate(mse):
    if mse == 0:
        psnr = float('inf')  # ідеальне зображення, немає помилок
    else:
        # Максимальне значення пікселя = 255
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)

    return psnr

def ssim_rate(original_path, filtered_path):
    """
    Обчислення SSIM за спрощеною формулою:
    Повертає значення SSIM (0..1, ближче до 1 — краще).
    """
    orig = Image.open(original_path).convert("L")  # grayscale оригінал
    filt = Image.open(filtered_path).convert("L")  # grayscale оброблене

    orig_array = np.array(orig, dtype=float)
    filt_array = np.array(filt, dtype=float)

    # Середні значення
    mu_x = np.mean(orig_array)
    mu_y = np.mean(filt_array)

    # Дисперсії
    sigma_x_sq = np.var(orig_array)  # ddof=0
    sigma_y_sq = np.var(filt_array)

    # Коваріація
    sigma_xy = np.mean((orig_array - mu_x) * (filt_array - mu_y))

    # Константи стабілізації (стандартні)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Захист від ділення на нуль
    epsilon = 1e-10

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)) + epsilon

    ssim_value = numerator / denominator

    # Обрізаємо негативні значення (рідко, але буває)
    ssim_value = max(ssim_value, 0.0)

    return ssim_value

def mask_to_edge_array(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask)

    mask_bin = mask_arr > 127

    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(mask_bin.astype(np.uint8)*255, kernel)
    edges = (dilated > 127) & (~mask_bin)

    return edges


def fom_rate(detected_path, alpha=1/9):
    global ground_truth_path

    if not ground_truth_path:
        return None

    detected = Image.open(detected_path).convert("L")
    detected_arr = np.array(detected) > 127

    # отримуємо edges з маски
    gt_edges = mask_to_edge_array(ground_truth_path)

    N_d = np.sum(detected_arr)
    N_g = np.sum(gt_edges)

    if N_d == 0 or N_g == 0:
        return 0.0

    dist_map = distance_transform_edt(~gt_edges)
    detected_distances = dist_map[detected_arr]

    penalties = 1 / (1 + alpha * detected_distances ** 2)
    fom = np.sum(penalties) / max(N_d, N_g)

    return fom

def auc_roc_rate(detected_raw_path):
    global ground_truth_path, metrics_label  # використовуємо глобальну змінну, як у тебе
    
    if not ground_truth_path:
        return None
        
    detected_raw = Image.open(detected_raw_path).convert("L")
    detected_arr = np.array(detected_raw, dtype=float)

        # Завантажуємо ground truth (еталон)
    gt = Image.open(ground_truth_path).convert("L")
    gt_arr = np.array(gt) > 127  # бінарна маска: True — позитивний клас (край/точка)


        # Перетворюємо в 1D масиви для roc_auc_score
    y_true = gt_arr.flatten().astype(int)      # 0/1
    y_score = detected_arr.flatten()           # "ймовірності" (чим вище — тим ймовірніше позитив)

        # Обчислюємо AUC-ROC
    auc = roc_auc_score(y_true, y_score)

    return auc

def interface(): 
    """
    Інтерфейс користувача
    """
    global root, image_label, preprocess_combobox 
    root = tk.Tk() 
    root.title('STEM: Image Processing') #заголовок вікна
    root.geometry(f"{ROOT_WIDTH}x{ROOT_HEIGHT}") #розміри вікна
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT) 
    
    btn_load = tk.Button(text="Load set of images", command=load_images, font=("Arial", 12, "bold"))  #кнопка завантаження зображень з відповідною командою
    btn_load.pack(side="top", pady=5)

    left_frame = tk.Frame(width=650, bg="#ffffff") 
    left_frame.pack(side="left", fill="y", padx=5, pady=5) 
    left_frame.pack_propagate(False) 

    #місце для відображення зображень
    image_label = tk.Label(left_frame, bg="#333333", fg="white", width=IMAGE_DISPLAY_SIZE, height=IMAGE_DISPLAY_SIZE)
    image_label.pack(side="top", padx=10, pady=10)

    #місце для кнопок
    nav_frame = tk.Frame(left_frame, bg="#ffffff")
    nav_frame.pack(side="bottom", pady=10)

    #кнопки для переходу до наступного / попереднього зображення
    tk.Button(nav_frame, text="Previous", command=prev_image, width=15).pack(side="left", padx=10)
    tk.Button(nav_frame, text="Next", command=next_image, width=15).pack(side="left", padx=10)

    #бокова панель для кнопок всіх фільтрів
    right_frame = tk.Frame(width=330, bg="#e9e9e9", relief=tk.RIDGE, bd=2)
    right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    right_frame.pack_propagate(False)
    tk.Label(right_frame, text="Filter Panel", bg="#e9e9e9",font=("Arial", 14, "bold")).pack(pady=10)

    #панель 
    preprocess_frame = tk.Frame(width=330, bg="#e9e9e9", relief=tk.RIDGE, bd=2)
    preprocess_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    preprocess_frame.pack_propagate(False)
    tk.Label(preprocess_frame, text="Preproccesing Panel", bg="#e9e9e9",font=("Arial", 14, "bold")).pack(pady=10)

    preprocess_combobox = ttk.Combobox(preprocess_frame, values=["Default", "White Balance", "Histogram Equalization", "Sharpen Filter", "CLAHE", "Dark channel"], width=30)
    preprocess_combobox.pack()
    preprocess_combobox.set("Default")

    global metrics_label
    metrics_label = tk.Label(
        preprocess_frame,
        text="MSE:",
        bg="#e9e9e9",
        fg="#333333",
        font=("Arial", 10),
        justify="left",
        wraplength=300
    )
    metrics_label.pack(side="bottom", pady=20, padx=10, fill="x")

    tk.Label(preprocess_frame, text="Ground Truth FOM:", bg="#e9e9e9").pack(pady=15)
    gt_entry = tk.Entry(preprocess_frame, width=40)
    gt_entry.pack(pady=5)
    gt_entry.insert(0, ground_truth_path)  
    def choose_ground_truth():
        global ground_truth_path
        path = filedialog.askopenfilename(title="Choose ground truth",filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")])
        if path:
            ground_truth_path = path
            gt_entry.delete(0, tk.END)
            gt_entry.insert(0, path)
            print(f"Selected Ground truth: {path}")

    tk.Button(preprocess_frame, text="Обрати файл", command=choose_ground_truth).pack(pady=5)

    tk.Button(right_frame, text="Canny Edge Detector", width=20, command=canny_filter).pack(pady=15) #кнопка для фільтра Кенніз командою 
    tk.Button(right_frame, text="Directional Filter", width=15, command=directional).pack(pady=15) #кнопка для напрямного фільтрування з командою
    tk.Button(right_frame, text="Pruitt Filter", width=15, command=pruitt).pack(pady=15) #кнопка для фільтру Прюітта з командою

    tk.Label(right_frame, text="Tracehold Value", bg="#e9e9e9").pack() #місце для введення значення для сегментації
    entry_tracehold = tk.Entry(right_frame) #зчитування значення
    entry_tracehold.pack()
    entry_tracehold.insert(0, "200")
    def apply_tracehold(): #передача параметру 
        threshold = int(entry_tracehold.get())
        tracehold_segmentation(threshold)
    tk.Button(right_frame, text="Tracehold segmentation", width=25, command=apply_tracehold).pack(pady=15) #кнопка для колірної сегментації з командо пісдя застосування сегментації

    tk.Label(right_frame, text="Harris angle detectors tracehold Value", bg="#e9e9e9").pack() #місце для введення порогового значення для виділення кутів
    harris_tracehold = tk.Entry(right_frame) #зчитування параметру
    harris_tracehold.pack()
    harris_tracehold.insert(0, "0.01")
    def apply_harris_threshold(): #застосування параметру для функції
        harris_threshold = float(harris_tracehold.get())
        harris_angle_detectors(harris_threshold)
    tk.Button(right_frame, text="Harris angle detectors", width=20, command=apply_harris_threshold).pack(pady=15) #кнопка для виділення кутів з командою пвсля застосування параметру

    tk.Label(right_frame, text=" Brightness threshold", bg="#e9e9e9").pack() #місце для введення порогу яскравості для функції виділення ключових точок
    brightness_threshold = tk.Entry(right_frame) #зчитування параметру
    brightness_threshold.pack()
    brightness_threshold.insert(0, "20")
    def apply_brightness_threshold(): #застосування параметру
        brightness_threshold_value = int(brightness_threshold.get())
        fast(brightness_threshold_value)
    tk.Button(right_frame, text="Features from Accelerated Segment Test", width=35, command=apply_brightness_threshold).pack(pady=15) #кнопка для виділення ключових точок з командою після застосування параметру

    root.mainloop() #відображення вікна
    
def main():
    interface()

if __name__ == "__main__":
    main()