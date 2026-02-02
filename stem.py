import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np 
import os

images = [] #обрані зображення
paths = [] #шляхи до зображень
current_index = 0
image_label = None   #мітка для відображення зображення
last_processed_image = None #останнє показане зображення
ROOT_WIDTH = 1000 #розміри відображення
ROOT_HEIGHT = 600
IMAGE_DISPLAY_SIZE = 550 #фіксований розмір для відображення

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

def canny_filter():
    """
    Функція для фільтру Кенні
    """
    global images, current_index, last_processed_image
    if not images:
        return
    img = images[current_index].convert("RGB") #відкриття та конвертація у rgb
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
    save_image(img_edges, paths[current_index], suffix="_canny") #збереження обробленого зображення
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_canny" + os.path.splitext(paths[current_index])[1]) #обчислення спотворення

def directional():
    """
    Напрямний фільтр - по куту 135
    """
    global images, current_index
    if not images:
        return

    img = images[current_index].convert("L") #відкриття зображення у відтінках сірого 
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
    save_image(img_result, paths[current_index], suffix="_directional") #збереження з суфіксом
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_directional" + os.path.splitext(paths[current_index])[1]) #обчислення спотворення

def pruitt():
    """
    Фільтр Прюітта
    """
    global images, current_index
    if not images:
        return

    img = images[current_index].convert("L") #відкриття у відтінках сірого 
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
    save_image(img_result, paths[current_index], suffix="_pruitt") #збереження з суфіксом
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_pruitt" + os.path.splitext(paths[current_index])[1]) #розрахунок спотворення

def white_balance():
    """
    Баланс білого 
    """
    global images, current_index
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
    save_image(img_result, paths[current_index], suffix="_white_balance") #збереження з суфіксом
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_white_balance" + os.path.splitext(paths[current_index])[1]) #обрахунок спотворення

def tracehold_segmentation(tracehold):
    """
    Колірна сегментація
    """
    global images, current_index
    if not images:
        return

    img = images[current_index]
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
    save_image(img_result, paths[current_index], suffix="_segmentation") #збееження з суфіксом
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_segmentation" + os.path.splitext(paths[current_index])[1]) #обрахунок спотворення

def histogram_equalization():
    """
    Гістограмне вирівнювання
    """
    global images, current_index
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
    save_image(img_result, paths[current_index], suffix="_histogram") #збееження з суфіксом
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_histogram" + os.path.splitext(paths[current_index])[1]) #обчислення спотворення

def harris_angle_detectors(threshold):
    """
    Функція виділення кутів Гарісса
    """
    global images, current_index
    if not images:
        return

    img = images[current_index].convert("L")  #зображення у відтінках сірого
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
    save_image(img_result, paths[current_index], suffix="_harris")  #збереження з суфіксом
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_harris" + os.path.splitext(paths[current_index])[1])  #спотворення

def fast(t):
    """
    Виділення ключових точок
    """
    global images, current_index
    if not images:
        return

    img = images[current_index].convert("L") #зображення у відтінках сірого
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
    save_image(img_result, paths[current_index], suffix="_fast") #збереження з суфіксом
    distortion_rate(paths[current_index], os.path.splitext(paths[current_index])[0] + "_fast" + os.path.splitext(paths[current_index])[1]) #обчислення спотворення

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

    print(f"Distortion rate: {mse}") #відображення у консолі

def interface(): 
    """
    Інтерфейс користувача
    """
    global root, image_label 
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

    tk.Button(right_frame, text="Canny Edge Detector", width=20, command=canny_filter).pack(pady=15) #кнопка для фільтра Кенніз командою 
    tk.Button(right_frame, text="Directional Filter", width=15, command=directional).pack(pady=15) #кнопка для напрямного фільтрування з командою
    tk.Button(right_frame, text="Pruitt Filter", width=15, command=pruitt).pack(pady=15) #кнопка для фільтру Прюітта з командою
    tk.Button(right_frame, text="White Ballance", width=15, command=white_balance).pack(pady=15) #кнопка для балансу білого з командою

    tk.Label(right_frame, text="Tracehold Value", bg="#e9e9e9").pack() #місце для введення значення для сегментації
    entry_tracehold = tk.Entry(right_frame) #зчитування значення
    entry_tracehold.pack()
    entry_tracehold.insert(0, "200")
    def apply_tracehold(): #передача параметру 
        threshold = int(entry_tracehold.get())
        tracehold_segmentation(threshold)
    tk.Button(right_frame, text="Tracehold segmentation", width=25, command=apply_tracehold).pack(pady=15) #кнопка для колірної сегментації з командо пісдя застосування сегментації

    tk.Button(right_frame, text="Histogram Equalization", width=20, command=histogram_equalization).pack(pady=15) #кнопка для гістагармного вирівнювання з командою

    tk.Label(right_frame, text="Harris angle detectors tracehold Value", bg="#e9e9e9").pack() #місце для введення порогового значення для виділення кутів
    harris_tracehold = tk.Entry(right_frame) #зчитування параметру
    harris_tracehold.pack()
    harris_tracehold.insert(0, "0.01")
    def apply_harris_threshold(): #застосування параметру для функції
        harris_threshold = float(harris_tracehold.get())
        harris_angle_detectors(harris_threshold)
    tk.Button(right_frame, text="Harris angle detectors", width=20, command=apply_harris_threshold).pack(pady=15) #кнопка для виділення кутів з командою пвсля застосування параметру

    tk.Label(right_frame, text="Brightness threshold", bg="#e9e9e9").pack() #місце для введення порогу яскравості для функції виділення ключових точок
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