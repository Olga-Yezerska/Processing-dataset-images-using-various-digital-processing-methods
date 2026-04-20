import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import os
from scipy.ndimage import distance_transform_edt
import cv2
from sklearn.metrics import roc_auc_score
from abc import ABC, abstractmethod
from typing import Optional

def save_image(img: Image.Image, original_path: str, suffix: str = "_processed") -> str:
    """
    Метод збереження зображень
    """
    base, ext = os.path.splitext(original_path)
    save_path = f"{base}{suffix}{ext}"
    img.save(save_path)
    return save_path

class PreprocessingStrategy(ABC):
    """
    Клас стратегії для методів попередньої обробки
    """
    @abstractmethod
    def apply(self, img: Image.Image) -> Image.Image:
        """
        Реалізація стратегії виконання методу
        """
        pass
    @property
    def suffix(self) -> str:
        """
        Додавання суфікусу відповдіно до методу
        """
        return ""

class DefaultPreprocessing(PreprocessingStrategy):
    """
    Конкретна реалізація класу для методу без обробки - дефолт
    """
    def apply(self, img: Image.Image) -> Image.Image:
        return img.copy()

class WhiteBalancePreprocessing(PreprocessingStrategy):
    """
    Клас конкетної реалізації стратегії балансу білого
    """
    @property
    def suffix(self) -> str:
        return "_pre_white_balance"

    def apply(self, img: Image.Image) -> Image.Image:
        array = np.array(img, dtype=float)
        R, G, B = array[:,:,0], array[:,:,1], array[:,:,2]

        r_mean = np.mean(R)
        g_mean = np.mean(G)
        b_mean = np.mean(B)

        kR = g_mean / r_mean if r_mean != 0 else 1
        kB = g_mean / b_mean if b_mean != 0 else 1

        R = np.clip(R * kR, 0, 255)
        G = np.clip(G, 0, 255)
        B = np.clip(B * kB, 0, 255)

        balanced = np.stack([R, G, B], axis=2).astype(np.uint8)

        return Image.fromarray(balanced)

class HistogramEqualizationPreprocessing(PreprocessingStrategy):
    """
    Клас конкетної реалізації стратегії гістограмного вирівнювання
    """
    @property
    def suffix(self) -> str:
        return "_pre_histogram"

    def apply(self, img: Image.Image) -> Image.Image:
        array = np.array(img, dtype=float)
        height, width, _ = array.shape

        Y = 0.299 * array[:,:,0] + 0.587 * array[:,:,1] + 0.114 * array[:,:,2]
        hist, _ = np.histogram(Y, bins=256, range=(0, 255))

        p = hist / Y.size

        cdf = np.zeros(256)
        cdf[0] = p[0]

        for i in range(1, 256):
            cdf[i] = cdf[i-1] + p[i]
        result = np.zeros_like(Y, dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                result[i,j] = round(cdf[int(Y[i,j])] * 255)

        return Image.fromarray(result)

class SharpenPreprocessing(PreprocessingStrategy):
    """
    Клас конкетної реалізації стратегії фільтру "гостроти"
    """
    @property
    def suffix(self) -> str:
        return "_pre_sharpen"

    def apply(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")

        array = np.array(img, dtype=float)
        height, width = array.shape
        result = np.zeros_like(array)

        k = np.array([
            [0, -1, 0], 
            [-1, 5, -1], 
            [0, -1, 0]
        ])

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = array[i-1:i+2, j-1:j+2]
                result[i, j] = np.sum(region * k)

        result = np.clip(result, 0, 255).astype(np.uint8)

        return Image.fromarray(result)

class CLAHEPreprocessing(PreprocessingStrategy):
    """
    Клас конкетної реалізації стратегії CLAHE
    """
    @property
    def suffix(self) -> str:
        return "_pre_clahe"

    def apply(self, img: Image.Image) -> Image.Image:
        array = np.array(img, dtype=np.uint8)
        img_bw = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=5)
        clahe_img = np.clip(clahe.apply(img_bw) + 30, 0, 255).astype(np.uint8)

        return Image.fromarray(clahe_img)

class DarkChannelPreprocessing(PreprocessingStrategy):
    """
    Клас конкетної реалізації стратегії dark channel фільтру 
    """
    @property
    def suffix(self) -> str:
        return "_pre_dark_channel"

    def apply(self, img: Image.Image) -> Image.Image:
        array = np.array(img.convert("RGB"), dtype=float) / 255.0
        height, width, _ = array.shape

        patch_size = 15
        half_patch = patch_size // 2
        omega = 0.95
        t0 = 0.1

        dark = np.ones((height, width))
        for i in range(half_patch, height - half_patch):
            for j in range(half_patch, width - half_patch):
                patch = array[i-half_patch:i+half_patch+1, j-half_patch:j+half_patch+1, :]
                dark[i, j] = np.min(np.min(patch, axis=(0,1)))

        flat_dark = dark.flatten()
        flat_img = array.reshape(-1, 3)

        num_brightest = max(1, int(flat_dark.size * 0.001))
        brightest_indices = np.argsort(flat_dark)[-num_brightest:]
        A = np.mean(flat_img[brightest_indices], axis=0)

        t = np.ones((height, width))
        for i in range(half_patch, height - half_patch):
            for j in range(half_patch, width - half_patch):
                patch = array[i-half_patch:i+half_patch+1, j-half_patch:j+half_patch+1, :]
                min_patch = np.min(patch / A, axis=2)
                t[i, j] = 1 - omega * np.min(min_patch)

        t = np.maximum(t, t0)
        J = np.zeros_like(array)
        for c in range(3):
            J[:,:,c] = (array[:,:,c] - A[c]) / t + A[c]

        J = np.clip(J * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(J)

class FilterStrategy(ABC):
    """
    Клас стратегії для методів комп'ютерного зору
    """
    @abstractmethod
    def apply(self, img: Image.Image) -> Image.Image:
        pass

    @property
    def suffix(self) -> str:
        return ""

class CannyFilter(FilterStrategy):
    """
    Клас конкетної реалізації стратегії фільтру Кенні
    """
    @property
    def suffix(self) -> str:
        return "_canny"
    
    def _connect_endpoints(self, edge_map: np.ndarray) -> np.ndarray:
        """
        Пост-обробка: знаходить кінцеві точки розірваних країв і з'єднує найближчі пари
        """
        edges = edge_map.copy().astype(np.uint8)
        edge_points = np.argwhere(edges > 127)
        if len(edge_points) < 2:
            return edges
        endpoints = []
        for y, x in edge_points:
            # 8-сусідів
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < edges.shape[0] and 0 <= nx < edges.shape[1]:
                        if edges[ny, nx] > 127:
                            neighbors += 1
            if neighbors <= 1:          
                endpoints.append((y, x))

        if len(endpoints) < 2:
            return edges

        endpoints = np.array(endpoints)
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(endpoints, endpoints)

        paired = set()
        for i in range(len(endpoints)):
            if i in paired:
                continue
            min_dist = float('inf')
            best_j = -1
            for j in range(len(endpoints)):
                if i != j and j not in paired:
                    if dist_matrix[i, j] < min_dist and dist_matrix[i, j] < 30:  
                        min_dist = dist_matrix[i, j]
                        best_j = j
            if best_j != -1:
                y1, x1 = endpoints[i]
                y2, x2 = endpoints[best_j]
                cv2.line(edges, (x1, y1), (x2, y2), 255, thickness=1)
                paired.add(i)
                paired.add(best_j)

        return edges

    def apply(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        arr = np.array(img, dtype=float)
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

        gauss_kernel = np.array([
            [2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]
        ]) / 159.0
        padded = np.pad(gray, 2, mode='edge')
        blurred = np.zeros_like(gray)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                blurred[i,j] = np.sum(gauss_kernel * padded[i:i+5, j:j+5])

        Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        Ix = np.zeros_like(blurred)
        Iy = np.zeros_like(blurred)
        padded_blur = np.pad(blurred, 1, mode='edge')
        for i in range(blurred.shape[0]):
            for j in range(blurred.shape[1]):
                Ix[i,j] = np.sum(Kx * padded_blur[i:i+3, j:j+3])
                Iy[i,j] = np.sum(Ky * padded_blur[i:i+3, j:j+3])

        G = np.hypot(Ix, Iy)
        theta = np.arctan2(Iy, Ix) * (180.0 / np.pi)
        theta[theta < 0] += 180

        nms = np.zeros_like(G)
        for i in range(1, G.shape[0]-1):
            for j in range(1, G.shape[1]-1):
                angle = theta[i,j]
                q = r = 255
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
                if (G[i,j] >= q) and (G[i,j] >= r):
                    nms[i,j] = G[i,j]

        highThreshold = nms.max() * 0.2
        lowThreshold = highThreshold * 0.5
        res = np.zeros_like(nms)
        strong = 255
        weak = 50
        strong_i, strong_j = np.where(nms >= highThreshold)
        weak_i, weak_j = np.where((nms <= highThreshold) & (nms >= lowThreshold))
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        for i in range(1, res.shape[0]-1):
            for j in range(1, res.shape[1]-1):
                if res[i,j] == weak:
                    neighbors = [
                        res[i+1,j-1], res[i+1,j], res[i+1,j+1],
                        res[i,j-1],               res[i,j+1],
                        res[i-1,j-1], res[i-1,j], res[i-1,j+1]
                    ]
                    if strong in neighbors:
                        res[i,j] = strong
                    else:
                        res[i,j] = 0

        res = self._connect_endpoints(res)

        return Image.fromarray(res.astype(np.uint8))

class DirectionalFilter(FilterStrategy):
    """
    Клас конкетної реалізації стратегії напрямного фільтру 135 градусів
    """
    @property
    def suffix(self) -> str:
        return "_directional"

    def apply(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        array = np.array(img, dtype=float)
        height, width = array.shape
        result = np.zeros_like(array)

        k = np.array([
            [0,1,2],
            [-1,0,1],
            [-2,-1,0]
        ])

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                result[i, j] = np.sum(array[i-1:i+2, j-1:j+2] * k)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

class PruittFilter(FilterStrategy):
    """
    Клас конкетної реалізації стратегії фільтру Прьюітта
    """
    @property
    def suffix(self) -> str:
        return "_pruitt"

    def apply(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        array = np.array(img, dtype=float)
        height, width = array.shape
        result = np.zeros_like(array)

        g_x = np.array([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
        ])
        g_y = np.array([
            [-1,-1,-1],
            [0,0,0],
            [1,1,1]
        ])

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                region = array[i-1:i+2, j-1:j+2]
                Gx = np.sum(region * g_x)
                Gy = np.sum(region * g_y)
                result[i, j] = np.sqrt(Gx**2 + Gy**2)

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

class ThresholdSegmentationFilter(FilterStrategy):
    """
    Клас конкетної реалізації стратегії колірної сегментації
    """
    def __init__(self, threshold: int):
        self.threshold = threshold

    @property
    def suffix(self) -> str:
        return "_tracehold"

    def apply(self, img: Image.Image) -> Image.Image:
        array = np.array(img, dtype=float)
        height, width, _ = array.shape
        result = np.zeros_like(array)

        for i in range(1, height-1):
            for j in range(1, width-1):
                if array[i,j,0] > self.threshold and array[i,j,1] > self.threshold and array[i,j,2] > self.threshold:
                    result[i,j] = [255,255,255]
                else:
                    result[i,j] = [0,0,0]

        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)

class HarrisFilter(FilterStrategy):
    """
    Клас конкетної реалізації стратегії фільтру Гарріса
    """
    def __init__(self, threshold: float):
        self.threshold = threshold

    @property
    def suffix(self) -> str:
        return "_harris"

    def apply(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        array = np.array(img, dtype=float)
        height, width = array.shape
        result = np.zeros_like(array)

        Ix = np.zeros_like(array)
        Iy = np.zeros_like(array)
        R = np.zeros_like(array)

        for i in range(1, height-1):
            for j in range(1, width-1):
                Ix[i,j] = array[i,j+1] - array[i,j]
                Iy[i,j] = array[i+1,j] - array[i,j]

        for i in range(1, height-1):
            for j in range(1, width-1):
                ix_block = Ix[i-1:i+2, j-1:j+2]
                iy_block = Iy[i-1:i+2, j-1:j+2]
                Sxx = np.sum(ix_block**2)
                Syy = np.sum(iy_block**2)
                Sxy = np.sum(ix_block * iy_block)
                M = np.array([[Sxx, Sxy],[Sxy, Syy]])
                R[i,j] = np.linalg.det(M) - 0.04 * (Sxx + Syy)**2

        t = self.threshold * np.max(R)
        result[R > t] = 255
        return Image.fromarray(result.astype(np.uint8))

class FASTFilter(FilterStrategy):
    """
    Клас конкетної реалізації стратегії фільтру FAST
    """
    def __init__(self, t: int):
        self.t = t

    @property
    def suffix(self) -> str:
        return "_fast"

    def apply(self, img: Image.Image) -> Image.Image:
        img = img.convert("L")
        array = np.array(img, dtype=float)
        height, width = array.shape
        result = np.zeros_like(array)

        c = [(-3,0),(-3,1),(-2,2),(-1,3),(0,3),(1,3),(2,2),(3,1),(3,0),(3,-1),(2,-2),(1,-3),(0,-3),(-1,-3),(-2,-2),(-3,-1)]
        N = 12

        for i in range(3, height-3):
            for j in range(3, width-3):
                p = array[i,j]
                circle = [array[i+di, j+dj] for di,dj in c]
                bright = [cc > p + self.t for cc in circle]
                dark   = [cc < p - self.t for cc in circle]

                bright2 = bright + bright
                dark2   = dark + dark

                count_bright = count_dark = 0
                keypoint = False

                for k in range(32):
                    if bright2[k]:
                        count_bright += 1
                        if count_bright >= N:
                            keypoint = True
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
                if keypoint:
                    result[i,j] = 255

        return Image.fromarray(result.astype(np.uint8))

class MetricsCalculator:
    """
    Клас розрахунку метрик оцінки
    """
    def __init__(self, gt_path: str = ""):
        self.ground_truth_path = gt_path

    def set_ground_truth(self, path: str):
        """
        Встановлення шляху до еталону
        """
        self.ground_truth_path = path

    def distortion_rate(self, original_path: str, filtered_path: str):
        """
        MSE метрика
        """
        orig = Image.open(original_path).convert("RGB")
        filt = Image.open(filtered_path).convert("RGB")

        orig_array = np.array(orig, dtype=float)
        filt_array = np.array(filt, dtype=float)

        return np.mean((filt_array - orig_array) ** 2)

    def psnr_rate(self, mse: float):
        """
        PSNR
        """
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 10 * np.log10((max_pixel ** 2) / mse)

    def ssim_rate(self, original_path: str, filtered_path: str) -> float:
        """
        SSIM
        """
        orig = Image.open(original_path).convert("L")
        filt = Image.open(filtered_path).convert("L")

        orig_array = np.array(orig, dtype=float)
        filt_array = np.array(filt, dtype=float)

        mu_x = np.mean(orig_array)
        mu_y = np.mean(filt_array)

        sigma_x_sq = np.var(orig_array)
        sigma_y_sq = np.var(filt_array)

        sigma_xy = np.mean((orig_array - mu_x) * (filt_array - mu_y))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)) + 1e-10
        ssim_value = num / den

        return max(ssim_value, 0.0)

    def mask_to_edge_array(self, mask_path: str):
        """
        Перетворення еталонних значень у масив для порівняння
        """
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask)

        mask_bin = mask_arr > 127
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(mask_bin.astype(np.uint8)*255, kernel)
        edges = (dilated > 127) & (~mask_bin)

        return edges

    def fom_rate(self, detected_path: str, alpha=1/9) -> Optional[float]:
        """
        FOM
        """
        if not self.ground_truth_path:
            return None
        
        detected = Image.open(detected_path).convert("L")
        detected_arr = np.array(detected) > 127
        gt_edges = self.mask_to_edge_array(self.ground_truth_path)

        N_d = np.sum(detected_arr)
        N_g = np.sum(gt_edges)

        if N_d == 0 or N_g == 0:
            return 0.0
        
        dist_map = distance_transform_edt(~gt_edges)
        detected_distances = dist_map[detected_arr]
        penalties = 1 / (1 + alpha * detected_distances ** 2)

        return np.sum(penalties) / max(N_d, N_g)

    def auc_roc_rate(self, detected_path: str) -> Optional[float]:
        """
        AUC-ROC
        """
        if not self.ground_truth_path:
            return None
        
        detected_raw = Image.open(detected_path).convert("L")
        detected_arr = np.array(detected_raw, dtype=float)
        gt = Image.open(self.ground_truth_path).convert("L")
        gt_arr = np.array(gt) > 127

        y_true = gt_arr.flatten().astype(int)
        y_score = detected_arr.flatten()

        if len(set(y_true)) < 2:
            return 0.0
        
        return roc_auc_score(y_true, y_score)

class ImageProcessingFacade:
    """
    Клас фасаду для застосування методів
    """
    def __init__(self):
        self.pre_strategies = {
            "Default": DefaultPreprocessing(),
            "White Balance": WhiteBalancePreprocessing(),
            "Histogram Equalization": HistogramEqualizationPreprocessing(),
            "Sharpen Filter": SharpenPreprocessing(),
            "CLAHE": CLAHEPreprocessing(),
            "Dark channel": DarkChannelPreprocessing(),
        }
        self.metrics = MetricsCalculator()

    def process_image(self, img: Image.Image, preprocess_name: str, filter_name: str,filter_param: Optional[float | int] = None) -> tuple[Image.Image, str]:
        """
        Повертає оброблене зображення та суфікс для збереження
        """
        # Препроцесинг
        pre = self.pre_strategies.get(preprocess_name, self.pre_strategies["Default"])
        processed = pre.apply(img)

        # Фільтр 
        if filter_name == "Canny":
            filt = CannyFilter()
        elif filter_name == "Directional":
            filt = DirectionalFilter()
        elif filter_name == "Pruitt":
            filt = PruittFilter()
        elif filter_name == "Tracehold":
            thresh = int(filter_param) if filter_param is not None else 200
            filt = ThresholdSegmentationFilter(thresh)
        elif filter_name == "Harris":
            thresh = float(filter_param) if filter_param is not None else 0.01
            filt = HarrisFilter(thresh)
        elif filter_name == "FAST":
            t = int(filter_param) if filter_param is not None else 20
            filt = FASTFilter(t)
        else:
            raise ValueError(f"Unknoqn filter: {filter_name}")

        result_img = filt.apply(processed)
        #суфікс
        suffix = pre.suffix + filt.suffix

        return result_img, suffix

class ImageApp:
    """
    Клас інтерфейсу програмного модуля
    """
    def __init__(self):
        self.state = {
            "images": [],
            "paths": [],
            "current_index": 0,
            "last_processed_image": None,
        }
        self.facade = ImageProcessingFacade()  #фасад
        self.metrics = self.facade.metrics     #метрики теж з фасаду

        self.root = tk.Tk()
        self.root.title('Image Processing')
        self.root.geometry("1000x600")
        self.root.minsize(1000, 600)

        self.preprocess_combobox = None
        self.metrics_label = None
        self.gt_entry = None

        self._create_ui()
        self.root.mainloop()

    def _create_ui(self):
        """
        Графічний інтерфейм модуля
        """
        tk.Button(self.root, text="Load set of images", command=self.load_images, font=("Arial", 12, "bold")).pack(side="top", pady=5)

        left_frame = tk.Frame(self.root, width=650, bg="#ffffff")
        left_frame.pack(side="left", fill="y", padx=5, pady=5)
        left_frame.pack_propagate(False)

        self.image_label = tk.Label(left_frame, bg="#333333", width=550, height=550)
        self.image_label.pack(side="top", padx=10, pady=10)

        nav_frame = tk.Frame(left_frame, bg="#ffffff")
        nav_frame.pack(side="bottom", pady=10)
        tk.Button(nav_frame, text="Previous", command=self.prev_image, width=15).pack(side="left", padx=10)
        tk.Button(nav_frame, text="Next", command=self.next_image, width=15).pack(side="left", padx=10)

        right_frame = tk.Frame(self.root, width=330, bg="#e9e9e9", relief=tk.RIDGE, bd=2)
        right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Filter Panel", bg="#e9e9e9", font=("Arial", 14, "bold")).pack(pady=10)

        preprocess_frame = tk.Frame(self.root, width=330, bg="#e9e9e9", relief=tk.RIDGE, bd=2)
        preprocess_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        preprocess_frame.pack_propagate(False)

        tk.Label(preprocess_frame, text="Preproccesing Panel", bg="#e9e9e9", font=("Arial", 14, "bold")).pack(pady=10)

        self.preprocess_combobox = ttk.Combobox(preprocess_frame, values=list(self.facade.pre_strategies.keys()), width=30)
        self.preprocess_combobox.pack()
        self.preprocess_combobox.set("Default")

        self.metrics_label = tk.Label(preprocess_frame, text="Metrics", bg="#e9e9e9", fg="#333333", font=("Arial", 10),
            justify="left", wraplength=300)
        self.metrics_label.pack(side="bottom", pady=20, padx=10, fill="x")

        tk.Label(preprocess_frame, text="Ground Truth FOM:", bg="#e9e9e9").pack(pady=15)
        self.gt_entry = tk.Entry(preprocess_frame, width=40)
        self.gt_entry.pack(pady=5)

        tk.Button(preprocess_frame, text="Обрати файл", command=self.choose_ground_truth).pack(pady=5)

        tk.Button(right_frame, text="Canny Edge Detector", width=20, command=self.canny_filter).pack(pady=15)
        tk.Button(right_frame, text="Directional Filter", width=15, command=self.directional).pack(pady=15)
        tk.Button(right_frame, text="Pruitt Filter", width=15, command=self.pruitt).pack(pady=15)

        tk.Label(right_frame, text="Tracehold Value", bg="#e9e9e9").pack()
        entry_tracehold = tk.Entry(right_frame)
        entry_tracehold.pack()
        entry_tracehold.insert(0, "200")
        tk.Button(right_frame, text="Tracehold segmentation", width=25, command=lambda: self.tracehold_segmentation(int(entry_tracehold.get()))).pack(pady=15)

        tk.Label(right_frame, text="Harris angle detectors tracehold Value", bg="#e9e9e9").pack()
        harris_entry = tk.Entry(right_frame)
        harris_entry.pack()
        harris_entry.insert(0, "0.01")
        tk.Button(right_frame, text="Harris angle detectors", width=20, command=lambda: self.harris_angle_detectors(float(harris_entry.get()))).pack(pady=15)

        tk.Label(right_frame, text=" Brightness threshold", bg="#e9e9e9").pack()
        fast_entry = tk.Entry(right_frame)
        fast_entry.pack()
        fast_entry.insert(0, "20")
        tk.Button(right_frame, text="Features from Accelerated Segment Test", width=35, command=lambda: self.fast(int(fast_entry.get()))).pack(pady=15)

    def load_images(self):
        """
        Завантаження картинок користувача
        """
        paths = filedialog.askopenfilenames(title="Choose an image", filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp;*.gif;*.webp")])
        if not paths:
            return

        self.state["images"] = [Image.open(p) for p in paths]
        self.state["paths"] = list(paths)
        self.state["current_index"] = 0
        self.show_image(self.state["images"][0])

    def show_image(self, img: Image.Image):
        """
        Відображення після обробки
        """
        if img is None:
            return
        
        img_resized = img.resize((550, 550), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def next_image(self):
        """
        Перехід між зображеннями
        """
        if not self.state["images"]:
            return
        
        self.state["current_index"] = (self.state["current_index"] + 1) % len(self.state["images"])
        self.show_image(self.state["images"][self.state["current_index"]])

    def prev_image(self):
        """
        Перехід між зображеннями
        """
        if not self.state["images"]:
            return
        
        self.state["current_index"] = (self.state["current_index"] - 1) % len(self.state["images"])
        self.show_image(self.state["images"][self.state["current_index"]])

    def choose_ground_truth(self):
        """
        Зчитування переданого еталону
        """
        path = filedialog.askopenfilename(title="Choose ground truth", filetypes=[("Image files", "*.jpg;*.png;*.jpeg;*.bmp")])
        if path:
            self.metrics.set_ground_truth(path)
            self.gt_entry.delete(0, tk.END)
            self.gt_entry.insert(0, path)

    def _update_metrics_display(self, mse, ssim=None, fom=None, psnr=None, roc=None):
        """
        Виведення обчислених метрик
        """
        text = f"MSE: {mse:.4f}\n"
        if ssim is not None:
            text += f"SSIM: {ssim:.4f} (0–1)\n"
        if fom is not None:
            text += f"FOM: {fom:.4f} (0–1)\n"
        if psnr is not None:
            text += f"PSNR: {psnr:.4f}\n"
        if roc is not None:
            text += f"ROC: {roc:.4f}"
        self.metrics_label.config(text=text)

    def _process_and_display(self, filter_name: str, filter_param: Optional[float | int] = None):
        """
        Загальна робота
        """
        if not self.state["images"]:
            return

        current_img = self.state["images"][self.state["current_index"]]
        current_path = self.state["paths"][self.state["current_index"]]

        pre_name = self.preprocess_combobox.get()

        result_img, suffix = self.facade.process_image(current_img, pre_name, filter_name, filter_param)

        save_path = save_image(result_img, current_path, suffix)

        self.state["last_processed_image"] = result_img
        self.show_image(result_img)

        mse = self.metrics.distortion_rate(current_path, save_path)
        ssim = self.metrics.ssim_rate(current_path, save_path)
        fom = self.metrics.fom_rate(save_path)
        psnr = self.metrics.psnr_rate(mse)
        roc = self.metrics.auc_roc_rate(save_path)

        self._update_metrics_display(mse, ssim, fom, psnr, roc)
    
    #Запуск усіх методів
    def canny_filter(self):
        self._process_and_display("Canny")

    def directional(self):
        self._process_and_display("Directional")

    def pruitt(self):
        self._process_and_display("Pruitt")

    def tracehold_segmentation(self, threshold: int):
        self._process_and_display("Tracehold", filter_param=threshold)

    def harris_angle_detectors(self, threshold: float):
        self._process_and_display("Harris", filter_param=threshold)

    def fast(self, t: int):
        self._process_and_display("FAST", filter_param=t)

if __name__ == "__main__":
    ImageApp()