import os
import csv
from PIL import Image
import stem

def log_metrics_to_csv(csv_path: str, row: list):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow(["Image_Name", "Preprocessing", "Filter", "MSE", "SSIM", "PSNR", "FOM", "AUC-ROC", "Save_Path"])
        writer.writerow(row)


def batch_process(images_folder: str, gt_folder: str, results_csv: str = "results2.csv", start_from: str = None):
    if not os.path.exists(images_folder) or not os.path.exists(gt_folder):
        print("Папка не знайдена")
        return

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg','.jpeg'))]
    image_files.sort()

    if start_from and start_from in image_files:
        image_files = image_files[image_files.index(start_from):]

    facade = stem.ImageProcessingFacade()

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        base_name = os.path.splitext(img_file)[0]
        gt_path = os.path.join(gt_folder, base_name + ".png")

        if not os.path.exists(gt_path):
            print(f"GT не знайдено: {img_file}")
            continue

        facade.metrics.set_ground_truth(gt_path)

        print(f"\n=== Обробка: {img_file} ===")
        img = Image.open(img_path)

        for pre_name in facade.pre_strategies.keys():
            for filter_name, default_param in [
                ("Canny", None),
                ("Directional", None),
                ("Pruitt", None),
                ("Tracehold", 200),
                ("Harris", 0.01),
                ("FAST", 20),
            ]:
                try:
                    result_img, suffix = facade.process_image(img, pre_name, filter_name, filter_param=default_param)

                    save_path = stem.save_image(result_img, img_path, suffix)

                    mse   = facade.metrics.distortion_rate(img_path, save_path)
                    ssim  = facade.metrics.ssim_rate(img_path, save_path)
                    psnr  = facade.metrics.psnr_rate(mse)
                    fom   = facade.metrics.fom_rate(save_path)
                    auc   = facade.metrics.auc_roc_rate(save_path)

                    log_metrics_to_csv(results_csv, [
                        img_file,
                        pre_name,
                        filter_name,
                        f"{mse:.4f}".replace('.', ','),
                        f"{ssim:.4f}".replace('.', ','),
                        f"{psnr:.4f}".replace('.', ','),
                        f"{fom:.4f}".replace('.', ',') if fom is not None else "N/A",
                        f"{auc:.4f}".replace('.', ',') if auc is not None else "N/A",
                        save_path
                    ])

                    print(f"  {pre_name} + {filter_name} → {save_path}")

                except Exception as e:
                    print(f"      Помилка {pre_name} / {filter_name}: {e}")

    print("\nГотово! Результати збережено в", results_csv)


if __name__ == "__main__":
    images_folder = r"D:\Новая папка\Uni 2 course  1 semestr\КГВ\STEM\images"
    gt_folder     = r"D:\Новая папка\Uni 2 course  1 semestr\КГВ\STEM\ground_truth"
    batch_process(images_folder, gt_folder, start_from="image_part_003.jpg")