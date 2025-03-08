import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from torch.cuda.amp import autocast
import gc
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
import shutil
import re
import unicodedata


def calculate_binary_metrics(original_image, generated_image, threshold=50):  # threshold artırıldı
    """
    Görüntüler arasındaki F1 ve Accuracy metriklerini hesaplar.
    threshold: İki piksel arasındaki farkın binary sınıflandırma için eşik değeri
    """
    # Görüntüleri numpy dizilerine dönüştür
    orig_array = np.array(original_image.convert('L'))  # Gri tonlamaya çevir
    gen_array = np.array(generated_image.convert('L'))

    # Piksel farklarını hesapla
    diff = np.abs(orig_array - gen_array)

    # Binary sınıflandırma yap (threshold'a göre)
    binary_diff = diff <= threshold

    # True/False Positive/Negative hesapla
    total_pixels = binary_diff.size
    true_positives = np.sum(binary_diff)
    false_positives = total_pixels - true_positives

    # Metrikleri hesapla
    accuracy = true_positives / total_pixels

    # Precision ve Recall hesapla (görüntü karşılaştırması için adapte edilmiş)
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / total_pixels

    # F1 skoru hesapla
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    return {
        'Accuracy': accuracy,
        'F1_Score': f1_score,
        'Precision': precision,
        'Recall': recall
    }


def calculate_metrics(original_image, generated_image):
    """Tüm görüntü metriklerini hesaplar."""
    # Görüntüleri numpy dizilerine dönüştür
    orig_array = np.array(original_image)
    gen_array = np.array(generated_image)

    # Görüntüleri gri tonlamaya çevir (SSIM için gerekli)
    orig_gray = np.array(original_image.convert('L'))
    gen_gray = np.array(generated_image.convert('L'))

    # Temel metrikleri hesapla
    mse_value = mean_squared_error(orig_array, gen_array)
    ssim_value = ssim(orig_gray, gen_gray, data_range=gen_gray.max() - gen_gray.min())
    psnr_value = psnr(orig_array, gen_array, data_range=255)

    # Binary metrikleri hesapla
    binary_metrics = calculate_binary_metrics(original_image, generated_image)

    # Tüm metrikleri birleştir
    return {
        'MSE': mse_value,
        'SSIM': ssim_value,
        'PSNR': psnr_value,
        **binary_metrics
    }


def normalize_hotel_name(name):
    """Otel isimlerini normalize eder."""
    # Unicode karakterleri ASCII'ye dönüştür
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    # Özel karakterleri temizle ve boşlukları '_' ile değiştir
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'[-\s]+', '_', name)
    return name


def setup_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA destekli GPU bulunamadı!")

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    return pipe


def process_image(pipe, image_path, prompt, strength=0.45):  # strength düşürüldü
    try:
        init_image = Image.open(image_path).convert("RGB")

        # Görüntü boyutunu 768x768 olarak ayarla
        width, height = init_image.size
        target_size = 768

        # En-boy oranını koruyarak yeniden boyutlandır
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))

        # Görüntüyü yeniden boyutlandır
        init_image = init_image.resize((new_width, new_height))

        # Eğer boyutlar 768'den küçükse, padding ekle
        if new_width < target_size or new_height < target_size:
            padded_image = Image.new("RGB", (target_size, target_size))
            # Görüntüyü merkeze yerleştir
            x_offset = (target_size - new_width) // 2
            y_offset = (target_size - new_height) // 2
            padded_image.paste(init_image, (x_offset, y_offset))
            init_image = padded_image

        # Prompt'u güçlendir
        enhanced_prompt = f"Same exact hotel room, {prompt}, preserve original layout and details"

        with autocast():
            result_image = pipe(
                prompt=enhanced_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=50,  # inference steps artırıldı
                guidance_scale=5.5,  # guidance scale düşürüldü
            ).images[0]

        return init_image, result_image
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return None, None


def generate_summary_report(metrics_df, results_dir):
    """Özet rapor oluşturur ve kaydeder."""
    summary = pd.DataFrame({
        'Metric': ['MSE', 'SSIM', 'PSNR', 'Accuracy', 'F1_Score', 'Precision', 'Recall'],
        'Mean': [
            metrics_df['MSE'].mean(),
            metrics_df['SSIM'].mean(),
            metrics_df['PSNR'].mean(),
            metrics_df['Accuracy'].mean(),
            metrics_df['F1_Score'].mean(),
            metrics_df['Precision'].mean(),
            metrics_df['Recall'].mean()
        ],
        'Std': [
            metrics_df['MSE'].std(),
            metrics_df['SSIM'].std(),
            metrics_df['PSNR'].std(),
            metrics_df['Accuracy'].std(),
            metrics_df['F1_Score'].std(),
            metrics_df['Precision'].std(),
            metrics_df['Recall'].std()
        ],
        'Min': [
            metrics_df['MSE'].min(),
            metrics_df['SSIM'].min(),
            metrics_df['PSNR'].min(),
            metrics_df['Accuracy'].min(),
            metrics_df['F1_Score'].min(),
            metrics_df['Precision'].min(),
            metrics_df['Recall'].min()
        ],
        'Max': [
            metrics_df['MSE'].max(),
            metrics_df['SSIM'].max(),
            metrics_df['PSNR'].max(),
            metrics_df['Accuracy'].max(),
            metrics_df['F1_Score'].max(),
            metrics_df['Precision'].max(),
            metrics_df['Recall'].max()
        ]
    })

    summary.to_csv(os.path.join(results_dir, 'metrics_summary.csv'), index=False)


def main():
    results_dir = "results-yeni"
    os.makedirs(results_dir, exist_ok=True)

    pipe = setup_pipeline()
    df = pd.read_csv("Hotel_with_Prompts_and_English.csv")
    images_dir = r"C:\Users\Ata_Onur_Ozdemir\PycharmProjects\image_manipulation_deneme\otel_room_images"

    metrics_data = []
    hotel_name_mapping = {normalize_hotel_name(hotel): hotel for hotel in df['Hotel_Name'].unique()}

    for original_hotel_name in tqdm(df['Hotel_Name'].unique(), desc="Oteller işleniyor"):
        normalized_hotel_name = normalize_hotel_name(original_hotel_name)
        prompt = df[df['Hotel_Name'] == original_hotel_name]['Generated_Prompt_English'].iloc[0]

        hotel_images_dir = os.path.join(images_dir, original_hotel_name)
        normalized_hotel_images_dir = os.path.join(images_dir, normalized_hotel_name)

        working_dir = None
        if os.path.exists(hotel_images_dir):
            working_dir = hotel_images_dir
        elif os.path.exists(normalized_hotel_images_dir):
            working_dir = normalized_hotel_images_dir

        if working_dir is None:
            print(f"⚠️ {original_hotel_name} için klasör bulunamadı.")
            continue

        hotel_results_dir = os.path.join(results_dir, normalized_hotel_name)
        os.makedirs(hotel_results_dir, exist_ok=True)

        image_files = [f for f in os.listdir(working_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in tqdm(image_files, desc=f"{original_hotel_name} görselleri işleniyor"):
            image_path = os.path.join(working_dir, image_file)
            original_image, result_image = process_image(pipe, image_path, prompt)

            if result_image and original_image:
                result_path = os.path.join(hotel_results_dir, f"manipulated_{image_file}")
                result_image.save(result_path)

                image_metrics = calculate_metrics(original_image, result_image)
                metrics_data.append({
                    'Hotel_Name': original_hotel_name,
                    'Image_Name': image_file,
                    **image_metrics
                })

        torch.cuda.empty_cache()
        gc.collect()

    # Metrikleri DataFrame'e dönüştür
    metrics_df = pd.DataFrame(metrics_data)

    # Otel bazında ortalama metrikleri hesapla
    hotel_metrics = metrics_df.groupby('Hotel_Name').mean()

    # Metrikleri kaydet
    metrics_df.to_csv(os.path.join(results_dir, 'image_metrics.csv'), index=False)
    hotel_metrics.to_csv(os.path.join(results_dir, 'hotel_average_metrics.csv'))

    # Özet rapor oluştur
    generate_summary_report(metrics_df, results_dir)


if __name__ == "__main__":
    main()