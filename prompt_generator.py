import pandas as pd
import torch
from transformers import pipeline
import gc
from googletrans import Translator  # Google Translate API

def load_text_generation_model(model_id="dbmdz/bert-base-turkish-cased"):
    print("Model yükleniyor...")
    text_gen_pipeline = pipeline(
        "text-generation",  # Metin üretme modeli kullanıyoruz
        model=model_id,
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model yüklendi!")
    return text_gen_pipeline


def generate_prompt(text, model_pipeline):
    try:
        response = model_pipeline(text, max_length=100, do_sample=True, temperature=0.7)
        return response[0]["generated_text"]
    except Exception as e:
        print(f"Prompt oluşturma hatası: {e}")
        return "Prompt oluşturulamadı."


def translate_prompt(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='tr', dest='en')
        return translated.text
    except Exception as e:
        print(f"Çeviri hatası: {e}")
        return "Translation failed."


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("Otel değerlendirmelerinden prompt oluşturma işlemi başlıyor...")

    print("Veri seti yükleniyor...")
    df = pd.read_csv("Hotel_Readable_With_Name.csv")
    df["Combined_Review"] = df["Positive_Review_Tr"].fillna("") + " " + df["Negative_Review_Tr"].fillna("")

    unique_hotels = df["Hotel_Name"].unique()
    print(f"Toplam {len(unique_hotels)} benzersiz otel işlenecek.")

    text_gen_pipeline = load_text_generation_model()

    hotel_prompts = {}
    hotel_prompts_en = {}  # İngilizce promptlar için yeni bir dictionary
    batch_size = 10
    for i in range(0, len(unique_hotels), batch_size):
        batch_hotels = unique_hotels[i:i + batch_size]
        print(f"Grup {i // batch_size + 1}/{(len(unique_hotels) // batch_size) + 1} işleniyor...")

        for hotel in batch_hotels:
            hotel_reviews = df[df["Hotel_Name"] == hotel]["Combined_Review"]
            combined_review = " ".join(hotel_reviews)
            if len(combined_review) > 512:
                combined_review = combined_review[:512] + "..."

            prompt = generate_prompt(combined_review, text_gen_pipeline)
            hotel_prompts[hotel] = prompt
            hotel_prompts_en[hotel] = translate_prompt(prompt)  # Türkçe promptu İngilizce'ye çevir
            print(f"  ✓ {hotel} - prompt oluşturuldu")

        clear_memory()

    df["Generated_Prompt"] = df["Hotel_Name"].map(hotel_prompts)
    df["Generated_Prompt_English"] = df["Hotel_Name"].map(hotel_prompts_en)  # Yeni İngilizce kolon
    print("Promptlar oluşturuldu, CSV dosyası kaydediliyor...")
    df.to_csv("Hotel_with_Prompts_and_English.csv", index=False)
    print("İşlem tamamlandı!")

    clear_memory()
    return df


def test_first_two_hotels():
    print("Test başlıyor: İlk 2 otel için prompt oluşturma")
    df = pd.read_csv("Hotel_Readable_With_Name.csv")
    df["Combined_Review"] = df["Positive_Review_Tr"].fillna("") + " " + df["Negative_Review_Tr"].fillna("")

    unique_hotels = df["Hotel_Name"].unique()[:2]
    print(f"Test edilecek oteller: {unique_hotels}")
    text_gen_pipeline = load_text_generation_model()

    for i, hotel in enumerate(unique_hotels):
        print(f"\n{'=' * 50}")
        print(f"OTEL {i + 1}: {hotel}")
        print(f"{'=' * 50}")

        hotel_reviews = df[df["Hotel_Name"] == hotel]["Combined_Review"]
        combined_review = " ".join(hotel_reviews)
        print(f"Örnek değerlendirme (ilk 200 karakter):")
        print(f"{combined_review[:200]}...")

        print("\nPrompt oluşturuluyor...")
        prompt = generate_prompt(combined_review[:512], text_gen_pipeline)

        print("\nOLUŞTURULAN PROMPT:")
        print(f"{prompt}")
        print("\nİNGİLİZCE ÇEVİRİ:")
        print(translate_prompt(prompt))  # Çeviri de gösteriliyor
        clear_memory()

    print("\nTest tamamlandı!")


if __name__ == "__main__":
    mode = input("Test mi yapmak istiyorsunuz (1) yoksa tüm veriyi işlemek mi (2)? (1/2): ")
    try:
        if mode == "1":
            test_first_two_hotels()
        else:
            result_df = main()
            print(f"Toplam {len(result_df)} satır işlendi.")
    except Exception as e:
        print(f"Program çalışırken bir hata oluştu: {e}")
