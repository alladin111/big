import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
import json
import matplotlib.pyplot as plt
import concurrent.futures
from transformers import pipeline

# Загрузка ресурсов NLTK для русского языка
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Инициализация инструментов предобработки
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))

# Загрузка нейросетевой модели для перевода (Helsinki-NLP)
print("Загрузка модели перевода (при первом запуске может занять время)...")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

# Функция для предварительной обработки текста (Токенизация, удаление стоп-слов, Лемматизация)
def preprocess_text(text):
    # 1. Токенизация
    tokens = word_tokenize(text.lower(), language="russian")
    
    # 2. Удаление стоп-слов и пунктуации
    filtered_tokens = [
        word for word in tokens 
        if word.isalnum() and word not in russian_stopwords
    ]
    
    # 3. Лемматизация
    lemmas = [morph.parse(word)[0].normal_form for word in filtered_tokens]
    
    return " ".join(lemmas)

# Функция для анализа тональности текста
def analyze_sentiment(text):
    # Предобработка
    processed_text = preprocess_text(text)
    
    # Нейросетевой перевод на английский для TextBlob
    try:
        translation_result = translator(text)
        text_for_blob = translation_result[0]['translation_text']
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        text_for_blob = text 
        
    # Анализ тональности
    blob = TextBlob(text_for_blob)
    polarity = float(blob.sentiment.polarity)  # type: ignore[attr-defined]
    subjectivity = float(blob.sentiment.subjectivity)  # type: ignore[attr-defined]
    # Логика классификации с настройкой нейтральной зоны
    NEUTRAL_THRESHOLD = 0.5  # Диапазон [-0.5, +0.5] считается нейтральным
    
    if abs(polarity) <= NEUTRAL_THRESHOLD:
        label = "NEUTRAL"
    elif polarity > 0:
        label = "POSITIVE"
    else:
        label = "NEGATIVE"

        
    return {
        "original_text": text,
        "translated_text": text_for_blob,
        "processed_text": processed_text,
        "label": label,
        "score": round(polarity, 4),
        "subjectivity": round(subjectivity, 4)
    }

# Функция для визуализации результатов анализа тональности
def visualize_results(results, threshold=0.5):
    """
    Визуализация результатов анализа тональности с пороговыми значениями.
    
    Args:
        results (list): список словарей с ключами 'original_text', 'score', 'label'
        threshold (float): ширина нейтральной зоны (по умолчанию 0.5)
    """
    texts = [r.get("original_text", f"Текст {i+1}") for i, r in enumerate(results)]
    scores = [r["score"] for r in results]
    labels = [r["label"] for r in results]
    
    x_pos = range(len(texts))
    # Цвета по классификации
    colors_map = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#95a5a6"}
    scatter_colors = [colors_map.get(l, "#3498db") for l in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Отображаем точки/столбцы по номеру текста
    ax.scatter(x_pos, scores, c=scatter_colors, s=250, zorder=3, edgecolors='black', linewidth=1.5)
    
    # Подписи над точками: значение + классификация
    for i in x_pos:
        score_str = f"{scores[i]:+.2f}"
        label_str = labels[i]
        # Текст выше точки, если полярность >= 0, иначе ниже
        va = "bottom" if scores[i] >= 0 else "top"
        ax.annotate(f"{score_str}\n({label_str})", 
                    xy=(x_pos[i], scores[i]), 
                    ha='center', va=va, fontsize=10, fontweight='bold', color=scatter_colors[i])
        
        # Подпись текста на оси X (сокращаем длинные фразы)
        short_txt = texts[i][:20].replace('\n', ' ') + ("..." if len(texts[i]) > 20 else "")
        ax.text(x_pos[i], -1.3, short_txt, ha='center', va='center', fontsize=9, rotation=45)

    # Горизонтальные линии порогов
    ax.axhline(y=threshold, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.8, label=f'+{threshold} (POSITIVE)')
    ax.axhline(y=-threshold, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8, label=f'-{threshold} (NEGATIVE)')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.2, alpha=0.6, label='Нулевая линия')
    
    # Настройки осей и заголовок
    ax.set_xticks(x_pos)
    ax.set_xlim(-0.5, len(texts) - 0.5)
    ax.set_ylim(-1.3, 1.2)
    ax.set_ylabel('Полярность (Polarity)', fontsize=12, fontweight='bold')
    ax.set_title('Анализ тональности с пороговой нейтральной зоной', fontsize=14, pad=15)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# Параллельная обработка большого объема данных
def parallel_analyze_sentiments(reviews):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(analyze_sentiment, review) for review in reviews]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    return results


# Основной блок кода для обработки и анализа текста
if __name__ == "__main__":
    # Заданные тексты
    texts = [
        "Сегодня ужасный день, все пошло не так.",
        "Я люблю Python. Это отличный язык программирования!",
        "Все в порядке, ничего особенного."
    ]
    
    # Одиночный анализ для проверки
    print("\n--- Одиночный анализ (Текст 1) ---")
    single_result = analyze_sentiment(texts[0])
    print(json.dumps(single_result, indent=2, ensure_ascii=False))
    
    # Параллельная обработка всех текстов
    print("\n--- Параллельный анализ всех текстов ---")
    parallel_results = parallel_analyze_sentiments(texts)
    for res in parallel_results:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        
    # Визуализация
    visualize_results(parallel_results, threshold=0.5)
   