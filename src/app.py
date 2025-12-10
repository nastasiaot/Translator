import os
from flask import Flask, render_template, request
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем API ключ из переменных окружения
API_KEY = os.getenv("MENTORPIECE_API_KEY")
API_URL = os.getenv("MENTORPIECE_API_URL", "https://api.mentorpiece.org/v1/process-ai-request")

# Инициализация Flask-приложения
app = Flask(__name__)

def call_llm(model_name, prompt):
    """
    Вспомогательная функция для обращения к LLM API.
    :param model_name: Имя модели для запроса
    :param prompt: Строка с промптом или сообщением
    :return: Ответ модели (строка) или сообщение об ошибке
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model_name": model_name,
        "prompt": prompt
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # Генерирует исключение для 4xx/5xx
        result = response.json()
        return result.get("response", "Нет ответа от модели.")
    except requests.exceptions.RequestException as e:
        # Обработка сетевых ошибок и ошибок API
        return f"Ошибка при обращении к LLM API: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    original_text = ''
    translated_text = ''
    verdict = ''
    selected_lang = 'en'
    if request.method == 'POST':
        # Получаем данные из формы
        original_text = request.form.get('original_text', '')
        selected_lang = request.form.get('language', 'en')
        # Формируем промпт для перевода
        lang_map = {'en': 'English', 'fr': 'French', 'de': 'German'}
        target_lang = lang_map.get(selected_lang, 'English')
        translate_prompt = f"Переведи следующий текст на {target_lang}:\n{original_text}"
        # Шаг 1: Перевод текста
        translated_text = call_llm(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            translate_prompt
        )
        # Шаг 2: Оценка перевода
        judge_prompt = (
            f"Оригинал: {original_text}\nПеревод: {translated_text}\n"
            "Оцени качество перевода от 1 до 10 и аргументируй."
        )
        verdict = call_llm(
            "claude-sonnet-4-5-20250929",
            judge_prompt
        )
    # Рендерим HTML-шаблон с результатами
    return render_template(
        'index.html',
        original_text=original_text,
        translated_text=translated_text,
        verdict=verdict,
        selected_lang=selected_lang
    )

if __name__ == '__main__':
    # Запуск приложения в режиме отладки
    app.run(debug=True, host='0.0.0.0', port=5000)
