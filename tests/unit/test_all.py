"""
Полный набор unit-тестов для `src/app.py`.

Объединённые тесты покрывают:
- Успешные вызовы LLM API (200, 500)
- Загрузку переменных окружения (API_KEY, API_URL)
- Обработку ошибок (сетевые ошибки, невалидный JSON)
- Заголовок Authorization и структуру JSON-тела запроса
- Flask-интеграцию (POST на `/`, наличие шаблона)
- Отсутствие/пустой API ключ
- Граничные случаи (пустой ввод, длинный ввод, выбор языков)
- GET-запрос к корневому маршруту

Особенности реализации:
- Тесты НЕ используют моки; для проверки сетевых вызовов поднимаются локальные HTTP-серверы,
  которые имитируют внешний Mentorpiece API и сохраняют полученные запросы для проверок.
- Модуль `app` перезагружается перед каждым сценарием, чтобы он прочитал обновлённые переменные окружения.
- Это позволяет проверять реальные сетевые сценарии: заголовки, тело запроса, поведение при ошибках.

Запуск:
  pytest tests/unit/test_all.py -v
  или просто: pytest tests/unit -q
"""

import os
import sys
import json
import importlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import pytest

# Добавляем src в sys.path для импорта модуля app
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _start_capture_server(response_body=None, status=200, delay_seconds=0):
    """
    Запускает локальный HTTP-сервер, который:
    - принимает POST запросы
    - сохраняет заголовки и тела в server.last_request и server.requests
    - возвращает response_body (словарь -> JSON) с указанным статусом

    Возвращает (server, url)
    """
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            req = {
                'path': self.path,
                'headers': dict(self.headers),
                'body': body,
            }
            # Сохраняем последний запрос в last_request
            self.server.last_request = req
            # Также сохраняем в список requests для совместимости с тестами
            if not hasattr(self.server, 'requests'):
                self.server.requests = []
            self.server.requests.append(req)

            # Опциональная искусственная задержка
            if delay_seconds:
                import time
                time.sleep(delay_seconds)

            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            if status == 200 and response_body is not None:
                self.wfile.write(json.dumps(response_body).encode('utf-8'))
            else:
                self.wfile.write(json.dumps({'error': 'server error', 'code': status}).encode('utf-8'))

        def log_message(self, format, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    url = f'http://{host}:{port}/v1/process-ai-request'
    return server, url


def _stop_server(server):
    try:
        server.shutdown()
        server.server_close()
    except Exception:
        pass


def _reload_app():
    """Перезагрузить модуль `app` из папки `src` и вернуть его."""
    if 'app' in sys.modules:
        del sys.modules['app']
    app = importlib.import_module('app')
    importlib.reload(app)
    return app


# ==================== ТЕСТЫ ====================


def test_positive_call_llm():
    """
    Positive Test: успешный ответ API (200) -> возвращается поле `response`.
    """
    server, url = _start_capture_server(response_body={'response': 'TEST_RESPONSE_OK'}, status=200)
    try:
        os.environ['MENTORPIECE_API_URL'] = url
        os.environ['MENTORPIECE_API_KEY'] = 'test-api-key'
        app = _reload_app()

        result = app.call_llm('dummy-model', 'translate this')
        assert result == 'TEST_RESPONSE_OK'
    finally:
        _stop_server(server)


def test_env_loading_api_key():
    """Environment Test: проверяем загрузку API_KEY из окружения."""
    os.environ['MENTORPIECE_API_KEY'] = 'env-key-12345'
    app = _reload_app()
    assert hasattr(app, 'API_KEY')
    assert app.API_KEY == 'env-key-12345'


def test_error_handling_call_llm():
    """Error Handling: сервер возвращает 500, функция должна вернуть строку ошибки."""
    server, url = _start_capture_server(response_body={}, status=500)
    try:
        os.environ['MENTORPIECE_API_URL'] = url
        os.environ['MENTORPIECE_API_KEY'] = 'test-api-key'
        app = _reload_app()

        result = app.call_llm('dummy', 'prompt')
        assert isinstance(result, str)
        assert 'Ошибка при обращении к LLM API' in result
    finally:
        _stop_server(server)


def test_authorization_header_and_body_structure():
    """
    Проверяем, что `call_llm` отправляет заголовок Authorization: Bearer <API_KEY>
    и формат JSON-тела содержит `model_name` и `prompt`.
    """
    server, url = _start_capture_server(response_body={'response': 'ok'}, status=200)
    try:
        os.environ['MENTORPIECE_API_URL'] = url
        os.environ['MENTORPIECE_API_KEY'] = 'super-secret-key'
        app = _reload_app()

        out = app.call_llm('my-model', 'my-prompt')

        assert hasattr(server, 'last_request'), 'Сервер должен сохранить последний запрос в server.last_request'
        last = server.last_request

        # Authorization header
        auth = last['headers'].get('Authorization') or last['headers'].get('authorization')
        assert auth == 'Bearer super-secret-key', f'Authorization header должен быть установлен, получено: {auth}'

        # JSON body проверка
        body = last['body'].decode('utf-8')
        payload = json.loads(body)
        assert 'model_name' in payload and payload['model_name'] == 'my-model'
        assert 'prompt' in payload and payload['prompt'] == 'my-prompt'

        # И сам вызов возвращает текст из response
        assert out == 'ok'
    finally:
        _stop_server(server)


def test_flask_route_post_integration_and_template_presence():
    """
    Интеграционный тест POST на `/` через Flask test client и проверка HTML-шаблона.
    """
    server, url = _start_capture_server(response_body={'response': 'TRANSLATED_TEXT'}, status=200)
    try:
        os.environ['MENTORPIECE_API_URL'] = url
        os.environ['MENTORPIECE_API_KEY'] = 'key'
        app = _reload_app()

        client = app.app.test_client()

        data = {
            'original_text': 'Привет, мир',
            'language': 'en'
        }
        resp = client.post('/', data=data)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        # Проверки присутствия элементов шаблона
        assert '<textarea' in html
        assert '<select' in html
        assert 'Перевод:' in html or 'Перевод' in html

        # Поскольку наш тестовый сервер возвращает 'TRANSLATED_TEXT', ожидаем его в HTML
        assert 'TRANSLATED_TEXT' in html
    finally:
        _stop_server(server)


def test_missing_api_key_handling():
    """
    Проверка поведения при отсутствии переменной окружения `MENTORPIECE_API_KEY`.
    """
    server, url = _start_capture_server(response_body={'response': 'ok'}, status=200)
    try:
        # Удаляем ключ окружения если он есть и явно устанавливаем пустой ключ.
        os.environ.pop('MENTORPIECE_API_KEY', None)
        os.environ['MENTORPIECE_API_KEY'] = ''
        os.environ['MENTORPIECE_API_URL'] = url

        app = _reload_app()
        assert getattr(app, 'API_KEY', 'exists') == ''

        # Вызов должен пройти и сервер должен получить заголовок Authorization с 'Bearer '
        res = app.call_llm('m', 'p')
        assert hasattr(server, 'last_request')
        last = server.last_request
        auth = last['headers'].get('Authorization') or last['headers'].get('authorization')
        assert auth == 'Bearer '
        assert res == 'ok'
    finally:
        _stop_server(server)


def test_connection_refused_handling():
    """
    Симулируем отказ соединения, указывая MENTORPIECE_API_URL на порт, где ничего не слушает.
    Ожидаем, что `call_llm` вернёт строку с описанием ошибки, а не бросит необработанное исключение.
    """
    os.environ['MENTORPIECE_API_URL'] = 'http://127.0.0.1:9/v1/process-ai-request'
    os.environ['MENTORPIECE_API_KEY'] = 'k'
    app = _reload_app()

    # Пытаемся вызвать — на 127.0.0.1:9 обычно нет сервера => ConnectionError
    res = app.call_llm('model', 'p')
    assert isinstance(res, str)
    assert 'Ошибка при обращении к LLM API' in res


def test_edge_cases_empty_and_long_input():
    """
    Тестируем поведение при пустом и длинном входном тексте.
    """
    # 1) Empty input
    server1, url1 = _start_capture_server(response_body={'response': 'EMPTY_OK'}, status=200)
    try:
        os.environ['MENTORPIECE_API_URL'] = url1
        os.environ['MENTORPIECE_API_KEY'] = 'key'
        app = _reload_app()
        out1 = app.call_llm('m', '')
        assert out1 == 'EMPTY_OK'
    finally:
        _stop_server(server1)

    # 2) Long input
    long_text = 'A' * 20000  # 20k символов
    server2, url2 = _start_capture_server(response_body={'response': 'LONG_OK'}, status=200)
    try:
        os.environ['MENTORPIECE_API_URL'] = url2
        os.environ['MENTORPIECE_API_KEY'] = 'key'
        app = _reload_app()
        out2 = app.call_llm('m', long_text)
        assert out2 == 'LONG_OK'
        assert 'body' in server2.last_request
        # Проверяем, что присланный prompt содержит часть длинного текста
        body_txt = server2.last_request['body'].decode('utf-8')
        assert 'A' * 100 in body_txt
    finally:
        _stop_server(server2)


def test_language_selection_prompt():
    """
    Проверяет, что при выборе разных языков в форме
    промпт для перевода формируется с нужным языком.
    """
    server, url = _start_capture_server(response_body={'response': 'ok'}, status=200)
    try:
        os.environ['MENTORPIECE_API_URL'] = url
        os.environ['MENTORPIECE_API_KEY'] = 'key'
        app = _reload_app()
        client = app.app.test_client()
        for lang, lang_name in [('en', 'English'), ('fr', 'French'), ('de', 'German')]:
            data = {'original_text': 'Тест', 'language': lang}
            client.post('/', data=data)
            # Проверяем промпт для перевода (первый запрос)
            reqs = getattr(server, 'requests', [])
            assert len(reqs) >= 1
            payload = json.loads(reqs[0]['body'].decode('utf-8'))
            assert f'на {lang_name}' in payload['prompt']
            server.requests.clear()  # очищаем для следующей итерации
    finally:
        _stop_server(server)


def test_invalid_json_response():
    """
    Проверяет, что если API возвращает невалидный JSON,
    функция call_llm возвращает строку с ошибкой, а не падает.
    """
    class BadJSONHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{invalid_json')

        def log_message(self, format, *args):
            return

    server = HTTPServer(('127.0.0.1', 0), BadJSONHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_address[1]}/v1/process-ai-request'
    try:
        os.environ['MENTORPIECE_API_URL'] = url
        os.environ['MENTORPIECE_API_KEY'] = 'key'
        app = _reload_app()
        result = app.call_llm('m', 'p')
        assert isinstance(result, str)
        assert 'Ошибка при обращении к LLM API' in result or 'Expecting' in result
    finally:
        server.shutdown()
        server.server_close()


def test_get_root_returns_form():
    """
    Проверяет, что GET-запрос к / возвращает HTML с формой (textarea, select, кнопки).
    """
    app = _reload_app()
    client = app.app.test_client()
    resp = client.get('/')
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert '<textarea' in html
    assert '<select' in html
    assert 'Перевести' in html
