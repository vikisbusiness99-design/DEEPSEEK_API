from flask import Flask, request, jsonify, Response
import requests
import json
import os
import time

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-r1"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 2048)
        stream = data.get('stream', False)
        top_p = data.get('top_p', 1.0)
        
        nim_payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if stream:
            return handle_streaming(nim_payload, headers)
        else:
            return handle_non_streaming(nim_payload, headers)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_non_streaming(payload, headers):
    response = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        json=payload,
        headers=headers,
        timeout=120
    )
    
    if response.status_code == 200:
        nim_response = response.json()
        openai_response = {
            "id": nim_response.get("id", f"chatcmpl-{int(time.time())}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": DEEPSEEK_MODEL,
            "choices": nim_response.get("choices", []),
            "usage": nim_response.get("usage", {})
        }
        return jsonify(openai_response)
    else:
        return jsonify({"error": response.text}), response.status_code

def handle_streaming(payload, headers):
    def generate():
        try:
            response = requests.post(
                f"{NVIDIA_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=120
            )
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        yield decoded + '\n\n'
                    else:
                        yield 'data: ' + decoded + '\n\n'
        except Exception as e:
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
    return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": DEEPSEEK_MODEL, "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-r1", "object": "model", "owned_by": "deepseek"}
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": DEEPSEEK_MODEL})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "NVIDIA NIM DeepSeek R1 Proxy",
        "model": DEEPSEEK_MODEL,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
