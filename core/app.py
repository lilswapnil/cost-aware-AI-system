import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from router import Router
from schemas import Request, Response

load_dotenv()

app = Flask(__name__)

@app.route('/route', methods=['POST'])
def route_prompt():
    data = request.get_json()
    prompt = data.get('prompt')
    max_cost = float(data.get('max_cost', 1.0))
    max_latency = float(data.get('max_latency', 2.0))
    if not prompt:
        return jsonify({'error': 'Missing prompt'}), 400
    req = Request(prompt=prompt, max_cost=max_cost, max_latency=max_latency)
    r = Router()
    result = r.route(req.prompt, req.max_cost, req.max_latency)
    resp = Response(**result)
    return jsonify({
        'model': resp.model,
        'cost': resp.cost,
        'latency_ms': resp.latency_ms,
        'within_budget': resp.within_budget,
        'within_latency': resp.within_latency
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
