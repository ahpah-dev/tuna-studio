"""Measure the same local chat request without changing model or conversation data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

def bounded_integer(minimum, maximum):
    def parse(text):
        value = int(text)
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(f'Use a value between {minimum} and {maximum}.')
        return value
    return parse

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=bounded_integer(1,10), default=3)
parser.add_argument('--tokens', type=bounded_integer(1,2048), default=128)
args = parser.parse_args()
url = 'http://127.0.0.1:8765/api'

def call(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url+path, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)

initial = call('/chat')['chat']
if initial['status'] != 'ready':
    raise RuntimeError('The selected model must be ready and idle before benchmarking.')
prompt = 'Write a complete single-file HTML game where a bird avoids pipes. Include HTML, CSS and JavaScript code.'
rows = []
for index in range(args.repeats):
    start = time.perf_counter()
    response = call('/chat/message', {'run_id':initial['run_id'],'source':initial['source'],
        'messages':[{'role':'user','content':prompt}], 'max_new_tokens':args.tokens,'temperature':0})['chat']
    identifier = response['request_id']
    first_text = None
    while True:
        if time.perf_counter()-start > 600:
            raise TimeoutError('The reply did not complete within ten minutes.')
        state = call('/chat')['chat']
        if state.get('request_id') == identifier and state.get('text') and first_text is None:
            first_text = time.perf_counter()-start
        reply = call('/chat/replies/'+identifier)['reply']
        if reply:
            if reply.get('error'): raise RuntimeError(reply['error'])
            text = reply['text']
            row = {'iteration':index+1,'wall_seconds':time.perf_counter()-start,
                   'first_visible_text_seconds':first_text,'usage':reply.get('usage'),
                   'finish_reason':reply['finish_reason'], 'output_sha256':hashlib.sha256(text.encode()).hexdigest(),
                   'text_prefix':text[:160]}
            rows.append(row)
            print(json.dumps(row),flush=True)
            break
        if state['status'] in ('error','unloaded'):
            raise RuntimeError(state.get('error') or 'The model was unloaded during the benchmark.')
        time.sleep(.05)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'run_id':initial['run_id'],'source':initial['source'],
    'device':initial.get('device'),'engine':initial.get('engine'), 'precision':initial.get('precision'),
    'optimizations':initial.get('optimizations'),
    'prompt':prompt,'max_new_tokens':args.tokens,'temperature':0,'results':rows}, indent=2),encoding='utf-8')
