#!/usr/bin/env python3
"""Exercise a running local server; no model downloads or server startup.

API keys are read from EXL3_API_KEY and never printed. Exit nonzero on failure.
"""
import argparse
import json
import os
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8000')
    ap.add_argument('--model', default='qwen38-27b-exl3')
    args = ap.parse_args()
    key = os.environ.get('EXL3_API_KEY')

    def request(path, body=None, authenticated=True):
        headers = {'Content-Type': 'application/json'}
        if key and authenticated:
            headers['Authorization'] = 'Bearer ' + key
        req = urllib.request.Request(args.url.rstrip('/') + path,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
        return urllib.request.urlopen(req, timeout=180)

    with request('/health') as response:
        assert json.load(response)['status'] == 'ok'
    with request('/v1/models') as response:
        assert args.model in [entry['id'] for entry in json.load(response)['data']]
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': 'Reply with the word hello.'}],
            'max_tokens': 32, 'temperature': 0, 'stream': False}
    with request('/v1/chat/completions', body) as response:
        result = json.load(response)
    assert result['object'] == 'chat.completion', result
    assert result['choices'][0]['message']['content'], result
    assert result['usage']['completion_tokens'] > 0, result
    print('PASS health, models, non-streaming chat, token usage')
    body['stream'] = True
    chunks, content, done = 0, '', False
    with request('/v1/chat/completions', body) as response:
        assert 'text/event-stream' in response.headers['Content-Type']
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            assert event['object'] == 'chat.completion.chunk', event
            chunks += 1
            for choice in event['choices']:
                content += choice.get('delta', {}).get('content') or ''
    assert chunks and content and done, (chunks, content, done)
    print('PASS streaming chat and [DONE] terminator')
    if key:
        try:
            request('/v1/chat/completions', body, authenticated=False)
        except urllib.error.HTTPError as error:
            assert error.code == 401, error.code
        else:
            raise AssertionError('Unauthenticated chat request was accepted')
        print('PASS unauthenticated request rejected')


if __name__ == '__main__':
    main()
