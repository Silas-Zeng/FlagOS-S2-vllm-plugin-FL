"""Run the two exact peer requests, preserving transport/raw output evidence."""
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request


def run(out, fixtures):
    folder=out/'mixed-requests';folder.mkdir()
    payloads={name: json.loads((fixtures/(name+'-request.json')).read_text()) for name in ['decode','prefill']}
    origin={name:dict(sha256=hashlib.sha256((fixtures/(name+'-request.json')).read_bytes()).hexdigest(),payload=body)
            for name,body in payloads.items()}
    (folder/'requests.json').write_text(json.dumps(origin,ensure_ascii=False,indent=2)+'\n')
    assert origin['decode']['sha256']=='90e8284043b131ead2842c04cf83b5e3af3aa2f79358bee412769b76f61574c2'
    assert origin['prefill']['sha256']=='ecf8e319aa48e3c66e8b40019df2dff483ace79dbfdd61de41d436bb3961a42b'
    first=threading.Event();streamed={'events':[],'content':[]};errors=[]
    def request(body):
        return urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',
            data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'},method='POST')
    def decode():
        streamed['started']=time.monotonic()
        try:
            with urllib.request.urlopen(request(payloads['decode']),timeout=300) as response:
                with (folder/'decode-sse.txt').open('wb') as log:
                    for line in response:
                        log.write(line)
                        if not line.startswith(b'data: '):continue
                        raw=line[6:].strip()
                        if raw==b'[DONE]':break
                        event=json.loads(raw);streamed['events'].append(event)
                        for choice in event.get('choices',[]):
                            text=choice.get('delta',{}).get('content')
                            if text:
                                streamed['content'].append(text)
                                if not first.is_set():streamed['first_content']=time.monotonic();first.set()
            streamed['finished']=time.monotonic()
        except Exception as error:
            errors.append(repr(error));first.set()
    worker=threading.Thread(target=decode,daemon=True);worker.start()
    if not first.wait(90):raise TimeoutError('Decode did not return first content')
    if errors:raise RuntimeError(errors)
    prefill_start=time.monotonic()
    with urllib.request.urlopen(request(payloads['prefill']),timeout=300) as response:
        prefill=json.load(response)
    prefill_end=time.monotonic()
    (folder/'prefill-response.json').write_text(json.dumps(prefill,ensure_ascii=False,indent=2)+'\n')
    worker.join(timeout=300)
    if worker.is_alive():raise TimeoutError('Decode stream did not terminate')
    if errors:raise RuntimeError(errors)
    (folder/'decode-response.json').write_text(json.dumps(streamed,ensure_ascii=False,indent=2)+'\n')
    def visible(raw):
        if '<think>' in raw:
            if '</think>' not in raw:return ''
            raw=raw.rsplit('</think>',1)[1]
        return raw.strip()
    decode_text=visible(''.join(streamed['content']))
    message=prefill['choices'][0].get('message',{})
    prefill_text=visible(message.get('content') or '')
    numbers=re.findall(r'(?<!\d)\d+(?!\d)',decode_text)
    answers=re.findall(r'(?<!\d)\d+(?!\d)',prefill_text)
    passed_count=numbers==[str(i) for i in range(1,129)]
    passed_long=bool(answers and answers[-1]=='85')
    summary=dict(completed=True,all_correct=passed_count and passed_long,
                 decode_correct=passed_count,prefill_correct=passed_long,
                 decode_visible=decode_text,prefill_visible=prefill_text,
                 prefill_usage=prefill.get('usage'),prefill_started=prefill_start,prefill_finished=prefill_end,
                 decode_started=streamed['started'],decode_finished=streamed['finished'],
                 http_overlap=prefill_start<streamed['finished'],
                 limitation='Exact peer requests once; semantic diagnostic only, not105-question accuracy or model-step overlap proof.')
    (folder/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary
