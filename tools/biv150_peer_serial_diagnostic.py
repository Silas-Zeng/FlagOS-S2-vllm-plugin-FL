"""Exact long request alone, once in a fresh process; no accuracy/throughput claim."""
import hashlib,json,re,time,urllib.request

def run(out,fixtures):
    folder=out/'serial-request';folder.mkdir()
    raw=(fixtures/'prefill-request.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='ecf8e319aa48e3c66e8b40019df2dff483ace79dbfdd61de41d436bb3961a42b'
    (folder/'request.json').write_bytes(raw)
    request=urllib.request.Request('http://127.0.0.1:9031/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'},method='POST')
    began=time.monotonic()
    with urllib.request.urlopen(request,timeout=300) as response:reply=json.load(response)
    (folder/'response.json').write_text(json.dumps(reply,ensure_ascii=False,indent=2)+'\n')
    text=reply['choices'][0]['message'].get('content') or ''
    if '<think>' in text:text=text.rsplit('</think>',1)[1] if '</think>' in text else ''
    values=re.findall(r'(?<!\d)\d+(?!\d)',text)
    passed=bool(values and values[-1]=='85')
    result=dict(completed=True,prefill_correct=passed,visible=text.strip(),usage=reply.get('usage'),
                started=began,finished=time.monotonic(),http_overlap=False,
                limitation='One exact long request alone; not an accuracy/throughput test or kernel attribution.')
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
