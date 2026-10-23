"""Paired steady-decode diagnostic; graph mode NONE is never a scored configuration."""
import hashlib
import concurrent.futures
import random
import re
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from run_biv150_operator_arm import get_json, benchmark


def post(path):
    req=urllib.request.Request("http://127.0.0.1:9031"+path,data=b"",method="POST")
    with urllib.request.urlopen(req,timeout=180) as r:
        print(json.dumps(dict(endpoint=path,status=r.status)),flush=True)


def diagnostic(folder):
    params=dict(model="minicpm",max_tokens=512,ignore_eos=True,temperature=1.0,top_p=0.95,stream=False)
    metadata=dict(params=params,input_tokens=15360,requests=32,initial_test_request=False,cudagraph_mode="NONE",purpose="diagnostic only; not a scored benchmark")
    (folder/"request-protocol.json").write_text(json.dumps(metadata,indent=2))
    rng=random.Random(20261023)
    prompts=[[rng.randrange(100,10000) for _ in range(15360)] for _ in range(32)]
    def request(index):
        payload=dict(params,prompt=prompts[index])
        req=urllib.request.Request("http://127.0.0.1:9031/v1/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
        start=time.monotonic()
        with urllib.request.urlopen(req,timeout=900) as response:result=json.load(response)
        usage=result["usage"]
        assert usage["prompt_tokens"]==15360 and usage["completion_tokens"]==512,usage
        return dict(index=index,duration=time.monotonic()-start,usage=usage)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        rows=list(pool.map(request,range(32)))
    (folder/"request-results.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(diagnostic_completed=len(rows))),flush=True)


def main(args):
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1",9031))!=0,"An unowned service is running"
    args.out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    for k in list(env):
        if k.startswith("VLLM_FL_BIV150_"):env.pop(k)
    env.update(VLLM_PLUGINS="fl",PYTHONPATH=str(args.source),VLLM_SERVER_DEV_MODE="1",
        VLLM_DISABLE_COMPILE_CACHE="1",TORCHINDUCTOR_CACHE_DIR=str(args.out/"inductor-cache"),
        VLLM_FL_BIV150_NATIVE_W8A8="1" if args.native else "0",BIV150_NATIVE_W8A8_LIBRARY=str(args.library),
        BIV150_NATIVE_W8A8_MIN_ROWS="17",BIV150_NATIVE_W8A8_LAYERS="qkv_proj,o_proj,gate_up_proj",
        BIV150_NATIVE_W8A8_COMPRESSED_ONLY="1",BIV150_NATIVE_W8A8_SKIP_EDGE_LAYERS="4",
        BIV150_NATIVE_W8A8_FUSE_SILU="1",PYTHONUNBUFFERED="1")
    config=dict(profiler="torch",torch_profiler_dir=str(args.out/"trace"),
                torch_profiler_with_stack=False,torch_profiler_record_shapes=True,
                torch_profiler_with_flops=False,torch_profiler_with_memory=False,
                torch_profiler_use_gzip=True,torch_profiler_dump_cuda_time_total=True,
                ignore_frontend=True,delay_iterations=300,max_iterations=8)
    cmd=["vllm","serve","/workspace/MiniCPM5-2B","--port","9031",
         "--compilation-config",'{"cudagraph_mode":"NONE"}',
         "--served-model-name","minicpm","--gpu-memory-utilization","0.85",
         "--max-model-len","131072","--profiler-config",json.dumps(config)]
    (args.out/"diagnostic-command.json").write_text(json.dumps(cmd,indent=2))
    manifest=dict(source=str(args.source),native=args.native,git_commit=subprocess.check_output(["git","-C",str(args.source),"rev-parse","HEAD"],text=True).strip(),git_status=subprocess.check_output(["git","-C",str(args.source),"status","--porcelain"],text=True),library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),purpose="steady decode diagnosis only; not a scored benchmark")
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    log=(args.out/"server.log").open("w")
    server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True,stdin=subprocess.DEVNULL)
    (args.out/"server.pid").write_text(str(server.pid))
    try:
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError("Diagnostic service exited")
            try:get_json("http://127.0.0.1:9031/v1/models");break
            except Exception:time.sleep(3)
        else:raise TimeoutError("Diagnostic service not ready")
        benchmark(args.out,"warm",4096,128,8,930001)
        capacity=re.search(r"GPU KV cache size: ([0-9,]+) tokens",(args.out/"server.log").read_text())
        assert capacity is not None,"KV capacity missing"
        tokens=int(capacity.group(1).replace(",",""))
        assert tokens>=507904,f"Insufficient KV capacity for the intended 32-sequence profile: {tokens}"
        print(json.dumps(dict(kv_capacity=tokens,diagnostic_only=True)),flush=True)
        post("/start_profile")
        diagnostic(args.out)
        post("/stop_profile")
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL);server.wait()
        log.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--native",action="store_true")
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
