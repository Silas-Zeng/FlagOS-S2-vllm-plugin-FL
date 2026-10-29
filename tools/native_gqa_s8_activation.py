"""Validate N30/S8 direct producer receipts without changing the S2/S4 reader."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

PREFIX = "BIV150_NATIVE_S8_GQA_"
S8_SHA = "b2cd075343fe385603db6b1daf79e0f55678fde66e56a7f7b05d33f96cd3aea6"
GQA_S2_S4_SHA = "c8a725daa29f7fb077d15f04dd9a3be8a419e12e857cc46b4b62870d4f97421b"


def capture(log_path, expected_s8_sha=S8_SHA):
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    installs, records, errors = [], {"SELECTED": [], "CAPTURED": []}, []
    for line in text.splitlines():
        if "BIV150_NATIVE_GQA_INSTALL " in line:
            payload = line.split("BIV150_NATIVE_GQA_INSTALL ", 1)[1]
            try:
                obj = json.loads(payload)
                if obj.get("s8_enabled"):
                    installs.append(obj)
            except (ValueError, TypeError):
                errors.append(line)
        if PREFIX not in line:
            continue
        kind, _, payload = line.split(PREFIX, 1)[1].partition(" ")
        if kind not in records:
            errors.append(line)
            continue
        try:
            records[kind].append(json.loads(payload))
        except (ValueError, TypeError):
            errors.append(line)

    def valid_record(r):
        return (isinstance(r, dict) and r.get("rows") == 30 and r.get("segments") == 8
                and r.get("grid") == [30, 2, 8] and r.get("threads") == 64
                and r.get("q_stride") == [2048, 128, 1]
                and type(r.get("device_index")) is int and r["device_index"] >= 0
                and r.get("q_shape") == [30, 16, 128]
                and r.get("q_dtype") == r.get("k_dtype") == r.get("v_dtype") == "torch.bfloat16"
                and isinstance(r.get("k_shape"), list) and isinstance(r.get("v_shape"), list)
                and len(r["k_shape"]) == len(r["v_shape"]) == 4
                and type(r["k_shape"][0]) is int and r["k_shape"][0] > 0
                and r["k_shape"][0] == r["v_shape"][0]
                and r["k_shape"][1:] == r["v_shape"][1:] == [16, 2, 128]
                and r.get("lengths_shape") == [30]
                and r.get("lengths_dtype") == r.get("table_dtype") == "torch.int32"
                and isinstance(r.get("table_shape"), list) and r["table_shape"][0] == 30
                and r["table_shape"][1] >= 1024
                and isinstance(r.get("table_stride"), list) and r["table_stride"][1] == 1
                and isinstance(r.get("partial_shape"), list)
                and r["partial_shape"][0] >= 30 and r["partial_shape"][1:] == [16, 8, 128]
                and r.get("partial_dtype") == r.get("maximum_dtype") == r.get("summation_dtype") == "torch.float32"
                and isinstance(r.get("partial_stride"), list)
                and r["partial_stride"][2:] == [128, 1]
                and isinstance(r.get("maximum_shape"), list)
                and r["maximum_shape"][0] >= 30 and r["maximum_shape"][1:] == [16, 8]
                and r.get("summation_shape") == r.get("maximum_shape")
                and isinstance(r.get("maximum_stride"), list)
                and r["maximum_stride"][2] == 1
                and r.get("summation_stride") == r.get("maximum_stride")
                and r.get("page_size") == 16 and r.get("num_warps") == r.get("num_stages") == 1
                and isinstance(r.get("softmax_scale_log2e"), (int, float))
                and math.isfinite(r["softmax_scale_log2e"]) and r["softmax_scale_log2e"] > 0
                and r.get("k_stride") == r.get("v_stride") == [8192, 256, 128, 1]
                and r.get("library_sha256") == GQA_S2_S4_SHA
                and r.get("s8_library_sha256") == expected_s8_sha
                and r.get("s8_variant") == "shared4k"
                and r.get("overlap_v") is True and r.get("extra_KV_bytes") == 0)

    selected = [r for r in records["SELECTED"] if valid_record(r)]
    captured = [r for r in records["CAPTURED"] if valid_record(r)]
    install_ok = (len(installs) == 1 and installs[0].get("installed") is True
                  and installs[0].get("s8_enabled") is True
                  and installs[0].get("s8_supported_shapes") == [[30, 8]]
                  and installs[0].get("s8_library_sha256") == expected_s8_sha
                  and installs[0].get("s8_variant") == "shared4k"
                  and installs[0].get("library_sha256") == GQA_S2_S4_SHA
                  and installs[0].get("overlap_v") is True
                  and installs[0].get("extra_KV_bytes") == 0
                  and installs[0].get("compile_at_runtime") is False)
    any_activation = bool(selected or captured)
    invalid = (not install_ok or not any_activation or errors
               or len(selected) != len(records["SELECTED"])
               or len(captured) != len(records["CAPTURED"]))
    mode = "captured" if captured else "eager_only" if selected else "not_exercised"
    return {"valid": not bool(invalid), "covered": not bool(invalid), "activation_mode": mode,
            "install": installs, "selected": selected, "captured": captured,
            "errors": errors, "expected_s8_sha256": expected_s8_sha,
            "scope": "N30/S8 actual selected/captured marker only; not accuracy, request coverage or official score"}



def official_first_warm_from_text(text, expected_s8_sha=S8_SHA):
    """Classify natural first-warm activation; missing shape is transparent, never forced."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="s8-first-warm-") as td:
        log_path=Path(td)/"worker.log"
        log_path.write_text(text,encoding="utf-8")
        result=capture(log_path,expected_s8_sha)
    install_ok=(len(result['install'])==1
        and result['install'][0].get('installed') is True
        and result['install'][0].get('s8_enabled') is True
        and result['install'][0].get('s8_supported_shapes')==[[30,8]]
        and result['install'][0].get('s8_library_sha256')==expected_s8_sha
        and result['install'][0].get('s8_variant')=='shared4k'
        and result['install'][0].get('library_sha256')==GQA_S2_S4_SHA
        and result['install'][0].get('overlap_v') is True
        and result['install'][0].get('extra_KV_bytes')==0
        and result['install'][0].get('compile_at_runtime') is False)
    result['install_valid']=install_ok
    has_s8_calls=bool(result['selected'] or result['captured'])
    if not has_s8_calls and not result['errors'] and install_ok:
        status='not_exercised'
    elif result['valid'] and install_ok:
        status='exercised_valid'
    else:
        status='activation_invalid'
    result.update(status=status,official_qualification_pass=(status=='exercised_valid'),
        policy='首暖沿官方负载自然触发；缺少S8 selected/captured时记录not_exercised，不修改负载或预热强制触发。')
    return result


def official_first_warm(log_path, expected_s8_sha=S8_SHA):
    """Classify a worker log without changing the workload."""
    return official_first_warm_from_text(Path(log_path).read_text(encoding="utf-8",errors="replace"),expected_s8_sha)
    has_s8_calls=bool(result['selected'] or result['captured'])
    has_markers=bool(result['errors'])
    if not has_s8_calls and not has_markers:
        status='not_exercised'
    elif result['valid']:
        status='exercised_valid'
    else:
        status='activation_invalid'
    result.update(status=status,official_qualification_pass=(status=='exercised_valid'),
        policy='首暖沿官方负载自然触发；缺少S8 selected/captured时记录not_exercised，不修改负载或预热强制触发。')
    return result

def install_identity(log_path, expected_s8_sha=S8_SHA):
    """Prebenchmark S8 install identity only; no natural activation is inferred."""
    text=Path(log_path).read_text(encoding="utf-8",errors="replace")
    rows=[];errors=[]
    for line in text.splitlines():
        if "BIV150_NATIVE_GQA_INSTALL " not in line:continue
        try:
            value=json.loads(line.split("BIV150_NATIVE_GQA_INSTALL ",1)[1])
            if not isinstance(value,dict):raise ValueError("install record is not a dict")
            rows.append(value)
        except (ValueError,TypeError) as error:errors.append(dict(line=line,error=repr(error)))
    valid=(len(rows)==1 and not errors and rows[0].get("installed") is True
        and rows[0].get("s8_enabled") is True and rows[0].get("s8_supported_shapes")==[[30,8]]
        and rows[0].get("s8_library_sha256")==expected_s8_sha
        and rows[0].get("s8_variant")=="shared4k"
        and rows[0].get("library_sha256")==GQA_S2_S4_SHA
        and rows[0].get("overlap_v") is True and rows[0].get("extra_KV_bytes")==0
        and rows[0].get("compile_at_runtime") is False)
    return dict(valid=bool(valid),install=rows,errors=errors,
        status="installed" if valid else "install_invalid",
        limitation="安装回执只证明运行时启用与两库身份；不证明官方负载实际执行S8。")

def validate_first_round(log_path, expected_s8_sha=S8_SHA):
    result = capture(log_path, expected_s8_sha)
    if not result["valid"]:
        raise RuntimeError("N30/S8 首轮实际选择/capture回执未闭合")
    return result


def self_test():
    record = dict(rows=30, segments=8, grid=[30, 2, 8], threads=64,
                  device_index=0, q_shape=[30, 16, 128],
                  q_dtype="torch.bfloat16", k_dtype="torch.bfloat16", v_dtype="torch.bfloat16",
                  q_stride=[2048, 128, 1], k_stride=[8192, 256, 128, 1],
                  v_stride=[8192, 256, 128, 1], library_sha256=GQA_S2_S4_SHA,
                  s8_library_sha256=S8_SHA,
                  s8_variant="shared4k",
                  overlap_v=True, extra_KV_bytes=0)
    record.update(k_shape=[34268, 16, 2, 128], v_shape=[34268, 16, 2, 128],
                  lengths_shape=[30], table_shape=[30, 8192], table_stride=[8192, 1],
                  lengths_dtype="torch.int32", table_dtype="torch.int32",
                  partial_shape=[30, 16, 8, 128], partial_dtype="torch.float32", partial_stride=[16384, 1024, 128, 1],
                  maximum_shape=[30, 16, 8], maximum_stride=[128, 8, 1],
                  maximum_dtype="torch.float32",
                  summation_shape=[30, 16, 8], summation_stride=[128, 8, 1],
                  summation_dtype="torch.float32",
                  page_size=16, num_warps=1, num_stages=1, softmax_scale_log2e=0.127)
    install = dict(installed=True, s8_enabled=True, s8_supported_shapes=[[30, 8]],
                   s8_library_sha256=S8_SHA, overlap_v=True, extra_KV_bytes=0,
                   s8_variant="shared4k",
                   compile_at_runtime=False, library_sha256=GQA_S2_S4_SHA)
    with __import__("tempfile").TemporaryDirectory(prefix="s8-marker-") as td:
        path = Path(td) / "worker.log"
        path.write_text("BIV150_NATIVE_GQA_INSTALL " + json.dumps(install) + "\n"
                        + PREFIX + "SELECTED " + json.dumps(record) + "\n"
                        + PREFIX + "CAPTURED " + json.dumps(record) + "\n", encoding="utf-8")
        assert capture(path)["valid"]
        path.write_text("BIV150_NATIVE_GQA_INSTALL " + json.dumps(install) + "\n"
                        + PREFIX + "SELECTED " + json.dumps({**record, "q_stride": [2560, 128, 1]}) + "\n"
                        + PREFIX + "CAPTURED " + json.dumps(record) + "\n", encoding="utf-8")
        assert not capture(path)["valid"]
        path.write_text("BIV150_NATIVE_GQA_INSTALL " + json.dumps(install) + "\n"
                        + PREFIX + "SELECTED " + json.dumps(record) + "\n",encoding="utf-8")
        eager=capture(path)
        assert eager["valid"] and eager["activation_mode"]=="eager_only"
        path.write_text("BIV150_NATIVE_GQA_INSTALL " + json.dumps({**install, "s8_library_sha256": "wrong"}) + "\n"
                        + PREFIX + "SELECTED " + json.dumps(record) + "\n"
                        + PREFIX + "CAPTURED " + json.dumps(record) + "\n", encoding="utf-8")
        assert not capture(path)["valid"]
    install_only="BIV150_NATIVE_GQA_INSTALL "+json.dumps({"installed":True,"s8_enabled":True,
        "s8_supported_shapes":[[30,8]],"s8_library_sha256":S8_SHA,
        "s8_variant":"shared4k",
        "library_sha256":GQA_S2_S4_SHA,"overlap_v":True,"extra_KV_bytes":0,
        "compile_at_runtime":False})+"\n"
    no_trigger=official_first_warm_from_text(install_only)
    assert no_trigger["status"]=="not_exercised" and not no_trigger["official_qualification_pass"]
    eager=official_first_warm_from_text(install_only+PREFIX+"SELECTED "+json.dumps(record)+"\n")
    assert eager["status"]=="exercised_valid" and eager["activation_mode"]=="eager_only"
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("log", nargs="?")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps({"status": "pass" if self_test() else "fail"}))
    elif args.log:
        print(json.dumps(capture(args.log), ensure_ascii=False, indent=2))
    else:
        parser.error("provide --self-test or a log")


