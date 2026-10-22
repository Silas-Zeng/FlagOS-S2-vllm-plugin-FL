"""Stop only an owned evaluation once 105-example accuracy cannot reach 0.95."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def command(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().split(bytes([0]))
    except FileNotFoundError:
        return []


def descends_from(pid, root):
    seen=set()
    while pid>1 and pid not in seen:
        if pid==root:
            return True
        seen.add(pid)
        try:
            status=Path(f"/proc/{pid}/status").read_text().splitlines()
            pid=int(next(x for x in status if x.startswith("PPid:")).split()[1])
        except (FileNotFoundError,StopIteration):
            return False
    return False


def main(args):
    needle=str(args.out).encode()
    driver=command(args.driver_pid)
    assert any(b"run_biv150_operator_arm.py" in x for x in driver) and needle in driver
    prior=None
    while command(args.driver_pid)==driver:
        files=list((args.out/"math500-level3").glob("*/reviews/minicpm/*.jsonl*"))
        if len(files)==1:
            by_index={}
            for line in files[0].read_text().splitlines():
                try:
                    row=json.loads(line)
                    by_index[row["index"]]=row
                except json.JSONDecodeError:
                    pass  # A writer may currently be appending the final line.
            wrong=[r["index"] for r in by_index.values()
                   if r["sample_score"]["score"]["value"].get("accuracy",0)<1]
            snapshot=dict(reviewed=len(by_index),wrong_indices=sorted(wrong),
                          best_possible_accuracy=(105-len(wrong))/105)
            if snapshot!=prior:
                print(json.dumps(snapshot),flush=True)
                prior=snapshot
            if len(by_index)==105:
                # Let the evaluator finish its report; the service driver owns
                # the final metric/coverage gate and cleanup.
                return
            if len(wrong)>=6:
                snapshot["reason"]="At least six wrong answers makes 0.95 impossible on 105 examples"
                (args.out/"early-rejection.json").write_text(json.dumps(snapshot,indent=2))
                targets=[]
                for proc in Path("/proc").iterdir():
                    if not proc.name.isdigit():
                        continue
                    pid=int(proc.name)
                    argv=command(pid)
                    # Exact evalscope argv and ancestry prevent matching an SSH
                    # shell containing this script's source or any other task.
                    if len(argv)>2 and argv[1].endswith(b"/evalscope") and argv[2]==b"eval" and (needle+b"/math500-level3") in argv and descends_from(pid,args.driver_pid):
                        targets.append(pid)
                if len(targets)!=1:
                    raise RuntimeError(f"Expected one owned evalscope, got {targets}; no signal sent")
                os.kill(targets[0],signal.SIGTERM)
                print(json.dumps(dict(early_rejected=True,evalscope_pid=targets[0])),flush=True)
                return
        time.sleep(10)
    print(json.dumps(dict(driver_finished=True)),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--driver-pid",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
