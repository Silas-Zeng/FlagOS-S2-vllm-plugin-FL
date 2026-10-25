"""Scalar checks for probability quantization and the independent reference."""
import json
import math
import random
from biv150_int8_pv_reference import blocks,closed,streaming,quantize_values


def main():
    random.seed(20261026);cases=0;largest_error=0.
    for length in [1,2,15,16,17,31,32,33,63,64,65,127,128,129,257]:
        for segments in [1,2,4,8,16]:
            covered=[p for _,spans in blocks(length,segments) for start,end in spans for p in range(start,end)]
            assert covered==list(range(length)),(length,segments)
            scores=[random.uniform(-8,8) for _ in range(length)];values=[];scales=[]
            for i in range(length):
                codes,scale=quantize_values([random.uniform(-3,3) for _ in range(8)])
                assert all(-127<=v<=127 for v in codes) and scale>0
                values.append(codes);scales.append(scale)
            a=closed(scores,values,scales,segments);b=streaming(scores,values,scales,segments)
            error=max(abs(x-y) for x,y in zip(a,b));largest_error=max(largest_error,error)
            assert error<2e-12,(length,segments,error)
            cases+=1
    assert math.floor(.5+.5)==1 and math.floor(1.5+.5)==2
    assert quantize_values([127.,.5,-.5])[0]==[127,1,-1]
    zero,scale=quantize_values([0.]*128);assert zero==[0]*128 and scale>0
    extremes=streaming([-200,200,-200],[[127],[-127],[127]],[.01]*3,1)
    assert math.isfinite(extremes[0])
    print(json.dumps({'passed':True,'gpu_executed':False,'cases':cases,'closed_global_vs_online_max_abs':largest_error,
                      'coverage':'page16 segments; token32 PV blocks; positive symmetric INT8 probability codes',
                      'model_accuracy':'not evaluated','throughput':'not evaluated',
                      'limitation':'FP64 algebra normal-domain equivalence; device FP32 rounding and approximate P/V require separate validation'},indent=2))


if __name__=='__main__':main()
