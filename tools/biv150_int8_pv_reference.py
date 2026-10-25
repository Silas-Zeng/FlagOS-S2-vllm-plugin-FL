"""Independent scalar algebra for BF16 K, symmetric INT8 V and blockwise INT8 PV.

This is a deliberately approximate attention algorithm, not a model accuracy claim.
The closed reference quantizes globally normalized weighted probabilities. The
streaming reference quantizes before normalization, combines segments explicitly,
and shares no online recurrence implementation with a device kernel.
"""
import math


def blocks(length, segments, tile=32, page=16):
    span=((length+segments*page-1)//(segments*page))*page
    for segment in range(segments):
        lower=segment*span
        upper=min(lower+span,length)
        yield segment,[(p,min(p+tile,upper)) for p in range(lower,upper,tile)]


def quantize_values(values):
    largest=max((abs(v) for v in values),default=0.)
    assert math.isfinite(largest)
    scale=max(largest,1e-12)/127
    return [max(-127,min(127,(math.floor(abs(v/scale)+.5) if v>=0 else -math.floor(abs(v/scale)+.5)))) for v in values],scale


def closed(scores, values, scales, segments, tile=32):
    if not scores:return [0.]*len(values[0]) if values else []
    peak=max(scores);weights=[math.exp(s-peak) for s in scores]
    denominator=sum(weights);weights=[p/denominator for p in weights]
    result=[0.]*len(values[0]);codes=[]
    for segment,spans in blocks(len(scores),segments,tile):
        for start,end in spans:
            weighted=[weights[i]*scales[i] for i in range(start,end)]
            scale=max(weighted)/127
            if not scale:continue
            probability=[max(0,min(127,math.floor(p/scale+.5))) for p in weighted]
            codes.extend(probability)
            for channel in range(len(result)):
                result[channel]+=sum(q*values[i][channel] for q,i in zip(probability,range(start,end)))*scale
    return result


def streaming(scores, values, scales, segments, tile=32):
    if not scores:return [0.]*len(values[0]) if values else []
    partials=[]
    for segment,spans in blocks(len(scores),segments,tile):
        if not spans:continue
        peak=-math.inf;denominator=0.;accumulator=[0.]*len(values[0])
        for start,end in spans:
            new_peak=max(peak,max(scores[start:end]));alpha=math.exp(peak-new_peak)
            probability=[math.exp(scores[i]-new_peak) for i in range(start,end)]
            denominator=denominator*alpha+sum(probability)
            weighted=[p*scales[i] for p,i in zip(probability,range(start,end))]
            scale=max(max(weighted),1e-20)/127
            quantized=[max(0,min(127,math.floor(p/scale+.5))) for p in weighted]
            accumulator=[old*alpha+sum(q*values[i][d] for q,i in zip(quantized,range(start,end)))*scale
                         for d,old in enumerate(accumulator)]
            peak=new_peak
        partials.append((peak,denominator,accumulator))
    peak=max(p[0] for p in partials);denominator=0.;result=[0.]*len(values[0])
    for maximum,total,value in partials:
        alpha=math.exp(maximum-peak);denominator+=total*alpha
        result=[a+b*alpha for a,b in zip(result,value)]
    return [v/denominator for v in result]
