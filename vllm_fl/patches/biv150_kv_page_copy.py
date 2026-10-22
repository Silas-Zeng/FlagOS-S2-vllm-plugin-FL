"""独立KV成本门的连续页run复制；无安装、框架导入或现成offload路由。

只复制raw uint8页，不进行数值变换或分配GPU staging。调用者负责生产、
保存、复用和恢复之间的真实stream事件；返回值仅表示提交，绝非完成ack。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PagePlan:
    source_pages: int
    destination_pages: int
    page_bytes: int
    logical_pages: int
    runs: tuple[tuple[int, int, int], ...]

    def __post_init__(self):
        if (self.source_pages <= 0 or self.destination_pages <= 0
                or self.page_bytes not in (16384, 42 * 16384) or self.logical_pages <= 0):
            raise ValueError("仅支持已审BF16单层/42层raw页")
        source, destination = [], []
        for s, d, count in self.runs:
            if (any(type(x) is not int for x in (s, d, count)) or s < 0 or d < 0 or count <= 0
                    or s + count > self.source_pages or d + count > self.destination_pages):
                raise ValueError("页run越界或类型不符")
            source.extend(range(s, s + count))
            destination.extend(range(d, d + count))
        if (len(source) != self.logical_pages or len(set(source)) != len(source)
                or len(set(destination)) != len(destination)):
            raise ValueError("页映射必须一一对应，不能重叠写")


def make_plan(source_ids, destination_ids, source_pages, destination_pages, page_bytes):
    """保留逻辑次序，仅将src和dst同时连续的相邻条目合并；不排序。"""
    source_ids, destination_ids = tuple(source_ids), tuple(destination_ids)
    if not source_ids or len(source_ids) != len(destination_ids):
        raise ValueError("非空等长映射")
    if any(type(x) is not int for x in source_ids + destination_ids):
        raise ValueError("只接受Python整数页号")
    runs = []
    for source, destination in zip(source_ids, destination_ids):
        if runs and source == runs[-1][0] + runs[-1][2] and destination == runs[-1][1] + runs[-1][2]:
            s, d, n = runs[-1]
            runs[-1] = (s, d, n + 1)
        else:
            runs.append((source, destination, 1))
    return PagePlan(source_pages, destination_pages, page_bytes, len(source_ids), tuple(runs))


def submit_page_copy(source, destination, plan):
    """所有元数据校验在首次copy前完成；不等待、不更改buffer生命周期。"""
    if not isinstance(plan, PagePlan):
        raise TypeError("需要已验证PagePlan")
    if (tuple(source.shape) != (plan.source_pages, plan.page_bytes)
            or tuple(destination.shape) != (plan.destination_pages, plan.page_bytes)
            or str(source.dtype) != "torch.uint8" or str(destination.dtype) != "torch.uint8"
            or not source.is_contiguous() or not destination.is_contiguous()):
        raise ValueError("raw页shape/dtype/contiguous不符")
    if {source.device.type, destination.device.type} != {"cpu", "cuda"}:
        raise ValueError("只支持单设备CPU-CUDA传输")
    host = source if source.device.type == "cpu" else destination
    device = source if source.device.type == "cuda" else destination
    if not host.is_pinned() or device.device.index != 0:
        raise ValueError("需要pinned host和cuda:0")
    for s, d, count in plan.runs:
        destination.narrow(0, d, count).copy_(source.narrow(0, s, count), non_blocking=True)
    return {"submitted_bytes": plan.logical_pages * plan.page_bytes, "copy_calls": len(plan.runs),
            "completion_ack": False}
