from __future__ import annotations

from collections.abc import Iterable

from tasks import OUTSIDE


TEMPLATE_SLOT = "{P}"
LEADING_SEPARATOR = "|"


def _split_label(label: str) -> tuple[str, str]:
    if label.startswith("^"):
        value = label[1:]
        if LEADING_SEPARATOR in value:
            return tuple(value.split(LEADING_SEPARATOR, 1))  # type: ignore[return-value]
        return value, ""
    return "", "" if label == OUTSIDE else label


def _restore_label(leading: str, trailing: str) -> str:
    if leading and trailing:
        return f"^{leading}{LEADING_SEPARATOR}{trailing}"
    if leading:
        return "^" + leading
    return trailing or OUTSIDE


def select_punctuation(label: str, allowed: Iterable[str]) -> str:
    """从联合标签中抽取指定标点，并保留它们在原标签中的次序。"""
    allowed_set = set(allowed)
    leading, trailing = _split_label(label)
    selected_leading = "".join(mark for mark in leading if mark in allowed_set)
    selected_trailing = "".join(mark for mark in trailing if mark in allowed_set)
    return _restore_label(selected_leading, selected_trailing)


def structural_template(
    label: str,
    pause_punctuation: Iterable[str],
    structural_punctuation: Iterable[str],
) -> str:
    """
    把结构标点编码为可还原相对次序的模板标签。

    例如 ``。”`` -> ``{P}”``，``”。“`` -> ``”{P}“``。这避免两阶段
    独立预测后丢失标点的原始先后次序。
    """
    pause = set(pause_punctuation)
    structural = set(structural_punctuation)
    leading, trailing = _split_label(label)
    selected_leading = "".join(mark for mark in leading if mark in structural)
    if not selected_leading and not any(mark in structural for mark in trailing):
        return OUTSIDE
    parts: list[str] = []
    for mark in trailing:
        if mark in structural:
            parts.append(mark)
        elif mark in pause:
            parts.append(TEMPLATE_SLOT)
    return _restore_label(selected_leading, "".join(parts))


def merge_pause_labels(*labels: str) -> str:
    """合并句间与句内停顿标签；常规语料中二者互斥。"""
    values = [_split_label(label) for label in labels]
    leading = "".join(value[0] for value in values)
    trailing = "".join(value[1] for value in values)
    return _restore_label(leading, trailing)


def apply_structural_template(template: str, pause_label: str) -> str:
    """将上游停顿预测填回结构标点模板，得到最终联合标签。"""
    template_leading, template_trailing = _split_label(template)
    pause_leading, pause_trailing = _split_label(pause_label)
    merged_leading = template_leading + pause_leading
    if not template_trailing:
        return _restore_label(merged_leading, pause_trailing)
    if TEMPLATE_SLOT in template_trailing:
        slot_count = template_trailing.count(TEMPLATE_SLOT)
        if slot_count == 1:
            replacements = [pause_trailing]
        else:
            replacements = list(pause_trailing[: slot_count - 1])
            replacements.extend([""] * (slot_count - 1 - len(replacements)))
            replacements.append(pause_trailing[slot_count - 1 :])
        merged = template_trailing
        for replacement in replacements:
            merged = merged.replace(TEMPLATE_SLOT, replacement, 1)
    else:
        # 模型在同一间隙同时预测了独立结构标点和停顿标点时，
        # 以语料中更常见的“停顿在前”作为确定性回退。
        merged = pause_trailing + template_trailing
    return _restore_label(merged_leading, merged)
