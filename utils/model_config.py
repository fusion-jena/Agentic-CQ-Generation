import json
import re
from functools import lru_cache
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.json"


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


def get_model(pipeline: str, role: str, default: str = "deepseek-r1:32b") -> str:
    """Return the model name for a given pipeline and agent role."""
    return _load().get(pipeline, {}).get(role, default)


def get_config(pipeline: str, key: str, default=None):
    """Return any non-model config value (e.g. integers like top_k) for a pipeline."""
    return _load().get(pipeline, {}).get(key, default)


def model_slug(model_name: str) -> str:
    """Auto-derive a short filename-safe code from any model name.

    Algorithm:
      - Split on ':' into name_part and tag.
      - Each '-'-segment of name_part: take first 2 chars if ≥3 letters, else keep;
        preserve trailing digit groups (e.g. 'llama4' → 'll4', 'r1' → 'r1').
      - Tag: drop 'latest'; size tags (start with digit) keep first segment only
        ('235b-a22b' → '235b'); named variants truncated to 3 chars ('scout' → 'sco').

    Examples: deepseek-r1:32b → der1-32b | llama4:scout → ll4-sco | bge-m3:latest → bgm3
    """
    name, _, tag = model_name.partition(":")

    def _abbr(seg: str) -> str:
        m = re.match(r'^([a-zA-Z]*)(\d*)$', seg)
        if not m:
            return seg[:2]
        letters, numbers = m.group(1), m.group(2)
        return (letters[:2] if len(letters) >= 3 else letters) + numbers

    name_abbr = "".join(_abbr(s) for s in name.split("-"))

    if not tag or tag == "latest":
        return name_abbr
    if tag[0].isdigit():
        tag_abbr = tag.split("-")[0]      # '235b-a22b' → '235b'
    else:
        tag_abbr = tag[:3]                # 'scout' → 'sco'
    return f"{name_abbr}-{tag_abbr}"


def pipeline_slug(generator: str, validator: str, refiner: str) -> str:
    """Compact filename suffix for a generator/validator/refiner triple.
    Single slug when all three are the same; g/v/r prefixes when they differ.
    """
    g, v, r = model_slug(generator), model_slug(validator), model_slug(refiner)
    if g == v == r:
        return g
    return f"g-{g}_v-{v}_r-{r}"
