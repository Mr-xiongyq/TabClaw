import json
from pathlib import Path


def load_settings():
    settings_path = Path(__file__).parent / "setting.txt"
    settings = {}
    with open(settings_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                settings[key.strip()] = value.strip().strip('"').strip("'")
    return settings


def _parse_json_setting(raw: str, setting_name: str):
    if raw == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Exception(f"Invalid JSON format for {setting_name}") from exc


_s = load_settings()
API_KEY = _s.get("API_KEY", "")
BASE_URL = _s.get("BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = _s.get("DEFAULT_MODEL", "deepseek-ai/DeepSeek-V3")
DEFAULT_MODEL_EXTRA_JSON = _s.get("DEFAULT_MODEL_EXTRA_JSON", "")
DEFAULT_MODEL_EXTRA_PARAMS = _parse_json_setting(
    DEFAULT_MODEL_EXTRA_JSON, "DEFAULT_MODEL_EXTRA_JSON"
)

# A dedicated vision-capable model can be used for image -> HTML conversion.
VISION_MODEL = _s.get("VISION_MODEL", DEFAULT_MODEL)
VISION_MODEL_EXTRA_JSON = _s.get("VISION_MODEL_EXTRA_JSON", DEFAULT_MODEL_EXTRA_JSON)
VISION_MODEL_EXTRA_PARAMS = _parse_json_setting(
    VISION_MODEL_EXTRA_JSON, "VISION_MODEL_EXTRA_JSON"
)
