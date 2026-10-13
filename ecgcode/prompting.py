"""Request parameters: system prompt, code-execution tool, the record's file and demographics."""
import copy
import hashlib

from .config import project_path
from .data import LABELS

FILE_ID_PLACEHOLDER = "{FILE_ID}"
EXAMPLE_PREFIX = "{EXAMPLE_FILE:"


def load_system_prompt(cfg):
    text = project_path(cfg["prompt"]["system"]).read_text()
    return text, hashlib.sha256(text.encode()).hexdigest()


def output_schema(statement_level=False):
    """One entry per record: id, a short findings note, five independent probabilities and, at
    statement level, a probability for each of PTB-XL's 44 diagnostic statements."""
    props = {"record_id": {"type": "string"}, "key_findings": {"type": "string"}}
    if statement_level:
        from .statements import STATEMENT_CLASSES
        codes = sorted(STATEMENT_CLASSES)
        props["statements"] = {"type": "object", "properties": {c: {"type": "number"} for c in codes},
                               "required": codes, "additionalProperties": False}
    props.update({f"p_{label}": {"type": "number"} for label in LABELS})
    item = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return {"type": "object", "properties": {"records": {"type": "array", "items": item}},
            "required": ["records"], "additionalProperties": False}


def describe_patient(age, sex):
    """PTB-XL codes ages above 89 as 300 and sex as 0 = male, 1 = female."""
    age_text = "90 or older" if age is None or age >= 90 else f"{int(age)} years"
    return f"age {age_text}, {'female' if int(sex) == 1 else 'male'}"


def user_content(anon, age, sex, include_demographics=True, example_files=()):
    lines = [f"Record {anon}. The attached file {anon}.csv holds the raw 10-second 12-lead recording."]
    if include_demographics:
        lines.append(f"Patient: {describe_patient(age, sex)}.")
    if example_files:
        lines.append("The labelled example files described in the instructions are attached as well.")
    lines.append("Classify it as described, then answer with the JSON object only.")
    blocks = [{"type": "text", "text": " ".join(lines)},
              {"type": "container_upload", "file_id": FILE_ID_PLACEHOLDER}]
    blocks += [{"type": "container_upload", "file_id": f"{EXAMPLE_PREFIX}{name}}}"} for name in example_files]
    return blocks


def request_params(cfg, system_text, anon, age, sex, example_files=()):
    """Parameters for one record; the file id is filled in at submit time (with_file_id)."""
    model = cfg["model"]
    params = {
        "model": model["id"],
        "max_tokens": int(model["max_tokens"]),
        "system": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_content(anon, age, sex,
                                                              cfg["input"]["include_demographics"],
                                                              example_files)}],
        "tools": [{"type": model["code_tool"], "name": "code_execution"}],
        "output_config": {"effort": model["effort"],
                          "format": {"type": "json_schema",
                                     "schema": output_schema(cfg.get("output", {}).get("statement_level", False))}},
    }
    if model.get("thinking_display"):
        params["thinking"] = {"type": "adaptive", "display": model["thinking_display"]}
    return params


def with_file_id(params, file_id, files=None):
    """Copy of params with the record's uploaded file id, and the example files' ids from `files`
    (keys "examples/<name>")."""
    out = copy.deepcopy(params)
    for block in out["messages"][0]["content"]:
        if block.get("type") != "container_upload":
            continue
        if block["file_id"] == FILE_ID_PLACEHOLDER:
            block["file_id"] = file_id
        elif block["file_id"].startswith(EXAMPLE_PREFIX):
            name = block["file_id"][len(EXAMPLE_PREFIX):-1]
            if not files or f"examples/{name}" not in files:
                raise SystemExit(f"example file {name} is not uploaded yet; run `upload` first (free)")
            block["file_id"] = files[f"examples/{name}"]
    return out
