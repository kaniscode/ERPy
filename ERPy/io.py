from __future__ import annotations

import re


def parse_brainvision_header(vhdr_text: str) -> dict:
    info: dict[str, object] = {"channels": [], "resolutions": []}
    section = None
    for raw_line in vhdr_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if section == "Common Infos":
            if key == "NumberOfChannels":
                info["n_channels"] = int(value)
            elif key == "SamplingInterval":
                info["sfreq"] = 1_000_000.0 / float(value)
            elif key == "DataOrientation":
                info["orientation"] = value
            elif key == "DataFile":
                info["data_file"] = value
            elif key == "MarkerFile":
                info["marker_file"] = value
        elif section == "Binary Infos" and key == "BinaryFormat":
            info["binary_format"] = value
        elif section == "Channel Infos" and key.startswith("Ch"):
            parts = value.split(",")
            info["channels"].append(parts[0].replace("\\1", ","))
            try:
                info["resolutions"].append(float(parts[2]))
            except Exception:
                info["resolutions"].append(1.0)
    if info.get("orientation") not in {None, "MULTIPLEXED"}:
        raise ValueError(f"Only MULTIPLEXED BrainVision data are supported, got {info.get('orientation')}")
    if info.get("binary_format") not in {None, "IEEE_FLOAT_32"}:
        raise ValueError(f"Only IEEE_FLOAT_32 BrainVision data are supported, got {info.get('binary_format')}")
    if info.get("n_channels") and len(info["channels"]) != info.get("n_channels"):
        raise ValueError("BrainVision channel count mismatch")
    return info


def clean_channel_label(label: str) -> str:
    text = str(label).strip()
    text = re.sub(r"^POL\s*", "", text, flags=re.I)
    text = re.sub(r"[-\s_]*Ref$", "", text, flags=re.I)
    text = text.replace(" ", "")
    return text
