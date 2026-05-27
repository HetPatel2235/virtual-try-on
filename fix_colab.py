import json

with open('IDM_VTON_Worker.ipynb', 'r') as f:
    d = json.load(f)

source = d['cells'][2]['source']

patch_code = """
if "base_path = 'yisol/IDM-VTON'" in content:
    content = content.replace("base_path = 'yisol/IDM-VTON'", "base_path = 'camenduru/IDM-VTON-F16'")
"""

source.insert(-3, patch_code)

with open('IDM_VTON_Worker.ipynb', 'w') as f:
    json.dump(d, f, indent=2)

print("Notebook patched successfully.")
