import json

with open('IDM_VTON_Worker.ipynb', 'r') as f:
    d = json.load(f)

src = d['cells'][2]['source']
new_src = []
for line in src:
    if "content.replace('torch_dtype=torch.float16,', 'torch_dtype=torch.float16, low_cpu_mem_usage=True,')" in line:
        new_src.append("if 'low_cpu_mem_usage=True' not in content:\n")
        new_src.append("    " + line)
    else:
        new_src.append(line)

d['cells'][2]['source'] = new_src

with open('IDM_VTON_Worker.ipynb', 'w') as f:
    json.dump(d, f, indent=2)
