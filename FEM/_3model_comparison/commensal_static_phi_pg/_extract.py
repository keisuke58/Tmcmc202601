from __future__ import print_function
import sys
import json

try:
    from odbAccess import openOdb
except ImportError:
    sys.exit("Must run with abq2024 python")
import os

odb_file = [f for f in os.listdir(".") if f.endswith(".odb")][0]
odb = openOdb(odb_file, readOnly=True)
step = odb.steps[list(odb.steps.keys())[-1]]
frame = step.frames[-1]
result = {"job": odb_file}
if "S" in frame.fieldOutputs:
    s = frame.fieldOutputs["S"]
    mises_vals = [float(v.mises) for v in s.values if v.mises is not None]
    result["mises_max"] = max(mises_vals) if mises_vals else 0
    result["mises_mean"] = sum(mises_vals) / len(mises_vals) if mises_vals else 0
if "U" in frame.fieldOutputs:
    u = frame.fieldOutputs["U"]
    umags = [float(v.magnitude) for v in u.values]
    result["disp_max"] = max(umags) if umags else 0
    result["disp_mean"] = sum(umags) / len(umags) if umags else 0
odb.close()
out = odb_file.replace(".odb", "_stress.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
