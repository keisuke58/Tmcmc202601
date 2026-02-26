from odbAccess import openOdb
import sys
import os
import math


def _norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def _analyze_instance(odb, frame, inst_key):
    insts = odb.rootAssembly.instances
    if inst_key not in insts:
        return None
    region = insts[inst_key]
    fo = frame.fieldOutputs
    res = {}
    if "S" in fo:
        s_field = fo["S"].getSubset(region=region)
        values = s_field.values
        if values:
            res["max_mises"] = max(v.mises for v in values)
        else:
            res["max_mises"] = None
    else:
        res["max_mises"] = None
    if "U" in fo:
        u_field = fo["U"].getSubset(region=region)
        values = u_field.values
        if values:
            res["max_u"] = max(_norm(v.data) for v in values)
        else:
            res["max_u"] = None
    else:
        res["max_u"] = None
    if "RF" in fo:
        rf_field = fo["RF"].getSubset(region=region)
        values = rf_field.values
        if values:
            sx = sy = sz = 0.0
            for v in values:
                d = v.data
                sx += d[0]
                sy += d[1]
                sz += d[2]
            res["sum_rf"] = (sx, sy, sz)
            res["total_rf"] = _norm((sx, sy, sz))
        else:
            res["sum_rf"] = None
            res["total_rf"] = None
    else:
        res["sum_rf"] = None
        res["total_rf"] = None
    return res


def _analyze_step(odb, step_name, focus_instances=None):
    if step_name not in odb.steps:
        return None
    step = odb.steps[step_name]
    if not step.frames:
        return None
    frame = step.frames[-1]
    out = {}
    insts = odb.rootAssembly.instances
    if focus_instances:
        keys = [k for k in focus_instances if k in insts]
    else:
        keys = sorted(insts.keys())
    for inst_key in keys:
        res = _analyze_instance(odb, frame, inst_key)
        if res is not None:
            out[inst_key] = res
    return out


def _find_default_odb_paths():
    candidates = [
        "OJ_ThreeTooth_WithBiofilm.odb",
        "OJ_AllLower_WithBiofilm.odb",
    ]
    found = []
    for p in candidates:
        if os.path.exists(p):
            found.append(p)
    return found


def main(argv):
    if argv:
        odb_paths = argv
    else:
        odb_paths = _find_default_odb_paths()
    if not odb_paths:
        sys.stderr.write("No ODB paths given and no default ODBs found.\n")
        return 1
    for path in odb_paths:
        if not os.path.exists(path):
            sys.stderr.write("ODB not found: %s\n" % path)
            continue
        print("ODB:", path)
        odb = openOdb(path=path)
        for step_name, step in odb.steps.items():
            res = _analyze_step(odb, step_name)
            if not res:
                continue
            print(" Step:", step_name)
            for inst_key, vals in res.items():
                print("  Instance:", inst_key)
                mm = vals.get("max_mises")
                if mm is not None:
                    print("   max S_Mises:", mm)
                mu = vals.get("max_u")
                if mu is not None:
                    print("   max |U|:", mu)
                sr = vals.get("sum_rf")
                tr = vals.get("total_rf")
                if sr is not None and tr is not None:
                    print("   sum RF:", sr, "  |RF|:", tr)
        odb.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
