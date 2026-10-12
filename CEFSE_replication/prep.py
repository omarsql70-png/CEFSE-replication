"""Data preparation for the CEFSE line-level evaluation.

Reads the public file-level / line-level defect datasets of Wattanakriengkrai et al. (2022),
strips comments, tokenizes Java code lines and marks ground-truth defective lines.
"""
import os, re, pickle, difflib
import pandas as pd

DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext", "line-level-defect-prediction", "Dataset")

# Chronological release order per project (train = 1st, validation = 2nd, test = remaining)
PROJECTS = {
    "activemq": ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0"],
    "camel": ["camel-1.4.0", "camel-2.9.0", "camel-2.10.0", "camel-2.11.0"],
    "derby": ["derby-10.2.1.6", "derby-10.3.1.4", "derby-10.5.1.1"],
    "groovy": ["groovy-1_5_7", "groovy-1_6_BETA_1", "groovy-1_6_BETA_2"],
    "hbase": ["hbase-0.94.0", "hbase-0.95.0", "hbase-0.95.2"],
    "hive": ["hive-0.9.0", "hive-0.10.0", "hive-0.12.0"],
    "jruby": ["jruby-1.1", "jruby-1.4.0", "jruby-1.5.0", "jruby-1.7.0.preview1"],
    "lucene": ["lucene-2.3.0", "lucene-2.9.0", "lucene-3.0.0", "lucene-3.1"],
    "wicket": ["wicket-1.3.0-incubating-beta-1", "wicket-1.3.0-beta2", "wicket-1.5.3"],
}

TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][A-Za-z0-9_$]*|\d[\w.]*'
                      r'|==|!=|<=|>=|&&|\|\||\+\+|--|->|::|<<|>>|[^\sA-Za-z0-9_]')
JAVA_KW = set("""abstract assert boolean break byte case catch char class const continue default do double else enum
extends final finally float for goto if implements import instanceof int interface long native new package private
protected public return short static strictfp super switch synchronized this throw throws transient try void volatile
while true false null var String Object Override""".split())


def strip_comments(lines):
    out, in_block = [], False
    for s in lines:
        res, i = "", 0
        while i < len(s):
            if in_block:
                j = s.find("*/", i)
                if j < 0:
                    i = len(s)
                else:
                    in_block, i = False, j + 2
            else:
                j1, j2 = s.find("/*", i), s.find("//", i)
                if j2 != -1 and (j1 == -1 or j2 < j1):
                    res += s[i:j2]; break
                if j1 != -1:
                    res += s[i:j1]; in_block, i = True, j1 + 2
                else:
                    res += s[i:]; break
        out.append(res)
    return out


def tokenize(line):
    toks = []
    for t in TOKEN_RE.findall(line):
        if t[0] == '"':
            toks.append("<STR>")
        elif t[0] == "'":
            toks.append("<CHR>")
        elif t[0].isdigit():
            toks.append("<NUM>")
        else:
            toks.append(t)
    return toks


def load_release(rel):
    f = pd.read_csv(os.path.join(DS, "File-level", f"{rel}_ground-truth-files_dataset.csv"), encoding="latin-1")
    l = pd.read_csv(os.path.join(DS, "Line-level", f"{rel}_defective_lines_dataset.csv"), encoding="latin-1")
    bad = {}
    for fn, ln in zip(l.File, l.Line_number):
        bad.setdefault(fn, set()).add(int(ln))
    files = []
    for fn, bug, src in zip(f.File, f.Bug, f.SRC):
        raw = str(src).split("\n")
        code = strip_comments(raw)
        line_no, toks, text = [], [], []
        for i, c in enumerate(code):
            if c.strip():
                tk = tokenize(c)
                if tk:
                    line_no.append(i + 1); toks.append(tk[:50]); text.append(raw[i])
        files.append(dict(file=fn, bug=bool(bug), line_no=line_no, toks=toks, text=text,
                          defective=set(bad.get(fn, set())) & set(line_no), raw=raw))
    return files


def changed_lines(prev_raw, cur_raw):
    """1-based line numbers of cur_raw that are new or modified relative to prev_raw (GNU diff, Myers)."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".a", encoding="utf-8") as fa, \
         tempfile.NamedTemporaryFile("w", delete=False, suffix=".b", encoding="utf-8") as fb:
        fa.write("\n".join(s.strip() for s in prev_raw) + "\n"); fb.write("\n".join(s.strip() for s in cur_raw) + "\n")
    r = subprocess.run(["diff", "--unchanged-line-format=", "--old-line-format=", "--new-line-format=%dn\n",
                        fa.name, fb.name], capture_output=True, text=True)
    os.unlink(fa.name); os.unlink(fb.name)
    return {int(x) for x in r.stdout.split()}


def main():
    out = {}
    for proj, rels in PROJECTS.items():
        data = {r: load_release(r) for r in rels}
        for k in range(1, len(rels)):
            prev = {d["file"]: d["raw"] for d in data[rels[k - 1]]}
            for d in data[rels[k]]:
                if d["file"] in prev:
                    same = prev[d["file"]] == d["raw"]
                    d["changed"] = set() if same else changed_lines(prev[d["file"]], d["raw"])
                    d["is_new"] = False
                else:
                    d["changed"] = set(d["line_no"]); d["is_new"] = True
        for d in data[rels[0]]:
            d["changed"] = set(); d["is_new"] = None
        for r in rels:
            for d in data[r]:
                del d["raw"]
        print(proj, {r: (len(v), sum(x["bug"] for x in v)) for r, v in data.items()}, flush=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"data_{proj}.pkl"), "wb") as fh:
            pickle.dump(data, fh)


if __name__ == "__main__":
    main()
