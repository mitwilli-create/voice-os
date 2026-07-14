#!/usr/bin/env python3
"""draft_long: run long-form text through Voice OS section-by-section so the
single-pass pipeline cannot silently compress or drop sections.

WHY: `python3 -m voice_os draft` on a whole long document rewrites it in one
pass and tends to compress hard (observed: ~2,000 -> ~1,090 words) and drop
entire sections. This wrapper splits the input on Markdown '## ' section
headers, voices each section independently through the real pipeline, preserves
non-prose ':::...:::' marker blocks (embeds, image/infographic specs) verbatim,
reassembles in order, and runs a TRUNCATION GUARD so compression can't pass
unnoticed.

Usage:
  python3 scripts/draft_long.py --file BRIEF.md --out FINAL.md \
      [--channel doc --audience external --situation standard \
       --goal persuade --stakes high --max-revisions 1] \
      [--title "..."] [--subtitle "..."] [--min-retention 0.70]

Exit codes: 0 = ok, 3 = truncation guard FAILED (inspect summary), 2 = usage error.
Requires ANTHROPIC_API_KEY in env (auto-loaded via ~/.zshenv). Offline mode fails the guard.
"""
import argparse, json, os, re, subprocess, sys, tempfile

def split_markers(body_lines):
    """Separate ':::...:::' blocks from prose. Returns (prose_text, [blocks])."""
    prose, markers, buf, in_marker = [], [], [], False
    for l in body_lines:
        s = l.strip()
        if not in_marker and s.startswith(":::"):
            in_marker = True; buf = [l]; continue
        if in_marker:
            buf.append(l)
            if s == ":::":
                markers.append("\n".join(buf)); in_marker = False
            continue
        prose.append(l)
    if in_marker:  # unterminated block; keep as prose so nothing is lost
        prose.extend(buf)
    return "\n".join(prose).strip(), markers

def parse_chunks(raw):
    raw = re.split(r"\n[A-Z/ ]*MANIFEST", raw)[0]  # drop trailing editorial manifest, if any
    chunks, cur = [], {"heading": None, "lines": []}
    for l in raw.splitlines():
        if l.startswith("TITLE:") or l.startswith("SUBTITLE:"):
            continue
        if l.startswith("## "):
            if cur["lines"] or cur["heading"]:
                chunks.append(cur)
            cur = {"heading": l[3:].strip(), "lines": []}
        else:
            cur["lines"].append(l)
    if cur["lines"] or cur["heading"]:
        chunks.append(cur)
    return chunks

def derive(raw, key):
    for l in raw.splitlines():
        if l.startswith(key + ":"):
            return l[len(key) + 1:].strip()
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", default="doc")
    ap.add_argument("--audience", default="external")
    ap.add_argument("--situation", default="standard")
    ap.add_argument("--goal", default="persuade")
    ap.add_argument("--stakes", default="high")
    ap.add_argument("--max-revisions", default="1")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--min-retention", type=float, default=0.70)
    ap.add_argument("--voice-os", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()

    raw = open(a.file).read()
    title = a.title if a.title is not None else derive(raw, "TITLE")
    subtitle = a.subtitle if a.subtitle is not None else derive(raw, "SUBTITLE")
    axes = ["--channel", a.channel, "--audience", a.audience, "--situation", a.situation,
            "--goal", a.goal, "--stakes", a.stakes, "--max-revisions", a.max_revisions]

    chunks = parse_chunks(raw)
    input_prose_words = 0
    results = []
    for i, ch in enumerate(chunks):
        prose, markers = split_markers(ch["lines"])
        voice_input = ((f"## {ch['heading']}\n\n" if ch["heading"] else "") + prose).strip()
        input_prose_words += len(voice_input.split())
        if not voice_input:
            results.append({"heading": ch["heading"], "voiced": "", "markers": markers,
                            "fidelity": None, "mode": "skip"})
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
            tf.write(voice_input); tmp = tf.name
        proc = subprocess.run([sys.executable, "-m", "voice_os", "draft", *axes,
                               "--run-id", f"long_{i}", "--file", tmp],
                              cwd=a.voice_os, capture_output=True, text=True)
        os.unlink(tmp)
        try:
            env = json.loads(proc.stdout)
            voiced = re.sub(r"\n+Best,?\s*$", "", env["output_text"].strip()).strip()
            results.append({"heading": ch["heading"], "voiced": voiced, "markers": markers,
                            "fidelity": round(env["fidelity"]["overall"], 3),
                            "mode": env["mode"], "banned": env.get("banned_hits", [])})
        except Exception as e:
            results.append({"heading": ch["heading"], "voiced": "", "markers": markers,
                            "fidelity": None, "mode": "error",
                            "err": str(e), "stderr": proc.stderr[-300:]})

    out_lines = ([f"# {title}"] if title else []) + ([f"*{subtitle}*", ""] if subtitle else [])
    for r in results:
        if r["voiced"]:
            out_lines.append(r["voiced"])
        for m in r["markers"]:
            out_lines += ["", m]
        out_lines.append("")
    final = "\n".join(out_lines).strip() + "\n"
    open(a.out, "w").write(final)

    out_words = len(final.split())
    retention = round(out_words / input_prose_words, 3) if input_prose_words else 0.0
    fids = [r["fidelity"] for r in results if r["fidelity"] is not None]
    empties = [r["heading"] for r in results if r["mode"] not in ("skip",) and not r["voiced"]]
    not_live = [r["heading"] for r in results if r["mode"] not in ("live", "skip")]
    banned = [r["heading"] for r in results if r.get("banned")]
    em_dash = "—" in final

    guard_failures = []
    if retention < a.min_retention:
        guard_failures.append(f"retention {retention} < {a.min_retention} (possible compression)")
    if empties:
        guard_failures.append(f"empty/failed sections: {empties}")
    if not_live:
        guard_failures.append(f"non-live sections (offline/error): {not_live}")
    if em_dash:
        guard_failures.append("em-dash present (banned)")
    if banned:
        guard_failures.append(f"banned-word hits in sections: {banned}")

    summary = {
        "out": a.out, "sections": len(results),
        "input_prose_words": input_prose_words, "output_words": out_words,
        "retention": retention, "min_retention": a.min_retention,
        "avg_fidelity": round(sum(fids) / len(fids), 3) if fids else None,
        "all_live": not not_live, "em_dash_present": em_dash,
        "guard": "PASS" if not guard_failures else "FAIL",
        "guard_failures": guard_failures,
        "per_section": [{"heading": r["heading"], "mode": r["mode"],
                         "fidelity": r["fidelity"], "words": len(r["voiced"].split())}
                        for r in results],
    }
    json.dump(summary, open(os.path.splitext(a.out)[0] + "-fidelity.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    sys.exit(0 if not guard_failures else 3)

if __name__ == "__main__":
    main()
