#!/usr/bin/env python3
"""
v15.8 patch: add M6, the fixed classical cos(pi x) front-end (PREREGISTRATION_v15_8.md).

    python3 patch_v158_m6.py /g/data/wa66/yh2481/qcq_cnn_v158      (setup_v158.sh does this)

Two edits to the v15.5 train_gadi.py, nothing else:
  1. One block inserted immediately ABOVE the V155_INPUT_NOISE block. It wraps build_model and
     expected_params by name: "M6" builds QCQCNN(CosPixelFrontEnd(), ClassicalHead()); every
     other name goes to the previous function untouched. Because the v15.5 block comes after
     it, v15.5's noise wrapper and cross-evaluation see M6 like any other model.
  2. The --model argparse choices gain "M6".
No model, data, split, noise or training line is edited. Backup: train_gadi.py.v155bak.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "V158_M6_COS"
V155_HDR = "# V155_INPUT_NOISE  (inserted by patch_v155_input_noise.py"
CHOICE_OLD = 'choices=list(["M1", "M2", "M3", "M4"]) + ["M5"])'
CHOICE_NEW = 'choices=list(["M1", "M2", "M3", "M4"]) + ["M5"] + ["M6"])  # V158_M6_COS'

BLOCK = r'''# ═══════════════════════════════════════════════════════════════════════════════
# V158_M6_COS  (inserted by patch_v158_m6.py -- v15.8 fixed classical cos(pi x) front-end)
# M6 = M3's architecture with Circuit 4 replaced by the fixed map p -> cos(pi p) on each of the
# four pixels of every 2x2 patch (= Circuit 4 with every trainable gate at the identity).
# 0 front-end parameters; same backbone and classical head; draws no RNG.
# Every other model name goes to the previous build_model / expected_params untouched.
# ═══════════════════════════════════════════════════════════════════════════════
import math as _v158_math
import sys as _v158_sys


class CosPixelFrontEnd(nn.Module):  # noqa: F821
    """(B,1,H,W) -> (B,4,H/2,W/2): channel c = cos(pi * pixel c of the 2x2 patch),
    pixel order (r0c0, r0c1, r1c0, r1c1) = the circuit's (block A: q0,q1; block B: q2,q3)."""

    def forward(self, x):
        B, _, H, W = x.shape
        oh, ow = H // 2, W // 2
        patches = (x.unfold(2, 2, 2).unfold(3, 2, 2)
                   .permute(0, 2, 3, 1, 4, 5)
                   .reshape(B * oh * ow, 4))
        out = torch.cos(patches * _v158_math.pi)  # noqa: F821
        return out.reshape(B, oh, ow, 4).permute(0, 3, 1, 2).contiguous()

    def n_trainable_params(self):
        return 0


_v158_build_model_prev = build_model  # noqa: F821
_v158_expected_params_prev = expected_params  # noqa: F821


def build_model(model_name, *args, **kwargs):  # noqa: F811
    if model_name != "M6":
        return _v158_build_model_prev(model_name, *args, **kwargs)
    m = QCQCNN(CosPixelFrontEnd(), ClassicalHead())  # noqa: F821
    total, (fe, bb, be) = m.total_params()
    expected = expected_params("M6", N_CLASSES, N_OBS)  # noqa: F821
    n_req = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if total != expected or fe != 0 or n_req != total or len(list(m.front_end.parameters())) != 0:
        _v158_sys.exit("[V158] FATAL: M6 params total=%d front=%d requires_grad=%d expected=%d"
                       % (total, fe, n_req, expected))
    print(f"[VERIFIED] M6 params: front={fe} backbone={bb} back_end={be} total={total} "
          f"(expected {expected}) OK", flush=True)
    # self-check: the fixed map equals Circuit 4 with all weights 0, on a fixed probe (no global RNG)
    try:
        g = torch.Generator().manual_seed(158)  # noqa: F821
        probe = torch.rand(3, 1, 28, 28, generator=g)  # noqa: F821
        probe[:, :, :6, :] = 0.0
        probe[:, :, -4:, :] = 1.0
        c4 = TrainableQuanvFrontEndTorch(seed=0)  # noqa: F821  (private generator)
        with torch.no_grad():  # noqa: F821
            c4.weights.zero_()
            ref = c4(probe)
            got = m.front_end(probe)
        err = (ref - got).abs().max().item()
        ok = got.shape == (3, 4, 14, 14) and err < 1e-5
        print("[V158] M6 front-end = Circuit 4 at w=0 (cos(pi x) per pixel): max|diff|=%.2e %s"
              % (err, "PASS" if ok else "FAIL"), flush=True)
        if not ok:
            _v158_sys.exit("[V158] FATAL: M6 front-end does not equal Circuit 4 at w=0")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        _v158_sys.exit("[V158] FATAL: M6 self-check crashed: %s: %s" % (type(e).__name__, e))
    print("[V158] M6 = fixed cos(pi x) front-end, 0 params, classical head, warmup as M4", flush=True)
    return m


def expected_params(model_name, *args, **kwargs):  # noqa: F811
    # M6 has M4's parameter count: same backbone and head, 0 front-end params.
    return _v158_expected_params_prev("M4" if model_name == "M6" else model_name, *args, **kwargs)
# ═══ end V158_M6_COS ═════════════════════════════════════════════════════════════

'''


def die(msg):
    sys.exit("PATCH REFUSED: %s" % msg)


def main():
    if len(sys.argv) != 2:
        die("usage: patch_v158_m6.py <repo root>")
    root = Path(sys.argv[1]).resolve()
    tp = root / "train_gadi.py"
    if not tp.exists():
        die("%s not found" % tp)
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already v15.8-patched. Start from a clean v15.5 copy.")
    for m, what in (("V155_INPUT_NOISE", "v15.5 noise patch"), ("V14_SPLIT_AUDIT", "v14 split audit"),
                    ("V135_M5_PATCH", "v13.5 M5 patch")):
        if m not in t:
            die("train_gadi.py lacks the %s (%s). Build from qcq_cnn_v155." % (what, m))
    if "V15_FE_NOISE" in t or "_v16_strip_argv" in t:
        die("train_gadi.py carries a v15/v16 patch. Build from qcq_cnn_v155.")
    if t.count(V155_HDR) != 1:
        die("expected exactly one v15.5 block header, found %d" % t.count(V155_HDR))
    if t.count(CHOICE_OLD) != 1:
        die("the --model choices line was not found exactly once: %r" % CHOICE_OLD)
    for name in ("class QCQCNN(", "class ClassicalHead(", "TrainableQuanvFrontEndTorch", "def expected_params(",
                 "needs_warmup = model_name in (\"M2\", \"M3\", \"M5\") or isinstance(model.back_end, QNNBackend)"):
        if name not in t:
            die("train_gadi.py: %r not found" % name)
    # insert above the v15.5 block's opening rule line
    i = t.index(V155_HDR)
    j = t.rfind("\n# ═", 0, i)
    if j < 0:
        die("could not find the rule line above the v15.5 header")
    j += 1
    t2 = t[:j] + BLOCK + t[j:]
    t2 = t2.replace(CHOICE_OLD, CHOICE_NEW)
    bak = tp.with_name(tp.name + ".v155bak")
    if not bak.exists():
        shutil.copy2(str(tp), str(bak))
    tp.write_text(t2)
    print("  inserted the V158 block above the v15.5 block (line %d); --model choices += M6" % (t[:j].count("\n") + 1))
    print("Patched %s in %s  (backup %s)" % (tp.name, root, bak.name))


if __name__ == "__main__":
    main()
