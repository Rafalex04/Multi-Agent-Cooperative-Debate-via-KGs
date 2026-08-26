"""A mechanically different scorer for the same 16 findings.

The phrasing study showed that two wordings of one question, both put to the same
VLM by the same first-token logprob rule, can disagree by 0.44 AUC. Pooling two
views of a shared mechanism cannot fix a failure of that mechanism -- both arms
share the tokeniser, the instruction tuning and the yes/no readout, which is why
the negated arm can invert the content and the pooled score still looks fine.

BiomedCLIP is a different mechanism end to end: a contrastive image-text encoder
trained on 15M biomedical figure-caption pairs, with no autoregressive decoder and
no yes/no head. It cannot exhibit acquiescence bias because it is never asked a
question -- it is asked which of two captions is closer to the image. That is the
screening step in MAGDA and the prompting scheme in Xplainer.

Scoring, per finding, per image:

    s_f = cos(v, t_present_f) - cos(v, t_absent_f)      over a prompt ENSEMBLE

The contrast is against the negated caption of the same finding rather than
against the other findings, so the scale of a finding's caption cannot make it
win everywhere. The ensemble averages several templates before the subtraction,
which is the standard CLIP variance reduction and costs nothing.

WHICH SCORER FOR WHICH FINDING is left to the data. MedCBR reports that
web-scale CLIP declines specifically on echogenicity and posterior features in
ultrasound, hypothesising those depend on imaging physics that natural-image
corpora under-represent, while geometry survives. That is a hypothesis about
which channel to trust per finding, and BrEaST can test it directly: it annotates
the descriptors, so each scorer can be scored on whether it sees the finding, not
on whether it predicts cancer.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "breastMnist"))
sys.path.insert(0, str(_HERE.parents[0] / "10_debate_v5"))

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Ensembled templates. "present" and "absent" are the same sentence apart from
# the finding, so the difference isolates the finding rather than the style.
TEMPLATES = (
    ("breast ultrasound image of a mass with {}",
     "breast ultrasound image of a mass without {}"),
    ("sonogram showing a breast lesion with {}",
     "sonogram showing a breast lesion without {}"),
    ("ultrasound of a breast mass demonstrating {}",
     "ultrasound of a breast mass lacking {}"),
)


def finding_text():
    from run_debate_v5 import all_findings
    R = _HERE.parents[1] / "breastMnist"
    tr = json.loads((R / "data/breast/knowledge_graph.json").read_text())["triples"]
    sc = json.loads((R / "data/breast/schema.json").read_text())
    return [(f, d) for f, d, _ in all_findings(tr, sc)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import torch, open_clip
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
    tok = open_clip.get_tokenizer(MODEL)
    model = model.to(dev).eval()

    ft = finding_text()
    with torch.no_grad():
        T = []
        for _, desc in ft:
            pos = tok([p.format(desc) for p, _ in TEMPLATES]).to(dev)
            neg = tok([n.format(desc) for _, n in TEMPLATES]).to(dev)
            e = model.encode_text(torch.cat([pos, neg]))
            e = e / e.norm(dim=-1, keepdim=True)
            k = len(TEMPLATES)
            T.append(torch.stack([e[:k].mean(0), e[k:].mean(0)]))
        T = torch.stack(T)                                # F x 2 x d
        T = T / T.norm(dim=-1, keepdim=True)

    z = np.load(a.npz, allow_pickle=True)
    imgs = z["imgs"]
    S = np.zeros((len(imgs), len(ft)), dtype=np.float32)
    for i in range(0, len(imgs), a.batch):
        chunk = imgs[i:i + a.batch]
        px = torch.stack([preprocess(Image.fromarray(x).convert("RGB")) for x in chunk]).to(dev)
        with torch.no_grad():
            v = model.encode_image(px)
            v = v / v.norm(dim=-1, keepdim=True)
            sim = torch.einsum("nd,fkd->nfk", v, T)       # n x F x 2
            S[i:i + len(chunk)] = (sim[:, :, 0] - sim[:, :, 1]).cpu().numpy()
        if i % (a.batch * 10) == 0:
            print(f"  {i + len(chunk)}/{len(imgs)}", flush=True)

    np.savez_compressed(a.out, scores=S, findings=np.array([f for f, _ in ft]))
    print(f"wrote {a.out}  shape {S.shape}")


if __name__ == "__main__":
    main()
