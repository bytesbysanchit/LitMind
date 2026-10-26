"""Generate every figure of the project report from the measured result files."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "reports" / "figs"; OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 220})
NAVY, TEAL, ORANGE, GREY, RED, LIGHT = "#1f3b73", "#2a9d8f", "#e07a1f", "#6b7280", "#c0392b", "#eef2fb"

def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=8.5, dashed=False, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.2,
                                ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs * 1.12, color=tc, fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, p, q, color=NAVY, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color, connectionstyle=f"arc3,rad={rad}", ls=ls))

def canvas(w, h, xl=100, yl=60):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off"); return fig, ax

# ---------------------------------------------------------------- Fig 1: architecture
def fig_architecture():
    fig, ax = canvas(7.0, 4.9, 100, 72); ax.set_ylim(-9, 72)
    for x, t in [(8, "Inputs"), (53, "Processing modules (litmind/)"), (92, "Outputs")]:
        ax.text(x, 68.5, t, ha="center", fontweight="bold", color=NAVY)
    # lane 1: characters and relationships
    box(ax, 1, 50, 15, 12, "Novel\n(.txt / .pdf)", fc="#fff7e6", ec=ORANGE)
    box(ax, 21, 50, 18, 12, "text_io.py\nclean and\nnormalise")
    box(ax, 44, 50, 19, 12, "characters.py\nspaCy NER and\nalias clustering", fs=7.8)
    box(ax, 68, 50, 14, 12, "relations.py\nco-occurrence\nstrength", fs=7.4)
    box(ax, 86, 50, 13, 12, "Characters\nand relation\ngraph", fc="#f3f0ff", ec="#6c4ab6")
    for a, b in [((16.6, 56), (20.4, 56)), ((39.6, 56), (43.4, 56)), ((63.6, 56), (67.4, 56)), ((82.6, 56), (85.4, 56))]: arrow(ax, a, b)
    # lane 2: Gemini
    box(ax, 21, 28, 18, 12, "gemini_client.py\nprompts, retries,\nJSON schemas", fc="#e8f6f3", ec=TEAL)
    box(ax, 44, 28, 19, 12, "Gemini API\n(external service)", fc="#e8f6f3", ec=TEAL, dashed=True)
    box(ax, 86, 28, 13, 12, "Summary\nand genre", fc="#f3f0ff", ec="#6c4ab6")
    arrow(ax, (30, 49.6), (30, 40.6)); arrow(ax, (39.6, 34), (43.4, 34), color=TEAL, ls="--"); arrow(ax, (63.6, 34), (85.4, 34), color=TEAL, ls="--")
    arrow(ax, (53.5, 49.6), (53.5, 40.6), color=TEAL, ls="--"); arrow(ax, (75, 49.6), (62, 40.6), color=TEAL, ls="--")
    # lane 3: taste model
    box(ax, 1, 11, 15, 9, "Books.csv\n271,360 books", fc="#fff7e6", ec=ORANGE, fs=7.8)
    box(ax, 1, -1, 15, 9, "Reading\nhistory CSV", fc="#fff7e6", ec=ORANGE, fs=7.8)
    box(ax, 21, 1, 22, 17, "taste_model.py\nTF-IDF and one-hot\nridge regression\npercentile score")
    box(ax, 86, 5, 13, 10, "Taste match\nand picks", fc="#f3f0ff", ec="#6c4ab6")
    arrow(ax, (16.6, 15.5), (20.4, 13)); arrow(ax, (16.6, 3.5), (20.4, 6)); arrow(ax, (43.6, 10), (85.4, 10))
    ax.text(50, -7.5, "Solid arrows: local processing.   Dashed teal arrows: requests to the Gemini API.", ha="center", fontsize=8, color=GREY)
    fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: character pipeline
def fig_char_pipeline():
    fig, ax = canvas(7.0, 5.4, 100, 80)
    steps = [("1  Scan", "spaCy NER over the book;\nrecord label, sentence,\nsyntactic role, pronouns"),
             ("2  Recall", "regex adds 'Mr. Holmes'-\nstyle mentions that\nNER missed"),
             ("3  Normalise", "strip titles, possessives\nand headings; vote over\nall entity labels"),
             ("4  Hard filters", "too short, place word,\ncommon word, 'the X',\nlocation use"),
             ("5  Cluster aliases", "nickname pairs, short\nname to long name,\nspelling variants"),
             ("6  Cluster checks", "mention count, person-\nvs-place vote, evidence\nof acting as a person"),
             ("7  Title mentions", "'the Count' attached to\nthe only holder of\nthat title")]
    X = [2, 38, 74]; Y = [58, 31, 4]; W, H = 24, 19
    pos = [(X[0], Y[0]), (X[1], Y[0]), (X[2], Y[0]), (X[0], Y[1]), (X[1], Y[1]), (X[2], Y[1]), (X[0], Y[2])]
    for (t, d), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.25,rounding_size=1.2", fc=LIGHT, ec=NAVY, lw=1.2))
        ax.text(x + W / 2, y + H - 4, t, ha="center", fontweight="bold", fontsize=9.6, color=NAVY)
        ax.text(x + W / 2, y + 6.8, d, ha="center", va="center", fontsize=8.2, linespacing=1.25)
    mid = lambda y: y + H / 2
    arrow(ax, (X[0] + W + 0.6, mid(Y[0])), (X[1] - 0.6, mid(Y[0]))); arrow(ax, (X[1] + W + 0.6, mid(Y[0])), (X[2] - 0.6, mid(Y[0])))
    arrow(ax, (X[0] + W + 0.6, mid(Y[1])), (X[1] - 0.6, mid(Y[1]))); arrow(ax, (X[1] + W + 0.6, mid(Y[1])), (X[2] - 0.6, mid(Y[1])))
    def elbow(x0, y0, x1, y1, ym):
        ax.plot([x0, x0, x1], [y0, ym, ym], color=NAVY, lw=1.1); arrow(ax, (x1, ym), (x1, y1))
    elbow(X[2] + W / 2, Y[0] - 0.3, X[0] + W / 2, Y[1] + H + 0.5, 53.5)
    elbow(X[2] + W / 2, Y[1] - 0.3, X[0] + W / 2, Y[2] + H + 0.5, 26.5)
    box(ax, X[1] + 0.5, Y[2] + 1.5, W - 1, H - 3, "Output\ncharacters with\naliases, mentions,\ngender, positions", fc="#f3f0ff", ec="#6c4ab6", fs=7.8)
    box(ax, X[2] + 0.5, Y[2] + 1.5, W - 1, H - 3, "Optional Gemini\nreview: merge by\nworld knowledge,\ndrop non-characters", fc="#e8f6f3", ec=TEAL, dashed=True, fs=7.4)
    arrow(ax, (X[0] + W + 0.6, mid(Y[2])), (X[1] - 0.2, mid(Y[2]))); arrow(ax, (X[1] + W, mid(Y[2])), (X[2] + 0.2, mid(Y[2])), color=TEAL, ls="--")
    fig.savefig(OUT / "fig2_char_pipeline.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: alias clustering example
def fig_alias_example():
    E = json.load(open("/tmp/eval_entries.json"))["dracula"]["configs"]["full"]["entries"]
    pick = ["John Seward", "Arthur Holmwood", "Dracula", "Van Helsing"]
    fig, ax = canvas(7.0, 4.4, 100, 64)
    ax.text(21, 61, "Raw NER strings (mentions)", ha="center", fontweight="bold", color=NAVY)
    ax.text(80, 61, "One character", ha="center", fontweight="bold", color=NAVY)
    for k, name in enumerate(pick):
        e = next(x for x in E if x["name"] == name)
        al = [f"{a} ({c})" for a, c in sorted(e["aliases"].items(), key=lambda kv: -kv[1])[:6]]
        lines, cur = [], ""
        for a in al:
            if cur and len(cur) + len(a) + 3 > 34: lines.append(cur); cur = a
            else: cur = (cur + "   " + a) if cur else a
        lines.append(cur)
        top = 56 - k * 14.6; hgt = 12.2
        ax.add_patch(FancyBboxPatch((1, top - hgt + 3), 41, hgt, boxstyle="round,pad=0.2,rounding_size=1", fc="#fff7e6", ec=ORANGE, lw=1))
        ax.text(21.5, top - hgt / 2 + 3, "\n".join(lines), ha="center", va="center", fontsize=8.2, linespacing=1.45)
        box(ax, 62, top - hgt + 3.4, 36, hgt - 0.8, f"{name}\n{e['mentions']} mentions", fc=LIGHT, ec=NAVY, fs=8.2)
        arrow(ax, (43, top - hgt / 2 + 3), (61.4, top - hgt / 2 + 3))
    fig.savefig(OUT / "fig3_alias_example.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: taste training set
def fig_taste_setup():
    fig, ax = canvas(7.0, 3.9, 100, 55)
    box(ax, 1, 36, 25, 14, "My reading history\nN books\ntarget = 1", fc="#e8f6f3", ec=TEAL)
    box(ax, 1, 6, 25, 16, "Catalogue minus history\n(unread books)\nrandom sample, 5 x N\ntarget = 0", fc="#fff7e6", ec=ORANGE, fs=8)
    box(ax, 33, 21, 25, 16, "Features\ntitle TF-IDF (1-2 grams)\nauthor one-hot\npublisher one-hot", fs=7.8)
    box(ax, 64, 21, 15, 16, "Ridge\nregression", fs=8.4)
    box(ax, 85, 38, 14, 12, "Raw score\nof a book", fc="#f3f0ff", ec="#6c4ab6", fs=8)
    box(ax, 85, 6, 14, 16, "Percentile vs\nheld-out\ncatalogue\nbooks", fc="#f3f0ff", ec="#6c4ab6", fs=7.8)
    arrow(ax, (26.6, 43), (32.4, 33)); arrow(ax, (26.6, 14), (32.4, 25)); arrow(ax, (58.6, 29), (63.4, 29)); arrow(ax, (79.6, 33), (84.4, 42))
    arrow(ax, (92, 37.4), (92, 22.8))
    ax.text(50, 1, "Held-out reference: 5,000 unread catalogue books that were not used for training.", ha="center", fontsize=8, color=GREY)
    fig.savefig(OUT / "fig4_taste_setup.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: map-reduce
def fig_mapreduce():
    fig, ax = canvas(7.0, 3.5, 100, 44); ax.set_ylim(-6, 44)
    box(ax, 1, 16, 14, 12, "Book text\n(e.g. 454k\ncharacters)", fc="#fff7e6", ec=ORANGE, fs=8)
    for i in range(4):
        box(ax, 24, 33 - i * 10, 14, 7, f"Chunk {i+1}  (60k)" if i < 3 else "Chunk n", fs=7.6)
        box(ax, 46, 33 - i * 10, 16, 7, "Gemini: notes\n(150-200 words)", fs=7.2, fc="#e8f6f3", ec=TEAL)
        arrow(ax, (38.5, 36.5 - i * 10), (45.5, 36.5 - i * 10), color=TEAL); arrow(ax, (15.5, 22), (23.5, 36.5 - i * 10))
        arrow(ax, (62.5, 36.5 - i * 10), (70.5, 22), color=TEAL)
    box(ax, 71, 14, 14, 16, "Gemini:\nfinal analysis\n(JSON)", fs=8, fc="#e8f6f3", ec=TEAL)
    box(ax, 88, 16, 11, 12, "Summary,\ngenre,\nthemes", fc="#f3f0ff", ec="#6c4ab6", fs=7.8)
    arrow(ax, (85.5, 22), (87.5, 22))
    ax.text(50, -4, "Books under 350,000 characters are sent in a single request instead.", ha="center", fontsize=8, color=GREY)
    fig.savefig(OUT / "fig5_mapreduce.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 6: system comparison
def fig_compare():
    R = json.load(open(ROOT / "reports/character_eval.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    systems = [("raw_ner", "Raw NER", GREY), ("v1_litmind", "LitMind v1", ORANGE), ("full", "LitMind 2", NAVY)]
    metrics = [("unique_precision", "Unique-person\nprecision"), ("core_recall", "Core recall"), ("f1", "F1")]
    for ax, book, title in zip(axes, ("dracula", "sherlock"), ("Dracula (10 core characters)", "Sherlock Holmes Adventures (30 core)")):
        w = 0.26
        for j, (key, lab, col) in enumerate(systems):
            vals = [R[book][key][m] for m, _ in metrics]
            bars = ax.bar(np.arange(3) + (j - 1) * w, vals, w, label=lab, color=col)
            for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.6)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in metrics], fontsize=8); ax.set_ylim(0, 1.12); ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("score"); fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig6_system_compare.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 7: ablation
def fig_ablation():
    R = json.load(open(ROOT / "reports/character_eval.json"))
    cfg = [("full", "Full pipeline"), ("no_lexical", "without lexical filters"), ("no_location", "without location rule"), ("no_weak", "without weak-evidence rule"),
           ("no_clustering", "without alias clustering"), ("no_vote", "without label vote"), ("no_the_x", "without 'the X' test"),
           ("no_the_title", "without title mentions"), ("no_regex_recall", "without regex recall")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, book, title in zip(axes, ("dracula", "sherlock"), ("Dracula", "Sherlock Holmes Adventures")):
        vals = [R[book][k]["f1"] for k, _ in cfg]; cols = [NAVY if k == "full" else (RED if v < R[book]["full"]["f1"] - 1e-9 else GREY) for (k, _), v in zip(cfg, vals)]
        y = np.arange(len(cfg))[::-1]; ax.barh(y, vals, color=cols, height=0.62)
        for yy, v in zip(y, vals): ax.text(v + 0.008, yy, f"{v:.2f}", va="center", fontsize=7)
        ax.set_xlim(0.5, 1.02); ax.set_title(title, fontsize=9); ax.set_xlabel("F1 (unique-person precision and core recall)"); ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(np.arange(len(cfg))[::-1]); axes[0].set_yticklabels([l for _, l in cfg], fontsize=7.8)
    fig.tight_layout(); fig.savefig(OUT / "fig7_ablation.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 8: mention coverage
def fig_mentions():
    E = json.load(open("/tmp/eval_entries.json"))["dracula"]["configs"]["full"]["entries"]; v1 = json.load(open("/tmp/v1_results.json"))["dracula"]["counts"]
    v2 = {e["name"]: e["mentions"] for e in E}
    rows = [("Van Helsing", v1.get("Abraham Van Helsing", 0), v2["Van Helsing"]), ("Lucy Westenra", v1.get("Lucy Westenra", 0), v2["Lucy Westenra"]),
            ("Dracula", v1.get("Count Dracula", 0), v2["Dracula"]), ("Jonathan Harker", v1.get("Jonathan Harker", 0), v2["Jonathan Harker"]),
            ("John Seward", v1.get("John Seward", 0) + v1.get("Jack Seward", 0), v2["John Seward"]), ("Arthur Holmwood", v1.get("Arthur Holmwood", 0), v2["Arthur Holmwood"]),
            ("Quincey Morris", v1.get("Quincey Morris", 0), v2["Quincey Morris"]), ("Mina (Harker + Murray)", 0, v2["Mina Harker"] + v2["Mina Murray"]),
            ("Renfield", 0, v2["Renfield"]), ("Peter Hawkins", v1.get("Peter Hawkins", 0), v2["Peter Hawkins"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.3)); x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, [r[1] for r in rows], w, label="LitMind v1", color=ORANGE); ax.bar(x + w / 2, [r[2] for r in rows], w, label="LitMind 2", color=NAVY)
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r[1] + 5, str(r[1]), ha="center", fontsize=6.6); ax.text(i + w / 2, r[2] + 5, str(r[2]), ha="center", fontsize=6.6)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=7.8); ax.set_ylabel("mentions found"); ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25); fig.tight_layout(); fig.savefig(OUT / "fig8_mentions.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 9: relationship graph
def fig_graph():
    J = json.load(open(ROOT / "reports/v2_relations_runtime.json"))["dracula"]["rels"]
    E = json.load(open("/tmp/eval_entries.json"))["dracula"]["configs"]["full"]["entries"]; cnt = {e["name"]: e["mentions"] for e in E}
    g = nx.Graph()
    for r in J[:12]: g.add_edge(r["a"], r["b"], w=r["strength"])
    pos = nx.spring_layout(g, seed=7, k=1.1, weight="w"); fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for a, b, d in g.edges(data=True): ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=GREY, alpha=0.55, lw=1 + 16 * d["w"], solid_capstyle="round", zorder=1)
    for n in g: ax.scatter(*pos[n], s=60 + cnt.get(n, 20) * 1.6, color=NAVY, zorder=2, edgecolor="white")
    for n in g: ax.text(pos[n][0], pos[n][1] + 0.11, n, ha="center", fontsize=8, zorder=3, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
    ax.axis("off"); ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.3, 1.3)
    ax.text(0, -1.27, "Edge width = relationship strength; node size = mentions. Top 12 pairs of Dracula.", ha="center", fontsize=7.8, color=GREY)
    fig.savefig(OUT / "fig9_graph.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 10: taste ablation + learning curve
def fig_taste():
    R = json.load(open(ROOT / "reports/taste_eval.json")); fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fs = R["feature_sets"]; names = ["title", "author", "publisher", "title+author", "title+author+publisher", "title+author+publisher+year"]
    lab = ["title", "author", "publisher", "title +\nauthor", "title + author\n+ publisher", "+ year"]
    m = [fs[n]["mean"] for n in names]; s = [fs[n]["std"] for n in names]
    cols = [GREY] * 3 + [TEAL] + [NAVY] + [GREY]
    axes[0].bar(range(len(m)), m, yerr=s, color=cols, capsize=2, width=0.65); axes[0].axhline(0.5, color=RED, ls="--", lw=1); axes[0].set_xlim(-0.6, 6.1); axes[0].text(5.65, 0.5, "chance", color=RED, fontsize=7, va="center")
    axes[0].set_xticks(range(len(m))); axes[0].set_xticklabels(lab, fontsize=6.6); axes[0].set_ylim(0.4, 0.8); axes[0].set_ylabel("cross-validated AUC"); axes[0].set_title("Feature sets (36 books)", fontsize=9)
    for i, v in enumerate(m): axes[0].text(i, v + s[i] + 0.008, f"{v:.2f}", ha="center", fontsize=7)
    hs = R["history_size"]; ns = [int(k) for k in hs]; mm = np.array([hs[k]["mean"] for k in hs]); ss = np.array([hs[k]["std"] for k in hs])
    axes[1].plot(ns, mm, "-o", color=NAVY, ms=4); axes[1].fill_between(ns, mm - ss, mm + ss, color=NAVY, alpha=0.18); axes[1].axhline(0.5, color=RED, ls="--", lw=1)
    axes[1].set_xlabel("books in reading history"); axes[1].set_ylabel("cross-validated AUC"); axes[1].set_title("More history, better model", fontsize=9); axes[1].set_ylim(0.4, 0.85); axes[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig10_taste.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 11: hold-out
def fig_holdout():
    R = json.load(open(ROOT / "reports/taste_eval.json")); fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for k, (key, lab, col) in enumerate([("leave_one_book_out", "Hide one book", NAVY), ("leave_one_author_out", "Hide all books by its author", ORANGE)]):
        v = np.array(R[key]["percentiles"]); jitter = (np.random.default_rng(k).random(len(v)) - 0.5) * 0.25
        ax.scatter(v, np.full(len(v), k) + jitter, s=16, color=col, alpha=0.85, zorder=3)
    ax.axvspan(80.5, 85, color=GREY, alpha=0.22, zorder=1); ax.text(82.7, 1.52, "'no information'\nlevel (about 82)", ha="center", fontsize=6.8, color=GREY, va="bottom")
    ax.axvline(90, color=TEAL, ls="--", lw=1); ax.text(90.6, -0.44, "top 10%", color=TEAL, fontsize=7)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Hide one book", "Hide all books\nby its author"], fontsize=8); ax.set_xlim(0, 101); ax.set_ylim(-0.5, 1.95)
    ax.set_xlabel("percentile of the hidden book among catalogue books (higher is better)"); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig11_holdout.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Fig 12/13: DFD level 0 and 1
def _proc(ax, x, y, w, h, text, fs=7.8, fc="#eef2fb", ec=NAVY):
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x, y), w, h, fc=fc, ec=ec, lw=1.2)); ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.2)

def _ent(ax, x, y, w, h, text, fs=8):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, fc="#fff7e6", ec=ORANGE, lw=1.2)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2)

def _store(ax, x, y, w, h, text, fs=7.8):
    ax.plot([x, x + w], [y + h, y + h], color=GREY, lw=1.4); ax.plot([x, x + w], [y, y], color=GREY, lw=1.4)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def fig_dfd0():
    fig, ax = canvas(7.0, 3.4, 100, 50)
    _ent(ax, 1, 20, 18, 12, "User")
    _proc(ax, 50, 26, 30, 22, "0\nLitMind 2\nnovel analysis\nand taste\nprediction", fs=8.6)
    _ent(ax, 81, 33, 18, 11, "Gemini API\n(external)"); _store(ax, 78, 8, 21, 7, "Books.csv catalogue")
    arrow(ax, (19.6, 29), (35, 29)); ax.text(27, 31.2, "novel, history\nCSV, settings", ha="center", fontsize=7.2)
    arrow(ax, (35, 23), (19.6, 23)); ax.text(27, 17.2, "characters, graph,\nsummary, genre,\ntaste score", ha="center", fontsize=7.2)
    arrow(ax, (63, 32.5), (80.4, 36.5)); arrow(ax, (80.4, 40), (63.5, 35)); ax.text(72, 43, "prompts / JSON answers", ha="center", fontsize=7.2)
    arrow(ax, (78, 12), (62, 20)); ax.text(70, 8.2, "catalogue rows", ha="center", fontsize=7.2)
    fig.savefig(OUT / "fig12_dfd0.png", bbox_inches="tight"); plt.close(fig)

def fig_dfd1():
    fig, ax = canvas(7.0, 5.0, 100, 72)
    _ent(ax, 1, 28, 14, 14, "User")
    _proc(ax, 30, 60, 22, 14, "1.0\nLoad and\nclean text"); _proc(ax, 57, 60, 22, 14, "2.0\nExtract characters\nand aliases"); _proc(ax, 84, 60, 22, 14, "3.0\nScore\nrelationships")
    _proc(ax, 30, 36, 22, 14, "4.0\nSummarise\nand find genre"); _proc(ax, 57, 12, 22, 14, "5.0\nTrain and score\ntaste model")
    _ent(ax, 47, 31, 20, 10, "Gemini API\n(external)")
    _store(ax, 76, 32, 23, 6, "D1  session results"); _store(ax, 76, 16, 23, 6, "D2  Books.csv"); _store(ax, 76, 5, 23, 6, "D3  reading history")
    arrow(ax, (10, 42.6), (22, 54)); ax.text(9, 50, "novel", fontsize=7.2, ha="center")
    arrow(ax, (41.4, 60), (45.6, 60)); arrow(ax, (68.4, 60), (72.6, 60))
    arrow(ax, (30, 52.8), (30, 43.2)); ax.text(31.5, 48, "clean text", fontsize=7, ha="left")
    arrow(ax, (57, 52.8), (57, 41.6), color=TEAL, ls="--"); ax.text(58.5, 47, "names + context", fontsize=7, ha="left", color=TEAL)
    arrow(ax, (41.4, 37.5), (46.6, 37.5), color=TEAL, ls="--"); arrow(ax, (46.6, 34.5), (41.4, 34.5), color=TEAL, ls="--")
    ax.text(44, 29.5, "prompts / JSON", fontsize=7, ha="center", color=TEAL)
    arrow(ax, (86, 52.8), (88, 38.6)); ax.text(92.5, 46, "pairs", fontsize=7, ha="center")
    arrow(ax, (75.6, 19), (66.4, 15.5)); arrow(ax, (75.6, 8), (64, 9.5))
    arrow(ax, (8, 27.4), (45, 13)); ax.text(20, 22.5, "history CSV", fontsize=7, ha="center")
    arrow(ax, (47, 9.5), (13, 27.2)); ax.text(32, 11, "taste score", fontsize=7, ha="center")
    arrow(ax, (19, 35), (15.6, 35)); ax.text(17.5, 38, "summary,\ngenre", fontsize=6.8, ha="center")
    fig.savefig(OUT / "fig13_dfd1.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Fig 14: interface layout
def fig_ui():
    from matplotlib.patches import Rectangle
    fig, ax = canvas(7.0, 4.3, 100, 62)
    ax.add_patch(Rectangle((0, 0), 100, 62, fc="white", ec=NAVY, lw=1.4)); ax.add_patch(Rectangle((0, 0), 24, 62, fc="#f1f3f9", ec=NAVY, lw=1))
    ax.text(12, 58, "Settings", ha="center", fontweight="bold", fontsize=9.5, color=NAVY)
    for i, t in enumerate(["Gemini API key", "Gemini model", "spaCy model", "Books.csv path", "Reading history\n(CSV upload)"]):
        y = 49 - i * 9.5; ax.text(2, y + 4.6, t, fontsize=7.6); ax.add_patch(Rectangle((2, y), 20, 3.6, fc="white", ec=GREY, lw=0.8))
    ax.text(62, 58, "LitMind 2", ha="center", fontweight="bold", fontsize=11, color=NAVY)
    ax.add_patch(Rectangle((27, 47), 33, 6, fc="white", ec=GREY, lw=0.8)); ax.text(43.5, 50, "sample book  (drop-down)", ha="center", fontsize=7.6, color=GREY)
    ax.add_patch(Rectangle((63, 47), 34, 6, fc="white", ec=GREY, lw=0.8)); ax.text(80, 50, "upload .txt / .pdf", ha="center", fontsize=7.6, color=GREY)
    for i, (t, on) in enumerate([("Characters and relationships", True), ("Summary and genre", False), ("Taste match", False)]):
        ax.add_patch(Rectangle((27 + i * 24, 39.5), 23.5, 5, fc=LIGHT if on else "white", ec=NAVY, lw=1)); ax.text(38.7 + i * 24, 42, t, ha="center", fontsize=7.2)
    ax.add_patch(Rectangle((27, 4), 70, 34, fc="white", ec=GREY, lw=0.8))
    ax.text(29, 34, "Minimum mentions  [ slider ]      [ Find characters ]      [ Review with Gemini ]", fontsize=7.4)
    ax.add_patch(Rectangle((29, 19), 66, 12, fc="#fafafa", ec=GREY, lw=0.6)); ax.text(62, 25, "table: character, also called, mentions, gender, role, note", ha="center", fontsize=7.4, color=GREY)
    ax.add_patch(Rectangle((29, 6), 31, 11, fc="#fafafa", ec=GREY, lw=0.6)); ax.text(44.5, 11.5, "relationship graph", ha="center", fontsize=7.4, color=GREY)
    ax.add_patch(Rectangle((62, 6), 33, 11, fc="#fafafa", ec=GREY, lw=0.6)); ax.text(78.5, 11.5, "strength, type, reason, evidence", ha="center", fontsize=7.4, color=GREY)
    fig.savefig(OUT / "fig14_ui.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_architecture, fig_char_pipeline, fig_alias_example, fig_taste_setup, fig_mapreduce, fig_compare, fig_ablation, fig_mentions, fig_graph, fig_taste, fig_holdout, fig_dfd0, fig_dfd1, fig_ui):
        f(); print("ok", f.__name__)
