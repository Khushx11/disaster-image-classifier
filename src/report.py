"""Build the final report from measured result files and the fixed protocol."""

import json, html
from pathlib import Path
import numpy as np, pandas as pd
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from .data import ROOT
from .plots import NAMES, BASELINES


def run():
    styles = getSampleStyleSheet()
    for name in ["Title", "Heading1", "Heading2", "Heading3"]:
        styles[name].textColor = colors.black
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Title"].fontSize = 23
    styles["Title"].leading = 28
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 10
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 8
    styles.add(
        ParagraphStyle(
            name="SmallText", fontName="Helvetica", fontSize=8, leading=10, spaceAfter=6
        )
    )
    styles.add(
        ParagraphStyle(
            name="EquationText",
            fontName="Courier",
            fontSize=8.3,
            leading=12,
            spaceAfter=9,
            leftIndent=7,
        )
    )
    story = []
    scores = pd.read_csv(ROOT / "results/metrics.csv")
    frame = pd.read_csv(ROOT / "data/manifest.csv")
    audit = json.loads((ROOT / "results/data_audit.json").read_text())
    stats = json.loads((ROOT / "results/statistics.json").read_text())
    features = json.loads((ROOT / "results/feature_metadata.json").read_text())
    clean = scores.query("condition=='clean'")
    ensemble = pd.read_csv(ROOT / "results/ensemble_metrics.csv")
    means = clean.groupby("model").mean(numeric_only=True)
    sd = clean.groupby("model").std(numeric_only=True)
    best = max(BASELINES[:-1], key=lambda k: means.loc[k, "macro_f1"])

    def p(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))

    def h(text):
        p(text, "Heading1")

    def sub(text):
        p(text, "Heading2")

    def page():
        story.append(PageBreak())

    def fig(name, caption, width=17):
        path = ROOT / f"results/figures/{name}.png"
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            ratio = im.height / im.width
        story.append(Image(str(path), width=width * cm, height=width * ratio * cm))
        p(caption, "SmallText")

    def table(rows, widths=None):
        converted = [
            [Paragraph(html.escape(str(x)), styles["SmallText"]) for x in row]
            for row in rows
        ]
        t = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 8))

    def fmt(k, m="macro_f1"):
        return f"{means.loc[k,m]:.4f} ± {sd.loc[k,m]:.4f}"

    p("Image informativeness in disaster posts", "Title")
    p("Availability-aware fusion of image and text", "Heading2")
    p("COMP6001 Computer Vision and Multimodal Machine Learning | Final Capstone Project", "SmallText")
    p("Student: Khush Bhavin Maru | November 2026", "SmallText")
    sub("Abstract")
    p(
        f'This project tests whether text context improves classification of informative disaster images, and how fusion behaves when text or imagery is unavailable. CrisisMMD supplies {len(frame):,} image-text records from seven events. A label-blind duplicate audit creates group-disjoint training, validation and test partitions. Frozen ResNet18 features are combined with training-fitted TF-IDF/SVD text features. The custom PyTorch model uses availability-normalised gates, interaction features, auxiliary unimodal losses and modality dropout. Four benchmark approaches, a dropout-matched concatenation control and three ablations are evaluated with three seeds. Clean-test macro-F1 is {fmt("gated")} for the custom model; the strongest benchmark by observed clean-test mean is {NAMES[best]} at {fmt(best)}. The prespecified comparison against dropout-matched concatenation has a paired group permutation p-value of {stats["clean"]["permutation_p_two_sided"]:.4f}. All numbers are measured outputs of the included code. Missing-modality and shuffled-text tests, rather than clean accuracy alone, establish the practical limits of the approach.'
    )
    sub("Research question")
    p(
        "Can a small, explicitly availability-aware fusion network make useful use of accompanying text without becoming excessively dependent on it? The target is the image informativeness label. This matters because an informative message can accompany an irrelevant photograph, or a useful photograph can appear beside vague text. The study therefore retains disagreeing content instead of assuming that the two modalities always describe the same evidence."
    )
    sub("Project contribution")
    p(
        "The contribution is an implemented and tested fusion variant, together with a reproducible evaluation of its assumptions. It combines established gating and modality-dropout ideas with explicit availability normalisation, masked interaction features and auxiliary heads. This is a course-level design contribution. It is not a claim to have invented multimodal gating, beaten a published benchmark or built a deployable emergency-response service."
    )
    page()
    h("1. Dataset and problem definition")
    p(
        "CrisisMMD was released for research on multimodal disaster posts [1]. This project uses the authors' QCRI Hugging Face mirror at revision 10a5626ba112ab9c50369c473a2062bcf205d708. The annotation files provide text, image paths, event names and informativeness labels. Original per-event TSV tables supply the explicit image_info target. Images are the visual modality and post text is the language modality; neither event IDs nor file names are model inputs. The mirror lists a CC BY-NC-SA 4.0 licence. The data are used here for non-commercial academic analysis."
    )
    counts = frame.label.value_counts()
    splitrows = [["Partition", "Records", "Informative", "Not informative", "Groups"]]
    for split in ["train", "val", "test"]:
        f = frame[frame.split == split]
        splitrows.append(
            [split, len(f), int(f.y.sum()), int((f.y == 0).sum()), f.group.nunique()]
        )
    table(splitrows, [3 * cm, 3 * cm, 3.4 * cm, 4 * cm, 3 * cm])
    p(
        f'There are {audit["tweet_ids"]:,} distinct tweet IDs and {audit["groups"]:,} duplicate groups. The labels comprise {counts.get("informative",0):,} informative and {counts.get("not_informative",0):,} not-informative images. Class weighting is estimated from the training partition only. Macro-F1 gives each class equal weight; accuracy alone can conceal asymmetric errors.'
    )
    fig(
        "dataset_distributions",
        "Figure 1. Measured event and class counts, cleaned text lengths and image aspect ratios. The events and image shapes are unevenly represented; these are properties of this sample, rather than estimates of all disaster posts.",
    )
    p(
        f'The source audit found {audit["mirror_label_disagreements"]:,} JSON labels that differ from the original image_info field. All targets therefore come from the original TSV image annotation; the mirror label is retained only for checking. The target is image informativeness, not a consensus label for a combined post. Records sharing a tweet can have different image labels. The supplied single label is treated as the visual target, and text is contextual evidence. This differs from studies that retain only image-text pairs whose independently annotated labels agree [2]. Published scores under those protocols are not valid direct comparisons with the results below.'
    )
    page()
    h("2. Data integrity and preprocessing")
    p(
        "A union-find audit joins records sharing a tweet ID, a SHA-256 hash of decoded RGB pixels and dimensions, or identical cleaned text of at least 20 characters. Group formation uses content and identifiers without labels. Transitive links remain in one connected group. A shuffled five-fold stratified group splitter assigns fold 0 to test, fold 1 to validation and the remaining folds to training, with random state 42. The resulting sizes are measured rather than forced to exact percentages."
    )
    overlaps = audit["original_split_group_overlap"]
    p(
        f'The original split has {overlaps["train/dev"]:,} shared groups between train and development, {overlaps["train/test"]:,} between train and test, and {overlaps["dev/test"]:,} between development and test under this audit. The rebuilt split has zero shared groups, tweet IDs or exact pixel hashes across its partitions. This does not rule out near-duplicate crops, screenshots or paraphrases. No perceptual near-duplicate claim is made.'
    )
    p(
        "Text is repaired with ftfy, lowercased and stripped of URLs, user handles and the retweet prefix. Hashtag words remain. TF-IDF uses unigrams and bigrams, a minimum document frequency of two and at most 20,000 features. Vocabulary, IDF and the 256-dimensional SVD are fitted exclusively on training text. Both the resulting text vectors and the 512-dimensional visual vectors are L2 normalised. Empty text produces a zero text vector. The clean protocol still treats that zero vector as an observed input; synthetic absence is explicitly flagged."
    )
    fig(
        "paired_examples",
        "Figure 2. Six records selected with a fixed random seed, showing the actual paired inputs and supplied image labels. Images remain in colour to preserve the evidence; plots and diagram styling are monochrome.",
        width=15.7,
    )
    page()
    h("3. Literature and the scope of the design")
    p(
        "Early multimodal crisis analysis had to establish whether paired text and imagery were useful at all. Ofli, Alam and Imran [2] used modality-specific CNN representations and a shared representation, comparing text, image and multimodal models. They also addressed multiple images per tweet and separately annotated modality labels. Their study motivates unimodal and concatenation controls here. This project uses a different target and grouped split, so its measured scores are not presented as a replication of their table."
    )
    p(
        "Arevalo et al. [3] introduced gated multimodal units that learn multiplicative weights for combining modality representations. Their bimodal formulation mixes transformed visual and textual features using a sigmoid gate. The present model retains this basic idea, but renormalises weights when a modality is absent, adds masked difference/product features and trains auxiliary heads. A gate can suppress useful features as well as noise; learning one does not guarantee improvement."
    )
    p(
        "Neverova et al. [4] proposed ModDrop, which randomly removes modalities during training. Their application was gesture recognition, not disaster-image classification. The transferable idea is to expose the classifier to incomplete inputs before evaluation. Here, 80% of training examples retain both modalities, 10% lose text and 10% lose imagery. A concatenation model receives exactly the same distribution, so robustness cannot automatically be credited to the gate."
    )
    p(
        "Aljedani, Alotaibi and Cherif [5] represent a recent, more elaborate fusion approach. CAGMF uses Twitter-RoBERTa and ConvNeXt, bidirectional token/patch attention, scalar gates on class logits and an entropy-conditioned router (accepted manuscript, pp. 7-14). It also uses auxiliary branch losses and modality dropout. Its agreed-label protocol excludes conflicting modality annotations (p. 15), and the authors identify unfiltered and mismatched pairs as future work (pp. 29-30). This project instead uses pooled frozen features, coordinate-wise gates and explicit availability masks. That reduces architectural scope but also removes contextual token/patch alignment and uncertainty routing. Their scores cannot rank this model: targets, filtering, encoders and optimisation differ. The useful comparison is the design trade-off, not an apparent leaderboard gap."
    )
    sub("Gap addressed in this project")
    p(
        "The question tested here is whether a compact feature-level gate remains useful when text can conflict with the image label. All 18,082 records are retained, including 5,339 modality-label disagreements; the target remains the original image annotation. Relative to the implemented benchmarks, the custom design adds availability-normalised feature weights, masked interactions and auxiliary supervision. The matched dropout control and three ablations separate those choices. The contribution is this constrained implementation and evaluation, rather than a claim that gating or missing-input training is new. TF-IDF/SVD keeps language processing inexpensive but discards contextual understanding."
    )
    table(
        [
            [
                "Approach",
                "Reason for inclusion",
                "Difference or trade-off in this project",
            ],
            [
                "Unimodal and shared fusion [2]",
                "Establish whether a second modality adds value",
                "Retains disagreeing labels and explicitly tests missing inputs",
            ],
            [
                "Gated units [3]",
                "Learn feature-wise mixture weights",
                "Adds availability normalisation and masked interactions",
            ],
            [
                "ModDrop [4]",
                "Train with incomplete observations",
                "Uses identical dropout in the matched concatenation control",
            ],
            [
                "Confidence-aware fusion [5]",
                "Recent gating and missing-input research",
                "Different target/filtering; no token attention or entropy router",
            ],
        ],
        [4.1 * cm, 6.1 * cm, 6.2 * cm],
    )
    page()
    h("4. Custom model and mathematical analysis")
    fig(
        "architecture",
        "Schematic of the implemented custom fusion model. The raw visual encoder is frozen; the projections, gate and classifiers are learned.",
    )
    p(
        "Let t and i be the normalised text and image feature vectors. The projections have 128 hidden units, ReLU and dropout 0.2. Binary availability flags mT and mI are applied after projection, so missing inputs cannot reappear through a projection bias. Both flags being zero is outside the supported task. The difference and product terms are present only when both modalities are available."
    )
    for equation in [
        "hT = mT * ReLU(WT t + bT); hI = mI * ReLU(WI i + bI)",
        "d = mT*mI*abs(hT-hI); q = hT*hI",
        "g = sigmoid(Wg [hT,hI,d,q,mT,mI] + bg)",
        "Z = mT*g + mI*(1-g)",
        "wT = mT*g/Z; wI = mI*(1-g)/Z",
        "z = wT*hT + wI*hI",
        "logits = W2 dropout(ReLU(W1 [z,d,q] + b1)) + b2",
    ]:
        p(html.escape(equation), "EquationText")
    p(
        "Multiplication, division and the gate are element-wise. The implementation clamps Z below at 10^-8 for numerical safety. For both inputs present, Z=1, so each coordinate of z is a convex combination of the two projected coordinates. If only imagery is present and Z exceeds the numerical floor, wT=0 and wI=1. The analogous statement holds for text-only input. Regardless of that floor, changing the raw value of a flagged-missing modality cannot affect the output, because its projected features are zeroed before the gate and classifier. This invariance is checked by executable tests."
    )
    sub("Training objective")
    p(
        "The main loss is weighted cross-entropy. For logits l and target y, CE(l,y)=-a[y] log softmax(l)[y], where a[y] is inverse training-class frequency. Separate linear heads predict from hT and hI. Auxiliary losses are averaged over available modalities only. The total batch loss is:"
    )
    p("L = mean(CE(main,y)) + 0.2 * sum(m * CE(aux,y)) / sum(m)", "EquationText")
    p(
        "This gives each branch a direct supervised signal, while allowing the main classifier to learn cross-modal interactions. The no-auxiliary-loss ablation keeps the architecture but removes the second term. Equal-weight fusion removes the learned gate; the no-dropout ablation retains ordinary hidden-unit dropout but removes modality dropout. These changes test different mechanisms and are not assumed to improve performance."
    )
    p(
        "The equations prove a representation property, not an accuracy ordering. Greater capacity can fit spurious correlations, and an image-label task can legitimately favour an image-only model. A result on one held-out split cannot prove that any architecture is universally better."
    )
    page()
    h("5. Benchmarks and experimental protocol")
    table(
        [
            ["Model", "Implementation and rationale"],
            [
                "Text MLP",
                "256-dimensional text vector -> 128 hidden units -> two logits. Tests what text context predicts about an image label.",
            ],
            [
                "Image MLP",
                "512-dimensional frozen visual vector -> 128 hidden units -> two logits. Essential visual-target baseline.",
            ],
            [
                "Concatenation",
                "Separate 128-dimensional projections, concatenation plus availability flags, hidden layer and classifier. Tests ordinary feature-level fusion.",
            ],
            [
                "Late fusion",
                "Mean of independently trained text and image probabilities. When one modality is absent, uses the remaining branch. Tests decision-level fusion without a learned joint representation.",
            ],
            [
                "Concat + dropout",
                "Same concatenation network, with the custom model's modality-dropout distribution. Prespecified statistical comparator.",
            ],
            [
                "Custom gated fusion",
                "Availability-normalised gate, interaction features, auxiliary heads and modality dropout.",
            ],
        ],
        [4.1 * cm, 12.3 * cm],
    )
    p(
        "The benchmark designs test different assumptions. The text MLP is inexpensive but loses sentence-level word order and cannot directly see image evidence. The image MLP uses reusable visual features but lacks post context. Concatenation can learn nonlinear cross-modal interactions, although a missing branch changes its input distribution. Late fusion is modular and easy to fall back from, but cannot learn feature-level interactions and can be affected by unequal probability calibration. The dropout-matched control tests whether robustness comes from missing-input exposure rather than learned gating. The custom model differs from these controls in its normalised gate, masked interactions and auxiliary objective, at the cost of extra parameters and possible negative transfer."
    )
    p(
        "Every learned fusion variant uses seeds 42, 43 and 44, the same split and AdamW with learning rate 0.001, weight decay 0.0001 and batch size 256. Training runs for at most 80 epochs. Checkpoints are selected by validation macro-F1, with earliest tie-breaking, a minimum run of 20 epochs and patience 12. No target-dependent threshold is tuned; prediction is argmax over two classes. All choices are recorded in protocol.json before test evaluation."
    )
    p(
        "The text and image encoders remain fixed during head training. Only the training text is used to fit the sparse representation. No test image labels influence feature extraction, vocabulary fitting, gradients or checkpoint selection. Selected checkpoints are locked before test evaluation. Per-epoch test curves are computed retrospectively from stored epoch states, so they are diagnostic and cannot change the selected epoch."
    )
    sub("Metrics and tests")
    p(
        "Accuracy measures the fraction of correct predictions. Precision and recall are computed for each class and macro-averaged. Macro-F1 is the unweighted mean of the two class F1 scores, with F1=2TP/(2TP+FP+FN). AUROC and average precision use the informative probability. Log loss and Brier score assess probability quality. A fixed 0.5 argmax boundary is retained across all perturbations."
    )
    p(
        "The primary comparison is custom fusion versus concat + dropout on clean-test macro-F1. A paired bootstrap resamples duplicate groups 10,000 times for its difference interval. A two-sided paired randomisation test swaps the two models' predictions jointly within each group, also 10,000 times; p=(1+number of equally or more extreme differences)/(10001). Predictions are averaged across the three seeds before these tests. The interval is conditional on this split and trained models. Seed SD is reported separately. Three secondary robustness p-values receive Holm correction; neither an uncorrected search across all models nor a best-test-model significance claim is made."
    )
    page()
    h("6. Clean-test results and ablations")
    rows = [["Model", "Accuracy", "Macro-F1", "Precision", "Recall", "AUROC"]]
    for k in BASELINES + ["no_gate", "no_aux", "no_drop"]:
        rows.append(
            [
                NAMES[k],
                f'{means.loc[k,"accuracy"]:.4f}',
                fmt(k),
                f'{means.loc[k,"macro_precision"]:.4f}',
                f'{means.loc[k,"macro_recall"]:.4f}',
                f'{means.loc[k,"auroc"]:.4f}',
            ]
        )
    table(rows, [4.3 * cm, 2.1 * cm, 3.5 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm])
    p(
        "Table 1. Three-seed mean metrics. Macro-F1 includes seed standard deviation. Other standard deviations, probability metrics and per-run measurements are supplied in results/metrics.csv. These values are results from the current experiment, not copied published benchmark scores.",
        "SmallText",
    )
    fig(
        "comparison_ablations",
        "Figure 3. Benchmark and ablation macro-F1; error bars show variation across three optimisation seeds, not confidence intervals over the population.",
    )
    delta = means.loc["gated", "macro_f1"] - means.loc["concat_drop", "macro_f1"]
    direction = "higher" if delta > 0 else "lower"
    p(
        f'The custom model\'s clean macro-F1 is {abs(delta):.4f} {direction} than the dropout-matched concatenation mean. The strongest observed benchmark mean is {NAMES[best]} ({means.loc[best,"macro_f1"]:.4f}); this ranking is descriptive. The custom-to-image difference is {means.loc["gated","macro_f1"]-means.loc["image","macro_f1"]:+.4f}. Because the target is visual informativeness, additional text can introduce uncertainty rather than automatically improve the decision.'
    )
    ablations = []
    for k in ["no_gate", "no_aux", "no_drop"]:
        ablations.append(
            f'{NAMES[k].lower()}: {means.loc[k,"macro_f1"]-means.loc["gated","macro_f1"]:+.4f}'
        )
    p(
        "Relative to the full custom model, clean macro-F1 changes as follows: "
        + "; ".join(ablations)
        + ". These are component-removal comparisons under the fixed settings. They do not isolate parameter count, nor do they justify retuning the model after observing the test set."
    )
    page()
    h("7. Training behaviour and classification errors")
    fig(
        "learning_curves",
        "Figure 4. Seed 42 learning curves for representative models. The vertical line marks the checkpoint chosen using validation only. Dotted test curves were evaluated after selection was locked; they were not available to the optimiser.",
        width=16.7,
    )
    p(
        "The training/validation gap shows how well fitting the available training labels translates to held-out groups. A rising training curve with flat or falling validation performance is evidence of overfitting under this protocol. The selected epoch can precede the final recorded epoch because patience allows continued training after the last validation improvement. Different stopping times are a consequence of the same rule, rather than hand-selected budgets."
    )
    fig(
        "classification_diagnostics",
        "Figure 5. Confusion matrix and probability-ranking curves using three-seed ensembles. The confusion matrix contains counts, not percentages. ROC and precision-recall curves measure different operating trade-offs.",
        width=16.7,
    )
    p(
        "A false positive admits an uninformative image to the filtered stream; a false negative discards potentially useful visual evidence. The two errors have different costs in a real response workflow, but this dataset does not supply an operational cost model. Therefore no claim of deployment readiness or an optimal emergency-response threshold follows from these plots."
    )
    page()
    h("8. Missing inputs, mismatched text and uncertainty")
    fig(
        "robustness",
        "Figure 6. Identical held-out records under four conditions. Text shuffling keeps each image and its visual label fixed. Missing inputs are represented by a zeroed branch and availability flag. Error bars are seed SD.",
    )
    rows = [
        [
            "Condition",
            "Custom ensemble F1",
            "Concat + dropout ensemble F1",
            "Difference CI",
            "Paired p",
        ]
    ]
    for condition in ["clean", "no_text", "no_image", "shuffled_text"]:
        a = ensemble[
            (ensemble.model == "gated") & (ensemble.condition == condition)
        ].macro_f1.iloc[0]
        b = ensemble[
            (ensemble.model == "concat_drop") & (ensemble.condition == condition)
        ].macro_f1.iloc[0]
        st = stats[condition]
        ci = st["bootstrap_ci95"]
        pv = st.get("secondary_holm_p", st["permutation_p_two_sided"])
        rows.append(
            [
                condition.replace("_", " "),
                f"{a:.4f}",
                f"{b:.4f}",
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                f"{pv:.4f}",
            ]
        )
    table(rows, [3.5 * cm, 3.2 * cm, 3.2 * cm, 4.3 * cm, 2.0 * cm])
    st = stats["clean"]
    lo, hi = st["bootstrap_ci95"]
    p(
        f'Table 2. The primary clean ensemble difference is {st["difference_macro_f1"]:+.4f}, with a paired group-bootstrap 95% interval [{lo:+.4f}, {hi:+.4f}] and unadjusted primary p={st["permutation_p_two_sided"]:.4f}. Secondary rows show Holm-adjusted p-values; their intervals are unadjusted descriptive intervals. Ensemble differences need not equal differences between seed means.',
        "SmallText",
    )
    p(
        "Text absence tests dependence on context. Image absence is more severe because the target describes the image; a text-only fallback can be informed by correlations without directly observing the labelled evidence. Text shuffling tests a deliberately broken pairing. These perturbations are controlled stress tests, not estimates of how often real systems lose a modality. The image-only baseline should remain unchanged when text is removed or shuffled, which also serves as an evaluation check."
    )
    p(
        "The primary clean interval includes zero (p=0.1148), and none of the three secondary comparisons remains significant at 0.05 after Holm correction. This experiment therefore does not establish a statistical advantage over the matched control. The custom model also loses to simple late fusion on mean clean-test macro-F1. With shuffled text it falls to 0.7295, while the image-only model remains at 0.8059. Modality dropout partly protects against absence, but the gate does not reliably reject misleading text. The no-gate and no-modality-dropout ablations both have higher clean means than the full custom model. These negative findings argue against adding this gate solely to improve clean accuracy."
    )
    p(
        "An interval crossing zero is compatible with either small positive or negative effects under this bootstrap. A small p-value describes this prespecified comparison and null randomisation, not the practical importance of the difference. The same events appear across partitions, so even a clear within-dataset effect does not establish generalisation to an unseen disaster."
    )
    page()
    h("9. Feature structure and event-level behaviour")
    fig(
        "feature_distributions",
        "Figure 7. Two-dimensional PCA of held-out feature vectors, with PCA fitted on training data only. Overlap is expected: a two-dimensional projection discards most information and is not a measure of classifier quality.",
    )
    fig(
        "events_gates",
        "Figure 8. Event-specific ensemble macro-F1 and distributions of the custom model's mean text gate weight. Event sample sizes are in results/event_metrics.csv. Gate weights are descriptive network quantities, not validated explanations of a decision.",
    )
    p(
        "Event-level analysis checks whether aggregate performance hides uneven behaviour. Small event subsets have greater uncertainty and can yield unstable macro-F1. The project does not tune an event-specific model or use the event identifier as an input. Text can nevertheless encode event names, so correlations from the same seven disasters can still assist prediction. A leave-one-event-out study would address a different and stronger generalisation question."
    )
    p(
        "The gate histogram shows how this fitted network weights its projected modalities. Its average need not track causal importance: the classifier also receives interaction features, and projected feature scales influence the effective contribution. Shuffled text can alter gate values without the gate reliably recognising every mismatch. The measured robustness scores provide the relevant evidence; a visually appealing histogram does not prove reliability."
    )
    page()
    h("10. Computation, implementation and limitations")
    metadata = [
        json.loads(p.read_text())
        for p in (ROOT / "results/models").glob("*/metadata.json")
    ]
    cost = pd.DataFrame(metadata)
    rows = [
        ["Model", "Trainable parameters", "Mean fit time (s)", "Mean selected epoch"]
    ]
    for k in BASELINES:
        if k == "late_fusion":
            rows.append(
                [
                    NAMES[k],
                    "Sum of two unimodal heads",
                    "No additional training",
                    "Uses unimodal selections",
                ]
            )
            continue
        v = cost[cost.kind == k]
        rows.append(
            [
                NAMES[k],
                int(v.parameters.iloc[0]),
                f"{v.seconds.mean():.1f}",
                f"{v.best_epoch.mean():.1f}",
            ]
        )
    table(rows, [4.3 * cm, 4.5 * cm, 4.0 * cm, 3.4 * cm])
    p(
        f'Visual extraction used {features["image_extraction"]["device"]}; cached fusion heads used CPU. The local environment is recorded in each run\'s metadata and requirements.txt. The selected sparse text vocabulary has {features["text_vocab"]:,} terms; the SVD retains {features["svd_explained_variance"]:.3f} of training TF-IDF variance. Fit times exclude data download, frozen encoder inference and retrospective test evaluation. The streaming extraction timer includes download waiting, so it is not reported as isolated encoder throughput.'
    )
    sub("Time and memory analysis")
    p(
        "Let N be the number of records, B batch size, dT=256, dI=512 and h=128. Cached float32 text/image features require 4N(dT+dI) bytes, about "
        + f"{len(frame)*4*768/2**20:.1f} MiB"
        + ". This excludes temporary arrays and the encoder. Projection work is O(Bh(dT+dI)); the custom gate and hidden classifier add O(Bh^2). Hidden activation memory is O(Bh), while AdamW maintains parameter, gradient and first/second-moment tensors, roughly four parameter-sized float32 buffers before temporary workspace. Epoch snapshots scale with the number of epochs and parameters and are kept outside the submission archive. Sparse TF-IDF storage scales with nonzero term counts; fitting SVD adds work depending on vocabulary, rank and solver iterations. The frozen ResNet inference cost remains necessary for each new image even though its training gradients are absent."
    )
    sub("Implementation checks and remaining risks")
    p(
        "Checks cover missing-input invariance, gradient flow, split separation, cache invalidation, configuration changes and agreement between predictions and metrics. A separate seed-42 repeat reproduced the selected epoch and every model parameter exactly from the same cached features; it did not repeat image downloads or encoder inference. Fresh reproduction writes to an isolated folder; evaluation also works without temporary epoch archives. The downloader raises errors rather than substituting unavailable images. Remaining risks include corrupt files, dependency changes, insufficient memory, device-specific numerical variation and edited duplicates escaping exact hashing. The saved text pipeline is a trusted local serialisation; do not load an unknown replacement."
    )
    p(
        "The evaluation covers one historical dataset, two classes and one grouped split. Same-event correlations, annotation noise and edited duplicates limit generalisation. TF-IDF lacks contextual reasoning, and frozen ImageNet features may miss specialised visual evidence. Capacity and optimisation budgets differ across models; this is not an equal-compute architecture search. Predictions would need human verification in an operational setting."
    )
    sub("Conclusion")
    p(
        f'Late fusion was strongest on clean inputs. The custom model scored {means.loc["gated","macro_f1"]:.4f} mean macro-F1; its ensemble difference against the dropout-matched control was {st["difference_macro_f1"]:+.4f} (p={st["permutation_p_two_sided"]:.4f}). Its weaker shuffled-text performance shows that learned gates alone did not reliably protect against misleading context.'
    )
    page()
    h("References and reproduction")
    refs = [
        (
            "[1]",
            "Alam, F., Ofli, F. and Imran, M. (2018). CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. Proceedings of ICWSM, 12(1).",
            "https://arxiv.org/abs/1805.00713",
        ),
        (
            "[2]",
            "Ofli, F., Alam, F. and Imran, M. (2020). Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response. Proceedings of ISCRAM.",
            "https://arxiv.org/abs/2004.11838",
        ),
        (
            "[3]",
            "Arevalo, J., Solorio, T., Montes-y-Gomez, M. and Gonzalez, F. A. (2017). Gated Multimodal Units for Information Fusion. ICLR Workshop track.",
            "https://arxiv.org/abs/1702.01992",
        ),
        (
            "[4]",
            "Neverova, N., Wolf, C., Taylor, G. W. and Nebout, F. (2016). ModDrop: Adaptive Multi-Modal Gesture Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(8), 1692-1706.",
            "https://arxiv.org/abs/1501.00102",
        ),
        (
            "[5]",
            "Aljedani, N., Alotaibi, R. and Cherif, A. (2026). Confidence-aware adaptive multimodal fusion for crisis event analysis from social media. Scientific Reports. Early publication, 11 September 2026.",
            "https://doi.org/10.1038/s41598-026-68362-4",
        ),
    ]
    for key, text, url in refs:
        p(f'{key} {text}<br/><link href="{url}" color="black">{url}</link>')
    sub("Reproduction map")
    table(
        [
            ["Stage", "Code and evidence"],
            [
                "Data and grouping",
                "src/data.py; data/manifest.csv; results/data_audit.json",
            ],
            [
                "Visual/text features",
                "src/features.py; results/feature_metadata.json; results/text_pipeline.joblib",
            ],
            ["Custom method and baselines", "src/models.py; protocol.json"],
            [
                "Optimisation and selection",
                "src/train.py; results/models/*/history.json and metadata.json",
            ],
            [
                "Test and statistics",
                "src/evaluate.py; results/predictions/; metrics.csv; statistics.json",
            ],
            ["Figures and checks", "src/plots.py; src/verify.py; results/figures/"],
            ["Walkthrough and demo", "Capstone.ipynb; README.md; src/predict.py"],
        ],
        [4.1 * cm, 12.3 * cm],
    )
    p(
        "Run python -m src reproduce to rebuild the experiment in a new folder without overwriting submitted results. Individual stage commands can resume an interrupted run from that folder. The notebook separates viewing the supplied measurements from expensive full reproduction. Raw images, generated runs and temporary epoch caches are excluded from the archive; the pinned reproduction command rebuilds them. Model weights, predictions and the text transform required for the demonstration are included. The presentation video is created separately."
    )

    p(
        "End-to-end local validation rebuilt the same labels and split, extracted fresh GPU features and trained all 24 models. The maximum change in clean three-seed mean macro-F1 was 0.0041 across models; late fusion remained strongest. Fresh image features differed by at most 2.98e-8, and some early-stopping choices changed. The detailed comparison is included with the results. The tables in this report retain the original experiment; exact parameter reproduction was verified separately using its original cached features."
    )

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(0.3)
        canvas.line(1.8 * cm, 1.4 * cm, 19.2 * cm, 1.4 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            1.8 * cm, 1.0 * cm, "COMP6001 | Khush Bhavin Maru | Final Capstone Project"
        )
        canvas.drawRightString(19.2 * cm, 1.0 * cm, str(doc.page))

    doc = SimpleDocTemplate(
        str(ROOT / "Report.pdf"),
        pagesize=(21 * cm, 29.7 * cm),
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.9 * cm,
        title="Image informativeness in disaster posts",
        author="Khush Bhavin Maru",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Created", ROOT / "Report.pdf")
