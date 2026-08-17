# EmotiSense — Future Improvements, Scoped

This scopes out the five items listed under "Future Improvements" in the
main README, now that Phases 3–6 (multimodal fusion, smarter
recommendations, the ML dashboard, and robustness testing) plus Spotify
login are done. None of these are started yet - this is the planning pass
before picking one to actually build.

Two real numbers ground a few of the estimates below: the text dataset is
20,000 labelled sentences (16k/2k/2k train/val/test split, 6 emotions -
this is the well-known "Emotions dataset for NLP", originally scraped
from Twitter), and the audio dataset is RAVDESS, sorted by
`src/sort_ravdess.py` into 5 folders (happy/sad/angry/fear/neutral - a
reduced set of RAVDESS's full 8 emotions). RAVDESS is 24 professional
actors (12 male/12 female) reading two fixed, scripted sentences in a
studio - it's a well-controlled dataset, but that also means no accent
diversity, no spontaneous speech, and a small speaker pool.

---

## 1. Deep learning models (CNN/LSTM/BERT)

**STATUS: text half attempted, done - real, mixed results below.**

**What it means:** Replacing TF-IDF + Linear SVM (text) and hand-engineered
librosa features + Gradient Boosting (audio) with learned representations.

**Why it's actually motivated here, not just "newer = better":** Phase 6's
adversarial testing already found the concrete failure this would fix -
TF-IDF is bag-of-words, so it has no way to represent word order or
negation. That's exactly why "I am not happy at all" scored 72.6% Joy and
sarcasm scored 73.9% Joy. A transformer-based text model (e.g.
DistilBERT) understands negation and context because it encodes sequences,
not just word counts - this is the one item on this list with a
documented before/after test already sitting in `adversarial_probe.py`.

**Text (do this first):**
- Fine-tune a small pretrained model (DistilBERT is the practical choice -
  ~66M params, fine-tunes in a reasonable time on CPU, unlike full BERT)
  on the existing 20k-sentence dataset using `transformers` + `torch`.
- Inference path changes from `vectorizer.transform() -> model.predict_proba()`
  to `tokenizer() -> model.forward() -> softmax` - a real but contained
  change, isolated to `emotion_logic.py`'s prediction functions.
- Trade-off to flag: model size goes from a ~1MB `.pkl` to a ~250MB model
  folder, and single-prediction latency goes from ~instant to maybe
  50-200ms on CPU - still fine for a Streamlit app, just worth knowing.

**Audio (do this after, and treat as a separate decision):** RAVDESS's
~1440 clips (of which this project uses 5 of the 8 emotion classes) is too
small to train a CNN/LSTM from scratch without overfitting. The realistic
path is fine-tuning a pretrained audio embedding model (wav2vec2 or a
pretrained speech-emotion-recognition checkpoint from Hugging Face) rather
than training a network from zero - meaningfully bigger download (~1GB+)
and heavier dependencies (`torch`, `torchaudio`, `transformers`) than the
text half.

**Effort:** Medium (text) / High (audio). **Blockers:** needs `torch` +
`transformers` installed on your machine (`pip install -r
requirements-distilbert.txt` - kept out of the main requirements.txt
since app/app.py never imports them; not installable in this sandbox -
no pypi access here - but that's a non-issue on your own machine).
GPU isn't required for the text half but would make iteration much faster;
CPU-only fine-tuning of DistilBERT on 16k sentences is feasible but slow
(expect it to take a while per epoch).

**Suggested first step:** just the text half, since it directly answers a
failure you already found and measured.

**What actually happened (2026-08-15):** Fine-tuned `distilbert-base-uncased`
via `src/train_text_distilbert.py` - CPU-only on a 3.4GB-RAM machine, so a
smoke test first confirmed feasibility before committing to a full run (see
[[project-hardware-constraints]] in memory). Full fine-tuning over all
16,000 rows was ruled out on time grounds (~3.5 hours/epoch measured from
the smoke test); trained instead on a stratified 3,000-row subset
(class-weighted loss) for 2 epochs, ~39 minutes total.

Accuracy: **89.0%**, F1 (macro): **85.1%** - essentially matching the
production Linear SVM (89.2% / 83.9%) using only 19% of the training data,
and meaningfully better recall on the rare classes SVM struggled with
(surprise: 94% recall, love: 93% recall). Full report in
`results/distilbert_classification_report.txt`.

**The negation claim did not hold up.** This item's whole premise was that
a transformer would fix "I am not happy at all" scoring as Joy - re-running
that exact probe against the fine-tuned model: still 96.1% Joy, no
different from the TF-IDF model's failure. Worse, "I am not sad" scored
87.7% Sadness - backwards. Checked whether this was a training-data gap
before concluding anything: 12.1% of training rows contain a negation word
("not"/"never"/"n't"), so it's not a signal-scarcity problem - 2 epochs on
a subset apparently wasn't enough to teach the compositional "not + X
flips X" pattern this project needed, even though DistilBERT's
pretrained embeddings are contextual. **Architecture choice alone didn't
fix the thing it was chosen to fix** - worth remembering before assuming
any future model swap solves a specific failure mode without re-testing it
directly.

Not yet promoted to production (`results/emotion_model.pkl` is still the
calibrated Linear SVM) given the negation result and the accuracy gain
being marginal at best. Candidate model saved at
`results/distilbert_emotion_model/` if someone wants to pick this up -
more epochs, the full 16k rows, or deliberately oversampling
negation-containing examples would be the natural next things to try
before concluding transformers can't help here at all.

---

## 2. Larger, more real-world datasets

**What it means:** Supplementing or replacing the current datasets with
bigger/more diverse ones.

**Text:** The current 20k-sentence set is already a reasonable size, but
it's Twitter-only (2016-era, short and informal) and unevenly balanced -
`surprise` has 572 training examples against `joy`'s 5,362, a ~9x gap.
[GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
(58k Reddit comments, 27 fine-grained emotions - would need mapping down
to this project's 6) is the natural next dataset: bigger, a different
register of informal text, and Google-curated with documented
inter-annotator agreement.

**Audio:** RAVDESS alone means every training example is the same two
scripted sentences read by 24 actors - a model trained only on this can
latch onto phrasing/actor quirks rather than general emotional-speech
cues. Adding [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
(7,442 clips, 91 actors, more diverse) or
[TESS](https://tspace.library.utoronto.ca/handle/1807/24487) would add
speaker diversity without a licensing hurdle (both are freely available
for research use, unlike IEMOCAP - see item 4).

**Effort:** Medium - mostly data engineering, not modelling. The project
already has `compare_text_models.py`/`compare_audio_models.py` to
benchmark before/after, so "does more data actually help" is directly
measurable rather than a guess. Main work: download, relabel to this
project's emotion set, and (for audio) match the feature-extraction
pipeline's expectations.

**Note:** this pairs naturally with item 1 - a bigger, more diverse
dataset is what a deep learning model actually needs to earn its keep;
feeding DistilBERT the same 20k tweets a linear SVM already handles
reasonably well would show a smaller gain than feeding it a bigger, more
varied set.

---

## 3. Improved fairness and generalisation

**STATUS: leakage audit done (2026-08-15) - real, serious finding, fixed.
Demographic slicing and broader adversarial probing not yet started.**

**What it means:** Less a single feature, more an audit - checking *how*
well the models generalise, not just their headline accuracy.

**Three concrete, boundable pieces:**
- **Train/test leakage audit.** ~~With RAVDESS's small actor pool, it's
  worth explicitly confirming~~ **Done via `tests/actor_leakage_audit.py`:
  the audio train/test split leaked every single actor into both sides -
  100% of test-set actors (24/24 RAVDESS, 91/91 CREMA-D) also appeared in
  training, not a partial leak. Fixed by switching `train_audio_model.py`/
  `compare_audio_models.py` to `StratifiedGroupKFold` grouped by actor.
  The corrected numbers are meaningfully different, not just lower - the
  winning model changed from Gradient Boosting to Random Forest once the
  leak was removed (52.8% vs the old leaky 51.8%), and Random Forest turned
  out far better calibrated on non-speech input as a side effect (all 5
  synthetic adversarial audio probes now correctly flagged low-confidence,
  vs none before). Full detail in the README's "Fairness & Generalisation
  Audit" section.**
- **Demographic slicing.** RAVDESS actors are labelled by gender in the
  filename convention - cheap to compute per-gender accuracy and see if
  the model performs unevenly, similar to how the Dashboard tab already
  breaks down performance, just sliced a different way.
- **Broader adversarial probing.** `adversarial_probe.py` already tests
  sarcasm/negation/gibberish - extending it with dialectal variation,
  differently-gendered names in otherwise-identical sentences, and
  non-Western-name inputs would surface bias the current probe set
  doesn't target.

**Effort:** Low-Medium. This is the cheapest item on the list to scope
into something concrete, and it's a direct continuation of Phase 6's
robustness work rather than a new subsystem - could realistically be
"Phase 6.5" before touching any models.

**Word-level blind-spot audit (2026-08-17):** Following the same method
that found the "frustrating"-family anger words being misread as joy
(comparing word frequency in misclassified vs. correctly-classified
examples of the same true class, on val + test), two more candidates
turned up, both consistent across val and test:

- `strange`/`weird` driving bidirectional fear↔surprise confusion -
  `surprise -> fear` was the model's second-highest per-class error rate
  (16.7% of test), right behind `love -> joy`
- `stressed`/`hated` driving bidirectional anger↔sadness confusion, same
  shape as the original fix

**Attempted fix did not hold up.** Added ~5-6 new training examples per
word/class gap (22 total) using the exact same recipe as the
`frustrating` fix, retrained, re-measured the same four confusion pairs
on held-out test data: `surprise -> fear` moved slightly (16.7% -> 15.2%),
but `fear -> surprise` and `anger -> sadness` got marginally *worse*
(4.5% -> 4.9%, 4.7% -> 5.1%), and `sadness -> anger` didn't move at all
(2.9% -> 2.9%). Reverted rather than promoting a change that doesn't
demonstrably help - `results/emotion_model.pkl` and `data/raw/train.txt`
are both back to their pre-attempt state.

**Why this one likely didn't work when `frustrating` did:** 22 examples
against a ~16k-row dataset (~0.14%) may simply be too small a signal for
a linear model's decision boundary to pick up in a way that generalises
to held-out data. It's also plausible these particular words reflect
genuine contextual ambiguity ("a strange feeling" really can go either
way) rather than a clean labelling-frequency imbalance like `frustrating`
was. Worth knowing before assuming this recipe (spot a skewed word, add
a handful of counter-examples) generalises to every blind spot found this
way - it fixed one real case, not two.

**Suggested next step if picked back up:** a meaningfully larger batch
(15-20 examples per word/class gap instead of 5-6) would be a fairer test
of whether more data helps here, or accept that fear/surprise and
anger/sadness sit closer to genuine class-boundary ambiguity - similar to
the `joy`/`love` confusion, which is this model's single largest error
and wasn't attempted here for exactly that reason.

---

## 4. Jointly-trained multimodal model

**What it means:** Right now, Multimodal mode runs two independently
trained models and blends their output probabilities after the fact
(decision-level fusion). A *jointly-trained* model would instead learn
from text and audio together in one network - e.g., a text encoder and an
audio encoder feeding into one shared classifier trained end-to-end.

**Why this is the biggest lift here:** it needs **paired** data - the same
utterance with both a transcript and matching audio, labelled once. The
current datasets aren't paired (the 20k text dataset and RAVDESS have no
relationship to each other), so this isn't an incremental step on top of
what exists - it requires bringing in a genuinely different dataset like
[IEMOCAP](https://sail.usc.edu/iemocap/) or
[MELD](https://affective-meria.github.io/MELD/), both built specifically
for paired multimodal emotion recognition. Worth flagging: IEMOCAP
requires signing a research-use license agreement to access (not just a
download), which adds lead time before any modelling work can start.

**Effort:** Highest of the five. It also structurally depends on item 1
(a joint model is a deep learning architecture almost by definition - the
current SVM/Gradient Boosting pair has no natural way to be "jointly
trained"). Realistically this is a 3.0-version goal that supersedes the
current fusion approach entirely, not something to start before items 1
and 2 have landed.

---

## 5. Real-time streaming emotion detection

**What it means:** Continuous, live analysis instead of the current
record-then-submit flow - text scored as you type, audio scored on a
rolling window while you talk instead of after you stop and click
"Analyse."

**Text half (cheap, do this first if you want a quick win):** No
architectural problem at all - the TF-IDF+SVM pipeline is fast enough to
re-run on a debounce timer as text changes. Mostly a UI change (Streamlit
supports on-change callbacks/fragments for this) rather than a new
capability.

**Audio half (the actually hard part):** Streamlit's core model is
"rerun the whole script on each interaction," which doesn't naturally fit
a continuously-running microphone stream. Doing this properly needs
either the `streamlit-webrtc` component (a real-time audio/video component
built for exactly this) or a background thread reading from `sounddevice`
into a queue that the UI polls - both add real architectural complexity
on top of the ML, which is why this is rated higher effort despite the
underlying models not changing.

**Effort:** Low (text) / Medium-High (audio), and they're genuinely
separable - shipping live-as-you-type text scoring doesn't require
solving the streaming-audio problem at all.

---

## Suggested sequencing

~~If picking a starting point: item 3 (fairness/generalisation audit) is
the cheapest and most natural next step~~ **Done, 2026-08-15** - see item 3
above. ~~After that, item 1's text half (DistilBERT) is the most clearly
motivated upgrade~~ **Also done, 2026-08-15** - see item 1 above, though
it didn't land the specific win it was chosen for (negation handling),
which is itself worth knowing before investing further here.

Remaining open items: item 2 (more data - GoEmotions for text, CREMA-D/
TESS for audio; note CREMA-D is already merged in as of the fairness-audit
work above, so the audio half of this is partly done already) would pair
naturally with taking another pass at item 1's negation problem - more
data, specifically more negation-containing examples, oversampled, is the
most likely next lever given 12.1% negation coverage wasn't enough to
learn it in 2 epochs on a 3,000-row subset. Items 4 and 5's audio half
remain longer-term - item 4 in particular shouldn't start until item 1's
text half is actually working, and item 5's audio half is worth deferring
until there's a specific reason to want live-streaming audio rather than
upload/record.
