# Phase B -- abstention behavioural check

**What this is.** Every question in the frozen eval set (`questions.v2.jsonl`, 65 `has_answer: true` + 17 `has_answer: false`, labels fixed by pooled human judging in Phase E) run once through `answer()`. It records one behaviour: does the model decline when retrieval returns nothing relevant.

**What this is not.** Not a generation quality metric -- not faithfulness, not answer correctness, not citation precision. Nothing here says an answer is *good*, only whether the model abstained or not.

**The number that matters** is the abstention rate on the 65 answerable questions (the false-abstention rate). A model that abstains freely scores well on the negatives for the wrong reason, so the two rates only mean something side by side: the goal is high on the negatives *and* low on the answerable set.

## Config

```json
{
  "generated_at": "2026-09-09T00:13:12.331464+00:00",
  "git_commit": "f8a1112a56f4b81c817220198cbb41b57904c41d",
  "models": [
    "qwen2.5:3b",
    "qwen3:4b"
  ],
  "questions_file": "questions.v2.jsonl",
  "n_answerable": 65,
  "n_negative": 17,
  "mode": "hybrid_dbsf",
  "collection": "vlc_section",
  "top_k": 5,
  "temperature": 0.0,
  "max_tokens": 4096,
  "timeout_seconds": 120.0,
  "llm_base_url": "http://localhost:11434/v1",
  "dense_model": "BAAI/bge-small-en-v1.5",
  "sparse_model": "Qdrant/bm25",
  "prefetch_limit": 100,
  "rrf_k": 2
}
```

## Results

| metric | `qwen2.5:3b` | `qwen3:4b` |
|---|---|---|
| abstention rate (marker), has_answer: false (n=17) — want high | 70.6% (12/17) | 43.8% (7/16) |
| **abstention rate (marker), has_answer: true (n=65) — want low** | **56.9%** (37/65) | **16.4%** (10/61) |
| &nbsp;&nbsp;+ declined in prose without the marker (false: want high / true: want low) | neg +0, true +3 | neg +0, true +0 |
| effective abstention (marker **or** prose), has_answer: false — want high | 70.6% (12/17) | 43.8% (7/16) |
| effective abstention (marker **or** prose), has_answer: true — want low | 61.5% (40/65) | 16.4% (10/61) |
| answers with an invalid citation index | 0 | 0 |
| errored calls (GenerationError) | 0 | 5 |
| median generation latency (successful calls) | 3.4 s | 53.7 s |

The marker rate is the primary metric -- the model is asked to begin a refusal with a fixed token, and phrase-matching English refusals across models is brittle. The prose line is a heuristic caveat: an answer that reads as a refusal and cites nothing. It matters because a model that declines without the marker is still declining, and the marker rate alone would flatter it on the answerable set.

- `qwen2.5:3b` prose refusal on answerable qids (false abstentions the marker missed): q038, q043, q065
- `qwen3:4b` errors: q002: model 'qwen3:4b' at http://localhost:11434/v1/chat/completions did not respond after 3 attempt(s): timed out, q005: model 'qwen3:4b' at http://localhost:11434/v1/chat/completions did not respond after 3 attempt(s): timed out, q010: model 'qwen3:4b' at http://localhost:11434/v1/chat/completions did not respond after 3 attempt(s): timed out, q015: model 'qwen3:4b' at http://localhost:11434/v1/chat/completions did not respond after 3 attempt(s): timed out, q056: model 'qwen3:4b' at http://localhost:11434/v1/chat/completions did not respond after 3 attempt(s): timed out

## Full answers for manual inspection

Three answerable, two not. The FSO question is **not** in the eval set -- it is the probe where the two models diverged (one flagged 25 dB/km as a chosen simulation parameter, the other abstained); shown here for the same manual read, not counted in any rate above.

### `qwen2.5:3b`

**answerable · q034** — 'Why does the need for a clear line of sight hold back light-based links?'

- abstained: `False`  ·  generation: 5.3 s  ·  invalid citations: `[]`
- citations: [1] sensors-21-00861-v2 [3046:4279] 10.3390/s21030861

> The need for a clear line of sight holds back light-based links because the propagation and directionality characteristics of light require a direct path between the transmitter and receiver. This is evident from [1] which states that "Since light does not go through solid elements, it also confers an inherent security umbrella upon VLC, which is a very appreciated property nowadays, where security and safety awareness are growing up in our society. However, the obstructions in the direct link between transmitter and receiver for VLC are critical for communication and it is one of its main disadvantages, which is known as the line-of-sight (LoS) link blockage or shadowing."

**answerable · q038** — 'Can the light that bounces off a wall be strengthened when the direct path is blocked?'

- abstained: `False`  ·  generation: 3.1 s  ·  invalid citations: `[]`
- citations: (none)

> The provided sources do not cover the question.

**answerable · FSO probe (not in eval set)** — 'What is the limit for FSO links with severe fog?'

- abstained: `True`  ·  generation: 2.3 s  ·  invalid citations: `[]`
- citations: (none)

> The provided sources do not cover the question about the limit for FSO links with severe fog.

**not answerable · q022** — 'Why is the data rate so low when a phone camera is used as the receiver?'

- abstained: `False`  ·  generation: 4.7 s  ·  invalid citations: `[]`
- citations: [1] electronics-09-01339-v3 [7147:8090] 10.3390/electronics9091339; [3] hybrid-rf-vlc-systems-a-comprehensive-survey-on-network-topologies-performance-analyses-applications-and-future-directions [6599:7847] 10.1109/access.2021.3129154

> The data rate is low when a phone camera is used as the receiver because the camera's frame rate is typically in the range of 30 to 60 frames per second (fps) at a common resolution of 1920 × 1080 pixels. This is stated in [1] and [3]. Additionally, the use of high-speed cameras of 1000 FPS is possible to overcome this problem, as mentioned in [3]. However, the data rate is still very limited due to the fact that the sampling rate of commercially widely available cameras is not high, as explained in [3].

**not answerable · n002** — 'Does the aging of an LED affect the long-term reliability of a VLC link?'

- abstained: `True`  ·  generation: 2.7 s  ·  invalid citations: `[]`
- citations: (none)

> The provided sources do not cover the question about the effect of LED aging on the long-term reliability of a VLC link.

### `qwen3:4b`

**answerable · q034** — 'Why does the need for a clear line of sight hold back light-based links?'

- abstained: `False`  ·  generation: 47.5 s  ·  invalid citations: `[]`
- citations: [1] sensors-21-00861-v2 [3046:4279] 10.3390/s21030861; [5] hybrid-rf-vlc-systems-a-comprehensive-survey-on-network-topologies-performance-analyses-applications-and-future-directions [42255:42559] 10.1109/access.2021.3129154

> The need for a clear line of sight holds back light-based links because light cannot penetrate solid obstacles, causing blockage when physical objects such as people or furniture obstruct the direct path between transmitter and receiver [1][5].

**answerable · q038** — 'Can the light that bounces off a wall be strengthened when the direct path is blocked?'

- abstained: `False`  ·  generation: 101.4 s  ·  invalid citations: `[]`
- citations: [3] applsci-15-01617-v2 [2212:2991] 10.3390/app15031617

> Yes, the light that bounces off a wall can be strengthened when the direct path is blocked, as indicated by the multipath propagation effects described in source [3]. Specifically, source [3] states: "reflections caused by indoor surfaces can result in multipath propagation, which can enhance or deteriorate the signal, depending on how it is received." This implies that in certain conditions (e.g., when the direct path is blocked by objects, walls, or people), the reflected signal (bounced light) may enhance the overall signal strength, thereby being strengthened. The phrasing "can enhance" confirms that this strengthening is possible, though it depends on the specific reception conditions.
> 
> This conclusion is based solely on the provided sources and adheres to the requirement to distinguish measured/observed phenomena from assumptions or simulation parameters. No external knowledge is added.
> 
> **Citation**: Source [3] (as referenced in the query).

**answerable · FSO probe (not in eval set)** — 'What is the limit for FSO links with severe fog?'

- abstained: `False`  ·  generation: 39.3 s  ·  invalid citations: `[]`
- citations: [2] photonics-12-00649 [25716:26275] 10.3390/photonics12070649

> The simulation uses 25 dB/km as a chosen worst-case attenuation value for severe fog conditions [2]. This value was explicitly selected as a simulation parameter to represent highly lossy channel conditions under adverse environmental factors such as fog, not as a measured or established real-world limit.

**not answerable · q022** — 'Why is the data rate so low when a phone camera is used as the receiver?'

- abstained: `False`  ·  generation: 75.1 s  ·  invalid citations: `[]`
- citations: [1] electronics-09-01339-v3 [7147:8090] 10.3390/electronics9091339; [3] hybrid-rf-vlc-systems-a-comprehensive-survey-on-network-topologies-performance-analyses-applications-and-future-directions [6599:7847] 10.1109/access.2021.3129154

> The data rate is low because the camera's frame rate (reception-sampling rate) is limited to 30-60 fps for common resolutions, resulting in data rates ranging from several bits per second to several kilobits per second due to the requirement for significant image processing to extract information from the camera's 2D image sensor [1][3].

**not answerable · n002** — 'Does the aging of an LED affect the long-term reliability of a VLC link?'

- abstained: `True`  ·  generation: 27.2 s  ·  invalid citations: `[]`
- citations: (none)

> The provided sources do not cover the question of whether LED aging affects the long-term reliability of a VLC link.
