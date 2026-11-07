# Mistral Large 4 on 4× DGX Spark (GB10) with TensorFold: 1M context on one stream

**BertholomusAI recipe: Mistral AI's Mistral Large 4 served over four NVIDIA DGX Spark / GB10 nodes by the
`mistral_large4` model family we wrote for [TensorFold](https://github.com/ashhart/TensorFold) (by ashhart), with the
model's full 1,048,576-token context on one stream.**

> **Weights:** [bertholomus/Mistral-Large-4-1T-A52B-EXL3-2.5bpw](https://huggingface.co/bertholomus/Mistral-Large-4-1T-A52B-EXL3-2.5bpw)
> (315.2 GiB, and 3.9 GiB for the draft head), our EXL3 conversion of
> [mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) at revision `f12dabdc`.
> Model and weights are Mistral AI's, under the Mistral Ecosystem Open-Weight License. Read [License](#license) before
> you use them.
>
> This project is not affiliated with or endorsed by Mistral AI, NVIDIA or the TensorFold authors.

**Engine:** commit `f4a30b3e` on
[bertholomus/TensorFold, branch `mistral-large-4-tp4`](https://github.com/bertholomus/TensorFold/tree/mistral-large-4-tp4):
our `mistral_large4` family on TensorFold 0.6.6's Python CUDA engine, on top of our DeepSeek-V4.1 engine line. It holds
the code we served and measured, less our internal notes and paths (see [Exported build](#exported-build)). The model math follows the
Mistral Large 4 reference in Hugging Face Transformers.

Mistral Large 4 has 72 layers, MLA attention with a sparse indexer (top 2,048 rows), 128 routed experts (4 active) and
a shared expert, and a 1,048,576-token window. This release serves text only: the vision encoder is not in the pack. The draft head Mistral
ships with the model sits beside the pack, unchanged, and decoding uses it.

## v0.2.0: what changed vs v0.1.0

**Decoding now uses the draft head that Mistral ships with the model.** Each step the head proposes three tokens and the
model checks them in one pass. A token goes out only if the model itself picks it, so the text does not change: greedy
replies on our 12 fixed prompts are the same tokens with the head on or off. What changes is how many tokens a step gives,
and that depends on the text. One stream, tok/s, three runs each:

| Prompt | v0.1.0, any text | v0.2.0, explaining code | v0.2.0, an essay | v0.2.0, filler text |
|---|---|---|---|---|
| 8K | 19.8-19.9 | **29.9-30.3** | 25.3-25.4 | 37.4-40.8 |
| 32K | 19.5-19.6 | **29.8-30.0** | 26.1-26.2 | 37.9-39.5 |
| 127K | 18.4-18.6 | **29.2-29.4** | 25.8 | 37.6-39.8 |
| 1M (one run) | 12.89 | | 21.69 | |

The head keeps about 2.5 tokens a step on code, 2.1 on the essay and 3.3 on filler. At the model's default temperature of
1.0 the code prompts ran at 29.2-29.3 tok/s at 8K and 29.7-29.8 at 32K, so sampling costs little.

**Also new:**

- **Decode at 1M depth:** 21.69 tok/s, up from 12.89.
- **Eight streams:** 64.4 tok/s in all, up from 59.4: the step got shorter. Weights are read into L2 beside it (`TF_ML4_L2_MB`),
  and prebuild compiles one more extension for that.
- **A long prompt keeps moving under load.** Prompts run shortest first, so a short request overtakes a long one, and every
  eighth slice of prompt work goes to the prompt that came first (`TF_ML4_CP_OLDEST`).
- **Replies:** a request that sets no `max_tokens` gets 8,192 tokens, up from 512, because the model thinks first.
- **Memory:** the launch line keeps 5 GiB free on a node and caps the allocator at 103 GiB (`TF_ML4_MEM_CEILING_GIB`).

**What is not done, or not met:**

- **Prompt speed is close to v0.1.0's:** 1,675-1,856 tok/s up to 127K, and a full 1M prompt takes about 13.4 minutes.
- **Under a full load, long prompts wait.** With four chat loops and a 128K prompt loop beside it, a prompt of 269,593 tokens
  took 52 minutes to its first token. Alone it takes about 4 minutes.
- **Drafts need room in a step.** Eight streams take none, and two to four streams share the step's eight rows.
- **We use the head as Mistral ships it,** and have not tuned it on this pack.
- **Host memory on the first node grew about 0.1 GiB an hour** in the two hours of load, in malloc's per-thread arenas. We have
  not run past that at a stretch, so watch MemAvailable on a lane that stays up for days under load.
- **The conversion code is not published yet**, so the pack can be downloaded but not remade from the original.
- **What 2.5 bits costs against the original weights is not measured.** We measured the engine against a reference on
  the same weights, not the weights against the original model.
- We have run this on one set of four nodes and one fabric.

## Measured

Four GB10 nodes (128 GB of unified memory each) on a 200 Gb/s RoCE fabric, GPU clocks capped at 2,300 MHz, draft head on,
greedy (temperature 0) unless a row says otherwise. Tools are in the engine's `tools/ml4/`. Every row below ran on this engine
commit with the launch line in this README, after a start from an empty folder ([Exported build](#exported-build)), the two
hours of mixed load included.

**Decode by text** (`cp4_ask.py --max-tokens 800 --effort low`; tok/s, a range over three runs). Code asks the model to
explain the source of a serving engine, the essay sets a work log aside and writes on suspension bridges, and filler
(`cp4_load.py streams --tokens 800`) continues a text of random words, which the head finds easy:

| Prompt | Code | Essay | Filler |
|---|---|---|---|
| 8K | 29.94-30.34 | 25.27-25.37 | 37.4-40.8 |
| 32K | 29.75-30.02 | 26.07-26.19 | 37.9-39.5 |
| 127K | 29.16-29.40 | 25.81-25.84 | 37.6-39.8 |

Prompts of code have 7,731, 33,055 and 126,661 tokens, essay prompts 9,405, 36,237 and 144,752, and filler prompts 8,041-8,078,
32,057-32,125 and 126,976-127,214.

At 1,044,260 tokens (one run, `cp4_ask.py`, three codes and then a 500-word essay): 21.69 tok/s over 522 tokens, with the
head keeping 2.3 tokens a step falling to 1.6. At the default temperature of 1.0 (`--temperature 1`, two runs each), code ran
at 29.16-29.30 tok/s at 8K and 29.72-29.76 at 32K.

**Prompts** (`cp4_load.py streams`, one request at a time, three runs, ranges over the runs; prompts are cut from one long
text at a random place, so no two share a start; tok/s of the prompt is prompt tokens over time to the first token):

| Prompt tokens | Time to first token | Prompt tok/s |
|---|---|---|
| 8,041-8,078 | 4.7-4.8 s | 1,675-1,719 |
| 32,057-32,125 | 17.3-17.7 s | 1,811-1,856 |
| 126,976-127,214 | 72.5-72.6 s | 1,751-1,752 |
| 120,163 (one run, striped, see [How it runs](#how-it-runs)) | 68.8 s | 1,747 |
| 1,044,260 (one run, the 1M needle below, five short requests beside it) | 803.5 s (13.4 min) | 1,300 |

**Eight streams at once** (`cp4_load.py streams --lens 8000,12000,16000,20000,24000,28000,32000,32000 --tokens 4800`,
one run): **64.4 tok/s in all** over the 217.9 s that all eight decoded together, every stream at 8.1 tok/s. Prompts run one after
another, so the eight first tokens came up to 284.7 s after the start. Eight streams leave a step no room for drafts.

**A short request beside a long prompt** (`cp4_load.py ttft`, requests of about 2,000 tokens one after another with
5 s between them):

| Long prompt | Short requests | Time to first token | Long prompt's first token |
|---|---|---|---|
| 120,166 tokens | 12 | 1.95-2.73 s | 90.0 s (1,335 tok/s) |
| 300,491 tokens | 31 | 1.96-2.35 s | 232.3 s (1,293 tok/s) |
| 1,044,260 tokens, once about 900,000 are in | 5 | 2.8-3.0 s | 803.5 s |

**Two hours of mixed load** (`cp4_load.py soak --seconds 7200 --long 300000`): four chat loops, a loop of prompts up to 128K and a loop of prompts up to 300K, all at once for the whole two hours.
1,379 chats, 13 prompts of 66,019-124,213 tokens and 3 long prompts: **0 errors, no out-of-memory, no hang**. With the lane this
full every prompt waits its turn. A chat's first token came in a median of 2.68 s, with a slowest of 30.6 s. Prompts of up to
124K took 265-772 s, and the long ones 1,544 s (155,709 tokens), 2,527 s (223,096) and 3,093 s (269,593).

**Finding things in a long prompt** (`cp4_ask.py`, three registration codes hidden in a long text, one run each):
3 of 3 at 1,044,260 tokens and 3 of 3 at 120,163 tokens.

**Heat and memory** (each node sampled every 2 to 5 s): through all the rows above, highest temperature 84, 81, 86 and 82 °C and lowest MemAvailable 6.63, 6.82,
6.75 and 6.68 GiB. Through the two hours of load, 80, 79, 80 and 83 °C and 6.80, 7.07, 7.27 and 7.13 GiB. A wide prompt round waits for a rank at 85 °C (`TF_ML4_CP_MAX_TEMP`): that happened 55 times in the rows, all
but two on one node, and not at all in the two hours.

**Chat and tools** (`tools/ml4_step4_chat.py`): 6 of 6: plain chat, a multi-turn exchange, a coding task checked by
running the code, a single tool call, a chained pair of tool calls, and a code recovered from an 8K context.
Streamed tool-call arguments ([`tools/toolstream.py`](tools/toolstream.py), two calls, temperature 0) join to JSON that
parses and equals the non-streamed reply's call: 2 of 2.

**Loss against the exllamav3 reference.** We ran the same pack through our own exllamav3 loader for this model
(not published with this release), streamed module by module, and through this engine's striped path
(`cp4_run.py --tokens ... --nll 1024`, the served settings), on the same tokens: two texts of 2,047 predicted tokens.
Mean next-token loss in nats:

| Text | exllamav3 | This engine | Difference (± 2 s.e.) | Top token the same |
|---|---|---|---|---|
| Web text, then encyclopedia | 2.5177 | 2.5165 | -0.0012 ± 0.0086 | 1,946 of 2,047 (95.1 %) |
| Encyclopedia, then web text | 2.8772 | 2.9727 | +0.0954 ± 0.0681 | 1,783 of 2,047 (87.1 %) |

The reference keeps its cache in fp16 and the engine in fp8. The first text is our test text. The second breaks off
inside an unclosed document header, and from there on its rows swing with small rounding changes in any build: 60 of
its 2,047 rows carry about 80 % of the difference. These are the same numbers as in v0.1.0: every row matches that run's loss
bit for bit. **What 2.5 bits costs against the original weights is not measured.**

**What the long-prompt shortcuts cost.** Prompts past 163,840 rows take the shortcuts in [How it runs](#how-it-runs).
Teacher-forced loss on 295K tokens of mixed text: FP4 index scores +0.0005 ± 0.0017; twelve front layers sharing a
selection, with the small matrices in TF32, +0.0028 ± 0.0021, and on a code text +0.0008 ± 0.0008 (differences of the
mean, paired by row, ± two standard errors). The tokens decoded after those prompts lost nothing measurable.

## Get the weights

Each node needs the whole pack, in the same folder: 320 GiB with the draft head, plus about 90 GiB for the rank cache that
the first start writes. Download it once and copy it to the other nodes over the fabric:

```bash
hf download bertholomus/Mistral-Large-4-1T-A52B-EXL3-2.5bpw --local-dir "$ML4_PACK"
cd "$ML4_PACK" && sha256sum -c sha256.txt     # every file but ml4_pack.json and this card
```

If you already have the v0.1.0 pack, add the draft head with `--include 'draft/*'`. The head is `draft/mtp_dspark.safetensors`
in the pack's folder: Mistral's own tensors, unchanged, under the same license. `DRAFT=0` serves without it, at v0.1.0's speed.

## Run it

You need four GB10 nodes with nothing else on them, reachable by ssh as `spark1` to `spark4` from the machine you
start from, Docker with the NVIDIA runtime, and an RDMA fabric between the nodes.

**1. On every node,** fetch the engine and the container, and build the CUDA extensions (3.5 minutes on our nodes):

```bash
export ML4_TREE=$HOME/ml4                                   # the code, logs, extensions and rank cache
mkdir -p "$ML4_TREE/src" "$ML4_TREE/logs"
git clone -b mistral-large-4-tp4 https://github.com/bertholomus/TensorFold "$ML4_TREE/src/tensorfold"
git -C "$ML4_TREE/src/tensorfold" checkout f4a30b3e
docker pull nvcr.io/nvidia/pytorch:26.07-py3
git clone -b v0.2.0 https://github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10 "$ML4_TREE/recipe"
bash "$ML4_TREE/recipe/serve/prebuild.sh"
```

**2. From your own machine,** in a copy of this repository, start the four ranks:

```bash
export ML4_TREE=/home/you/ml4                               # the same on every node
export ML4_PACK=/data/Mistral-Large-4-1T-A52B-EXL3-2.5bpw   # the same on every node
export ML4_MASTER=10.0.0.1                                  # rank 0's address on the fabric
export ML4_IFNAME=<the fabric's network interface>
export ML4_HCA=<the RDMA devices, comma separated>          # ibdev2netdev lists them
bash serve/start_lane.sh ml4
```

The first start writes each rank's cache from the pack (about 2.5 minutes on our nodes, with the pack on NVMe); later starts read the cache
(24 s on a restart, with the file still in the page cache). The launch uses `NCCL_IB_GID_INDEX=5`, our fabric's RoCE v2 entry: check yours with `show_gids` and pass
`EXTRA='-e NCCL_IB_GID_INDEX=<n>'` if it differs. Each rank's log is `$ML4_TREE/logs/ml4-rank<r>.log`.

**3. Ask it something,** on `spark1`. The API is OpenAI compatible and listens on rank 0 at `127.0.0.1:8561`
(`bash serve/start_lane.sh ml4 --host 0.0.0.0` listens on every interface):

```bash
curl -s 127.0.0.1:8561/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ml4", "messages": [{"role": "user", "content": "Say hello in one line."}],
  "max_tokens": 300, "reasoning_effort": "low"}'
```

The model's thinking comes back in `reasoning_content` and counts against `max_tokens`, so a small limit can end inside the
thinking: `finish_reason` is `length` and `content` is empty. A request that does not set `max_tokens` gets 8,192 tokens.
`reasoning_effort` takes `low`, `medium` or `high`, the default. Tool calls use the OpenAI `tools` field, with `tool_choice`
`auto` or `none`. Without `temperature`, requests sample at the model's default of 1.0.

**Not supported in this release:**

- `tool_choice` set to `required` or to a named function: a 400, because the chat template has no tool-call marker to force.
- `response_format` of `json_object` or `json_schema`: a 400. Both need xgrammar, which the serving container does not have, and
  we have not tried them with it.
- `reasoning_effort` of `none`: the model has no off switch, so its thinking and the `[THINK]` marker come back in `content`.
  Use `low`. `xhigh` is refused with a 400 by the chat template.

**4. Stop it** with `bash serve/stop_lane.sh ml4`.

### The launch line

`tools/ml4/cp4_serve.sh` in the engine runs one rank; `serve/start_lane.sh` runs it on all four. Every rank gets the same
switches:

```bash
EXL3_INT8_GEMV=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
NCCL_IB_GID_INDEX=5 NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TF_ML4_CP_BLOCK=64 TF_ML4_CP_ROUND=16384 TF_ML4_CP_PAIR=8192 TF_ML4_CP_SLICE=1 TF_ML4_CP_SCORE_MB=512
TF_ML4_PROMPT_KERNEL=mma8 TF_ML4_RANK_CACHE=/w/rank-cache
TF_ML4_KV_TOKENS=262144 TF_ML4_MEM_FLOOR_GIB=5 TF_ML4_MEM_CEILING_GIB=103 TF_ML4_HEADROOM_GIB=3.0
TF_ML4_POOL_SAFETY_GIB=0 TF_ML4_L2_MB=16,8,16 TF_ML4_CP_MAX_TEMP=85
TF_ML4_DRAFT="$ML4_PACK/draft/mtp_dspark.safetensors"

python3 -m tensorfold serve "$ML4_PACK" --tp 4 --rank <r> --master "$ML4_MASTER" --master-port 29561 \
  --port 8561 --context 1048576 --max-tokens 8192
```

## How it runs

- **Experts** are split by rows, so each node holds a quarter of every expert. Decode is one four-rank step in a CUDA graph.
- **The draft head** runs on all four ranks between steps, about 6.4 ms a draft, from the taps of the model's own
  residual stream and a window of the last 2,048 rows. A step checks up to four rows at once; it takes 79.5 ms at 28,672 tokens
  of context against 44.8 ms for one row, so a draft pays once more than about 1.9 tokens a step stay. With many streams
  in a step there is no room for drafts: a step stays at 8 rows, and eight streams decode one row each.
- **The cache** holds fp8 latent rows, 51,440 bytes a token over all layers: 262,144 rows per node, 12.56 GiB, and
  1,048,576 rows over the four.
- **A request of up to 131,072 tokens,** prompt and reply together, lives on one node's rows. Several run at once, and
  their decode steps run together.
- **A longer request is striped** over the four nodes in blocks of 64 tokens, one such request at a time.
  Short requests go between its layers, so they don't wait for it.
- **Past 163,840 rows** a prompt changes gear: each layer's rows are gathered one layer ahead, and three shortcuts
  start: index scores in FP4; twelve of the front layers reuse an earlier layer's selection, as the model itself does
  from layer 17 on; and the small matrices that feed the cache and the index run in TF32. Decode keeps every layer's
  own selection. [What they cost](#measured) is measured above.

`tools/ml4/DESIGN-CP4.md` in the engine has the design and the reasoning behind it.

## Exported build

We started this engine commit from an empty folder on our four nodes with the steps above: clone, `prebuild.sh`,
`start_lane.sh`, with nothing kept from the working tree, not even a rank cache. `prebuild.sh` built eight extensions in 216-222 s on each node, the first start took 136 s and a start from the rank cache
26 s. Every row in [Measured](#measured), the two hours of load included, ran on that lane, and its greedy replies on 12 prompts (1,130
tokens, `cp4_greedy.py`) were the same tokens before and after those rows, and the same as with the head off. The source differs from the
build we developed on in comments, log text and paths only. That run used our nodes' copy of the
pack, whose manifest differs from the published one in two text fields.

## Credits

- **[Mistral AI](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)** made the model.
- **[TensorFold](https://github.com/ashhart/TensorFold)** by ashhart is the engine this family plugs into (Apache-2.0).
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp gives the EXL3 format and its converter (MIT).
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)**: the Mistral Large 4 modeling code was
  our reference for each operation (Apache-2.0). No code was copied.
- `tools/ml4/ATTRIBUTION.md` in the engine lists every source, with the reference code we read for the draft head, and its `NOTICE` lists every upstream file we changed.

## License

The files in this repository are ours, under the [Apache License 2.0](LICENSE). The engine is under TensorFold's
license, with the fork's changes listed in its `NOTICE`.

**The weights are not.** They are a derivative of Mistral Large 4 and stay under the
[Mistral Ecosystem Open-Weight License](https://huggingface.co/bertholomus/Mistral-Large-4-1T-A52B-EXL3-2.5bpw/blob/main/LICENSE),
a modified MIT license (attribution notice: 2026 - Mistral), and its notice must go with every copy. Condition 2 of that
license matters most: if you or an affiliate run a Model as a Service business, you must enter into "a separate
commercial license with Mistral" before using the model or its derivative works for any commercial purpose. Internal
use is exempt under condition 3, and the license defines these terms: read it before you use the weights.

"Mistral" and "Mistral Large" belong to Mistral AI; "DGX Spark" and "GB10" belong to NVIDIA. This project is not
affiliated with or endorsed by Mistral AI, NVIDIA or the TensorFold authors.
