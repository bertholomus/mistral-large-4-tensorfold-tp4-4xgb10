# Mistral Large 4 on 4× DGX Spark (GB10) with TensorFold: 1M context on one stream

**BertholomusAI recipe: Mistral AI's Mistral Large 4 served over four NVIDIA DGX Spark / GB10 nodes by the
`mistral_large4` model family we wrote for [TensorFold](https://github.com/ashhart/TensorFold) (by ashhart), with the
model's full 1,048,576-token context on one stream.**

> **Weights:** [bertholomus/Mistral-Large-4-1T-A52B-EXL3-2.5bpw](https://huggingface.co/bertholomus/Mistral-Large-4-1T-A52B-EXL3-2.5bpw)
> (315.2 GiB), our EXL3 conversion of
> [mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) at revision `f12dabdc`.
> Model and weights are Mistral AI's, under the Mistral Ecosystem Open-Weight License. Read [License](#license) before
> you use them.
>
> This project is not affiliated with or endorsed by Mistral AI, NVIDIA or the TensorFold authors.

**Engine:** commit `46af2fe1` on
[bertholomus/TensorFold, branch `mistral-large-4-tp4`](https://github.com/bertholomus/TensorFold/tree/mistral-large-4-tp4):
our `mistral_large4` family on TensorFold 0.6.6's Python CUDA engine, on top of our DeepSeek-V4.1 engine line. It holds
the code we served and measured; only our internal notes and paths are taken out. The model math follows the
Mistral Large 4 reference in Hugging Face Transformers.

Mistral Large 4 has 72 layers, MLA attention with a sparse indexer (top 2,048 rows), 128 routed experts (4 active) and
a shared expert, and a 1,048,576-token window. This release serves text only: the vision encoder and the MTP draft
layers are not in the pack.

## v0.1.0 (2026-11-06)

**What works:** one stream at the full 1,048,576-token window, with the three codes hidden in a 1,044,249-token prompt
all found; a prompt of that length in about 14 minutes; 18-20 tok/s of decode up to 127K tokens and about 13 tok/s at
1M; a short request's first token in under 2 seconds beside a long prompt; chat and tool calls.

**What is not done, or not met:**

- **The 2-hour mixed soak has not run on this build.** Everything below comes from runs of minutes, not hours.
- **Eight streams at once came in at 59.4 tok/s**, under our own target of 60 (one run; every stream held 7.4 tok/s).
- **Memory ran below our 8 GiB floor during a 1M prompt on a lane that had already served other load:** the lowest
  MemAvailable was 7.54 GiB on rank 0, and that node stayed under 8 GiB for most of the prompt. On a freshly started
  lane the low was 7.96 GiB. Nothing failed, but a GB10 that runs out of memory takes the node down, so leave nothing
  else on the nodes.
- **One short request took 3.17 s**, as the long prompt beside it finished. The other 31 took 1.69-1.92 s.
- **The conversion code is not published yet**, so the pack can be downloaded but not remade from the original.
- **What 2.5 bits costs against the original weights is not measured.** We measured the engine against a reference on
  the same weights, not the weights against the original model.
- We have run this on one set of four nodes and one fabric.

## Measured

Four GB10 nodes (128 GB of unified memory each) on a 200 Gb/s RoCE fabric, GPU clocks capped at 2,300 MHz, the launch
line below, greedy (temperature 0). Prompts are cut from one long text at a random place, so no two share a
start. Tools are in the engine's `tools/ml4/`. These runs used the build the engine commit was exported from; the
export changes comments, notes and paths only (see [Exported build](#exported-build)).

**One stream** (`cp4_load.py streams --lens N --tokens 300`, one request at a time, three runs, ranges over the runs;
tok/s of the prompt is prompt tokens over time to the first token):

| Prompt tokens | Time to first token | Prompt tok/s | Decode tok/s |
|---|---|---|---|
| 8,041-8,078 | 4.8-4.9 s | 1,641-1,680 | 19.8-19.9 |
| 32,057-32,125 | 18.1-18.2 s | 1,761-1,775 | 19.5-19.6 |
| 126,976-127,214 | 76.1-76.8 s | 1,656-1,669 | 18.4-18.6 |
| 131,181 (one run, striped, see [How it runs](#how-it-runs)) | 78.9 s | 1,662 | |
| 1,044,249 (one run, the 1M needle below) | 833 s (13.9 min) | 1,253 | 12.89 over 67 tokens |

The 1M run came right after the other rows, on warm nodes. The build before this one, which differs only in when the
tokenizer's buffers are freed, took 814.6 s from a cool start (1,282 tok/s), and decoded at 12.55 tok/s.

**Eight streams at once** (`cp4_load.py streams --lens 8000,12000,16000,20000,24000,28000,32000,32000 --tokens 2400`,
one run): **59.4 tok/s in all** over the 114.7 s that all eight decoded together, every stream at 7.4 tok/s. Prompts run
one after another, so the eight first tokens came 5.2 s to 258.6 s after the start.

**A short request beside a long prompt** (`cp4_load.py ttft`, requests of about 2,000 tokens one after another with
4-6 s between them):

| Long prompt | Short requests | Time to first token | Long prompt's first token |
|---|---|---|---|
| 120,190 tokens | 13 | 1.73-1.92 s; the last, as the long prompt finished, 3.17 s | 94.6 s (1,270 tok/s) |
| 300,530 tokens | 19 | 1.69-1.91 s | 221.2 s (1,359 tok/s) |

**Finding things in a long prompt** (`cp4_ask.py`, three registration codes hidden in a long text, one run each):
3 of 3 at 1,044,249 tokens and 3 of 3 at 120,163 tokens.

**Heat and memory during the 1M run** (each node sampled every 5 s): highest temperature 84, 82, 85 and 82 °C, lowest
MemAvailable 7.54, 7.97, 8.03 and 7.91 GiB. A rank pauses between steps at 85 °C (`TF_ML4_CP_MAX_TEMP`).

**Chat and tools** (`tools/ml4_step4_chat.py`): 6 of 6: plain chat, a multi-turn exchange, a coding task checked by
running the code, a single tool call, a chained pair of tool calls, and a code recovered from an 8K context. Streamed
tool-call arguments ([`tools/toolstream.py`](tools/toolstream.py), two calls, temperature 0) join to JSON that parses and
equals the non-streamed reply's call: 2 of 2.

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
its 2,047 rows carry about 80 % of the difference. **What 2.5 bits costs against the original weights is not measured.**

**What the long-prompt shortcuts cost.** Prompts past 163,840 rows take the shortcuts in [How it runs](#how-it-runs).
Teacher-forced loss on 295K tokens of mixed text: FP4 index scores +0.0005 ± 0.0017; twelve front layers sharing a
selection, with the small matrices in TF32, +0.0028 ± 0.0021, and on a code text +0.0008 ± 0.0008 (differences of the
mean, paired by row, ± two standard errors). The tokens decoded after those prompts lost nothing measurable.

## Get the weights

Each node needs the whole pack, in the same folder: 316 GiB, plus about 90 GiB for the rank cache that the first start
writes. Download it once and copy it to the other nodes over the fabric:

```bash
hf download bertholomus/Mistral-Large-4-1T-A52B-EXL3-2.5bpw --local-dir "$ML4_PACK"
cd "$ML4_PACK" && sha256sum -c sha256.txt     # every file but ml4_pack.json and this card
```

## Run it

You need four GB10 nodes with nothing else on them, reachable by ssh as `spark1` to `spark4` from the machine you
start from, Docker with the NVIDIA runtime, and an RDMA fabric between the nodes.

**1. On every node,** fetch the engine and the container, and build the CUDA extensions (3.5 minutes on our nodes):

```bash
export ML4_TREE=$HOME/ml4                                   # the code, logs, extensions and rank cache
mkdir -p "$ML4_TREE/src" "$ML4_TREE/logs"
git clone -b mistral-large-4-tp4 https://github.com/bertholomus/TensorFold "$ML4_TREE/src/tensorfold"
git -C "$ML4_TREE/src/tensorfold" checkout 46af2fe1
docker pull nvcr.io/nvidia/pytorch:26.07-py3
git clone https://github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10 "$ML4_TREE/recipe"
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

The model's thinking comes back in `reasoning_content` and counts against `max_tokens`; a request that does not set
`max_tokens` gets 512 tokens. `reasoning_effort` takes `low`, `medium` or `high`, the default. The model always thinks:
`"none"` does not stop it, and its thinking then lands in `content`, markers and all, so don't send `"none"`. Tool
calls use the OpenAI `tools` field. Without `temperature`, requests sample at the model's default of 1.0.

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
TF_ML4_KV_TOKENS=262144 TF_ML4_MEM_FLOOR_GIB=8.75 TF_ML4_HEADROOM_GIB=3.0 TF_ML4_POOL_SAFETY_GIB=0
TF_ML4_CP_MAX_TEMP=85

python3 -m tensorfold serve "$ML4_PACK" --tp 4 --rank <r> --master "$ML4_MASTER" --master-port 29561 \
  --port 8561 --context 1048576 --max-tokens 512
```

## How it runs

- **Experts** are split by rows, so each node holds a quarter of every expert. Decode is one four-rank step in a CUDA graph.
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
`start_lane.sh`, with nothing kept from the working tree, not even a rank cache. It came up in 135 s. Its greedy replies
on 12 prompts (1,130 tokens, `cp4_greedy.py`) are the same tokens as the working build's, the chat and tool set passed
6 of 6, and streamed tool-call arguments 2 of 2. The source differs from the build we measured in comments, log text
and paths only: a token-level comparison of the code finds no other change. We did not repeat the speed rows on it.
That run used our nodes' copy of the pack, whose manifest differs from the published one in two text fields.

## Credits

- **[Mistral AI](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B)** made the model.
- **[TensorFold](https://github.com/ashhart/TensorFold)** by ashhart is the engine this family plugs into (Apache-2.0).
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp gives the EXL3 format and its converter (MIT).
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)**: the Mistral Large 4 modeling code was
  our reference for each operation (Apache-2.0). No code was copied.
- `tools/ml4/ATTRIBUTION.md` in the engine lists every source, and its `NOTICE` lists every upstream file we changed.

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
