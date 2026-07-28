# The numerics: which arithmetic a round runs in

Every round pins its arithmetic, and participants either match it exactly or sit
the round out. There is no third option where they participate approximately.

```
arithmetic     bf16 params, bf16 compute, fp32 accumulate, flash attention,
               int8 contributions (class 0x1730f203)
```

## Why it has to be pinned

Everything rests on one property: a verifier re-runs a worker's inner loop and
gets **byte-identical** output. That works only if everyone computes identically,
and floating-point arithmetic is not identical across dtypes, kernels, or
reduction orders.

Measured on one card, same weights, same batch:

```
flash attention   loss 5.7656602859    repeat 5.7656602859    bit-identical
math attention    loss 5.7659835815    repeat 5.7659835815    bit-identical
                  difference 3.2e-04
```

Each kernel is individually reproducible. They are not reproducible against each
other. A worker free to pick one would compute an update nobody could reproduce
and would be indistinguishable from a worker that cheated.

The difference is small enough that nobody would ever notice it as "wrong",
which is exactly why it must be pinned in a hashed artifact rather than watched
for.

## What can be pinned

| field | values |
| --- | --- |
| `param_dtype` | bf16, fp16, fp32, **fp8_e4m3**, fp8_e5m2 |
| `compute_dtype` | same set |
| `accum_dtype` | same set, never narrower than compute |
| `attention_kernel` | flash (16-bit only), math (portable) |
| `contribution_format` | **int4**, int6, int8, all with a power-of-two scale |

Validation is about the object, never about the machine: an fp8 round parses on
a machine that cannot run it, because an artifact must not mean different things
on different hardware.

## The determinism class

Two workers reproduce each other if and only if their numerics agree — and the
class is **derived from the canonical bytes** rather than assigned from a
registry:

```python
determinism_class = sha3("rnet/determinism-class/v1" ‖ numerics)[:4]
```

Self-certifying: two independently configured nodes land in the same class
exactly when they will actually reproduce each other's output, and nobody has to
keep a registry honest. A network can run several classes at once, each
internally verifiable.

A verifier in a different class returns **INDETERMINATE**, never MISMATCH.
Reporting a mismatch it cannot distinguish from its own circumstances would
slash an honest worker for owning different hardware.

## Contribution quantisation

Three properties, each load-bearing:

- **The scale is a power of two**, so applying it touches only a float's
  exponent and dequantising is exact. A decimal scale would introduce a rounding
  whose result depends on operation order.
- **The range is symmetric**: two's complement offers one extra negative value,
  and using it would bias the mean of every update in a direction nobody chose,
  forever, by half a step.
- **Ties round away from zero**, not to even. numpy's default is to even; the
  rule that matters is that every implementation applies the same one, and
  "whatever numpy does" is not a specification.

The exponent is chosen with `math.frexp`, which reads the exponent out of the
float's bits, rather than `log2`, which computes it — and different C libraries
compute it differently in the last place. A worker one exponent off produces an
update twice the size it meant.

## What "quantisation" does not mean here

`Q4_K_M`, `Q5_K`, `Q8_0` and friends are **inference** formats from GGUF. They
are not training precisions and nothing in this protocol produces them. A
trained model is exported to GGUF and `llama-quantize` produces the whole range;
that is a converter, not part of consensus.

## Thread count is not consensus

Measured across 1, 2, 4 and 8 threads: identical loss to ten decimal places and
an identical hash over every gradient. It changes speed by more than an order of
magnitude in the wrong direction if left alone —

```
 1 thread    139 GFLOPS
 8 threads   397 GFLOPS
24 threads    10.9 GFLOPS      ← oversubscription
```

— so `tune_cpu_threads()` caps it. That it changes no bytes had to be
established rather than assumed: a reduction whose order varied with thread
count would put every CPU worker in its own determinism class.
