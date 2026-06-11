# Qdrant Quantization Guide

Quantization stores a compressed representation of each vector for search. It
can reduce memory use and improve search speed, but compression may reduce
recall. Qdrant retains the original vectors alongside the quantized vectors.

In this repository, quantization is configured when a collection is created by
`qdrant/scripts/configure_collection.py`. The default is
`QUANTIZATION_TYPE=NONE`, which preserves the existing unquantized behavior.

## Option Summary

| Type | Approximate compression | Strengths | Main tradeoffs |
| --- | ---: | --- | --- |
| `NONE` | 1x | Full-precision baseline | Highest vector memory use |
| `SCALAR` | 4x | Good general-purpose balance of speed, memory, and recall | Small accuracy loss |
| `BINARY` | 16x to 32x | Very high compression and fast comparisons | Data-sensitive; often needs rescoring for good recall |
| `PRODUCT` | 4x to 64x | Greatest configurable memory reduction | Typically slower and less accurate than scalar quantization |
| `TURBO` | 8x to 32x | Strong recall at high compression across many embedding models | Requires Qdrant 1.18 or newer |

Compression values describe the quantized representation relative to the
original `float32` vectors. Actual collection storage also includes original
vectors, payloads, indexes, and database metadata.

## Shared Parameters

### `QUANTIZATION_TYPE`

Selects the quantization method:

```bash
QUANTIZATION_TYPE=NONE
QUANTIZATION_TYPE=SCALAR
QUANTIZATION_TYPE=BINARY
QUANTIZATION_TYPE=PRODUCT
QUANTIZATION_TYPE=TURBO
```

The default is `NONE`.

### `QUANTIZATION_ALWAYS_RAM`

Controls whether Qdrant should keep the quantized vectors in RAM:

```bash
QUANTIZATION_ALWAYS_RAM=False
```

Set this to `True` when low search latency is more important than minimizing
RAM usage. This setting applies to every quantization type.

## Scalar Quantization

Scalar quantization converts each vector component from a 32-bit float to an
8-bit integer. This gives approximately 4x compression for the quantized
representation and is a practical starting point for most experiments.

```bash
QUANTIZATION_TYPE=SCALAR
QUANTIZATION_ALWAYS_RAM=True
QUANTIZATION_SCALAR_QUANTILE=0.99
```

`QUANTIZATION_SCALAR_QUANTILE` is optional and must be greater than zero and
less than or equal to one. It controls the quantization range used to reduce
the effect of outliers. Leave it empty to use Qdrant's default behavior.

Scalar quantization is suitable when:

- moderate memory reduction is sufficient
- recall should remain close to the full-precision baseline
- a conservative first quantization experiment is desired

## Binary Quantization

Binary quantization encodes vector components with one, one-and-a-half, or two
bits. It provides high compression and fast comparisons, but its recall is more
dependent on vector dimensionality and value distribution.

One-bit default:

```bash
QUANTIZATION_TYPE=BINARY
QUANTIZATION_BINARY_ENCODING=DEFAULT
QUANTIZATION_ALWAYS_RAM=True
```

Two-bit encoding:

```bash
QUANTIZATION_TYPE=BINARY
QUANTIZATION_BINARY_ENCODING=TWO_BITS
```

One-and-a-half-bit encoding:

```bash
QUANTIZATION_TYPE=BINARY
QUANTIZATION_BINARY_ENCODING=ONE_AND_HALF_BITS
```

Available encodings:

| Value | Bits per dimension | Approximate compression |
| --- | ---: | ---: |
| `DEFAULT` | 1 | 32x |
| `ONE_AND_HALF_BITS` | 1.5 | 24x |
| `TWO_BITS` | 2 | 16x |

Binary quantization is most promising for high-dimensional, centered vector
distributions. Qdrant recommends testing it with rescoring because rescoring
can substantially improve recall. The repository currently configures
collection quantization, but does not expose query-time quantization,
rescoring, or oversampling parameters through the schema.

## Product Quantization

Product quantization divides vectors into chunks and represents each chunk with
a learned centroid. It can minimize memory use, but usually has a larger
accuracy and search-speed tradeoff than scalar quantization.

```bash
QUANTIZATION_TYPE=PRODUCT
QUANTIZATION_PRODUCT_COMPRESSION=X16
QUANTIZATION_ALWAYS_RAM=False
```

Available compression values:

```text
X4 X8 X16 X32 X64
```

Product quantization is suitable when memory reduction is the primary goal and
lower recall or slower distance calculations are acceptable.

## TurboQuant

TurboQuant applies a randomized rotation before compression so that it works
well across a broader range of vector distributions. It is available in Qdrant
1.18 and newer.

```bash
QUANTIZATION_TYPE=TURBO
QUANTIZATION_TURBO_BITS=BITS4
QUANTIZATION_ALWAYS_RAM=True
```

Available bit depths:

| Value | Bits per dimension | Approximate compression |
| --- | ---: | ---: |
| `BITS4` | 4 | 8x |
| `BITS2` | 2 | 16x |
| `BITS1_5` | 1.5 | 24x |
| `BITS1` | 1 | 32x |

`BITS4` is the default and prioritizes recall. Lower bit depths increase
compression while generally increasing approximation error.

TurboQuant is a strong candidate when:

- more than 4x compression is needed
- binary quantization loses too much recall on the dataset
- the Qdrant runtime and Python client support Qdrant 1.18 features

## Suggested Experiment Order

For a new dataset, compare each configuration against `NONE` using the same
corpus, index settings, query set, and `TOP_K`:

1. Run `NONE` to establish latency, memory, and recall baselines.
2. Run `SCALAR` as the conservative compressed baseline.
3. Run `TURBO` with `BITS4`, then lower bit depths if more compression is
   needed.
4. Run `BINARY` when maximum speed or compression is important.
5. Run `PRODUCT` when minimizing memory is more important than latency or
   recall.

Measure at least:

- recall against exact or known ground truth
- query latency and throughput
- peak and steady-state memory
- index construction time
- collection storage size

## Using the Unified Submit Manager

Quantization values can be placed in an env config or passed with `--set`:

```bash
./pbs_submit_manager.sh \
  --engine qdrant \
  --config qdrant/sampleConfigs/insertion_testing_pes2o.env \
  --set QUANTIZATION_TYPE=SCALAR \
  --set QUANTIZATION_ALWAYS_RAM=True
```

Schema values also support sweeps. For example:

```bash
--set "QUANTIZATION_TYPE=NONE SCALAR BINARY"
```

Method-specific settings remain in the generated `run_config.env`, but only
the settings for the selected `QUANTIZATION_TYPE` are used when creating the
collection.

## Current Repository Scope

The current schema controls collection creation only. It does not yet expose:

- query-time quantization bypass
- rescoring
- oversampling
- binary asymmetric query encoding
- changing quantization on an existing collection

For the complete upstream behavior and compatibility details, see the
[Qdrant quantization documentation](https://qdrant.tech/documentation/manage-data/quantization/).
