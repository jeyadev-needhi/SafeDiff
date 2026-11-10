The Fast-dLLM decoding source and LLaDA model/configuration source derive from
NVlabs/Fast-dLLM (Apache License 2.0); original copyright headers and LICENSE
are retained. The pinned upstream commit is
`a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`.

`fastdllm.py` is upstream `v1/llada/generate.py`, with only the model import
changed to a relative package import. Decoder function bodies are unchanged.

`instant.py` is Han's contiguous K-block generator from the
2026-10-28 threshold-parallel-sweep60 experiment. The instant preset uses
K=8, threshold=0.9, BERM mu=16, and at most 64 forwards per group including
warmup.

Both accelerated modes share the model implementation from that experiment. Its
changes relative to the pinned upstream model thread optional `position_ids`
and `q_position_ids` through attention/RoPE for explicit cached positions.
Fast-dLLM uses the ordinary position path.

Original and packaged hashes are recorded in `provenance.json`. Historical
experiment paths identify source revisions. Runtime code is bundled here;
checkpoint weights are supplied separately.
