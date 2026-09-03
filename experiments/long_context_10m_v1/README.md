# B-role pilot configuration: `long_context_10m_v1`

This directory freezes the 10M-token protocol for the Longformer and Memformer
workstream. The current execution target is **pilot/screening only**: each
candidate receives 2,031,616 training tokens with seed 17. The 10,027,008-token
three-seed runs are configured but are not started by the pilot command.

The primary dataset remains PG-19. If PG-19 cannot be downloaded or tokenized,
the runner must stop or explicitly write a `synthetic_copy_stream_fallback`
result; it must not label synthetic data as PG-19.

## Frozen B-role candidates

- Longformer: left windows 128/512/1024/2048, represented by odd total
  windows 257/1025/2049/4097, no global tokens.
- Memformer: (segment length, memory slots) = (256,64), (512,32),
  (512,64), (512,128).

## Status

Configuration is frozen. Data preparation and pilot execution are pending
environment/dependency verification on the detected RTX 4070 Laptop GPU (8GB).
