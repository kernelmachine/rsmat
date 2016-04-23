# rsmat

matrix computations in rust

focus on benchmarking & parallelism

Fastest dot product so far:

  cargo rustc --release --bench bench -- -C target_cpu=corei7-avx

```rust
test bench_dot                 ... bench:     349,022 ns/iter (+/- 17,193)
test bench_dot_rayon           ... bench:     219,668 ns/iter (+/- 33,696)
test bench_dot_simple_parallel ... bench:     258,757 ns/iter (+/- 87,884)
```
