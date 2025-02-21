## NTT in Rust

Small Rust repo to showcase different ways of computing Rust, including

- naive base implementation using Montgomery transformation, implemented for primes <= 61 bit
- AVX2-enhanced implementation using Montgomery transformation, implemented for primes <= 61 bit using Karatsuba algorithm
- specialized implementation for NTT-friendly prime (Goldilock prime)

## Build and run

```sh
RUSTFLAGS='-C target-cpu=native' cargo run -r
```

Benchmark results on Intel i7-8565U

```
Running through 30 forward and 30 backward NTTs
base 237ms
AVX2 160ms
Goldilock 286ms
```


## Test suite

```sh
RUSTFLAGS='-C target-cpu=native' cargo test -r
```
