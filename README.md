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


## Unit test suite

```sh
RUSTFLAGS='-C target-cpu=native' cargo test -r
```

## Testing with other primes (and roots)

Note that the AVX2 implementation is heavily optimized for primes close to 2^64

### Base
```rust
use ntt::dft::{ntt::Table, DFT};

const PRIME: u64 = 0x1fffffffffe00001u64;
/// 2^17th root of unity
const ROOT: u64 = 0x15eb043c7aa2b01fu64;
const N: u64 = 2u64.pow(16);

Table::from_prime_and_root(PRIME, ROOT, N);
```

### AVX2

```rust
use ntt::dft::{ntt_avx2::TableAVX2, DFT};

const PRIME: i64 = 0x1fffffffffe00001;
/// 2^17th root of unity
const ROOT: i64 = 0x15eb043c7aa2b01f;
const N: i64 = 2i64.pow(16);

TableAVX2::from_prime_and_root(PRIME, ROOT, N);
```
