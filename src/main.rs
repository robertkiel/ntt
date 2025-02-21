use ntt::dft::{ntt::Table, ntt_avx2::TableAVX2, ntt_goldilock::TableGoldilock, DFT};
use rand::Rng;
use std::time::SystemTime;

fn main() {
    let mut rng = rand::rng();
    let table = Table::<u64>::new();
    let table_avx = TableAVX2::new();
    let table_goldilock = TableGoldilock::new();

    let mut a_avx2 = [0i64; 2u64.pow(16) as usize];

    for a_j in a_avx2.iter_mut() {
        *a_j = rng.random_range(0..table_avx.q);
    }
    let mut a = a_avx2.iter().map(|x| *x as u64).collect::<Vec<u64>>();

    let mut a_goldilock = [0u64; 2u64.pow(16) as usize];

    for a_j in a_goldilock.iter_mut() {
        *a_j = rng.random_range(0..table_goldilock.q);
    }

    println!("Running through 30 forward and 30 backward NTTs");

    let before_base = SystemTime::now();

    for _ in 0..30 {
        table.forward_inplace(&mut a);
        table.backward_inplace(&mut a);
    }

    println!("base {}ms", before_base.elapsed().unwrap().as_millis());

    let before_avx = SystemTime::now();

    for _ in 0..30 {
        table_avx.forward_inplace(&mut a_avx2);
        table_avx.backward_inplace(&mut a_avx2);
    }

    println!("AVX2 {}ms", before_avx.elapsed().unwrap().as_millis());

    let before_goldilock = SystemTime::now();

    for _ in 0..30 {
        table_goldilock.forward_inplace(&mut a_goldilock);
        table_goldilock.backward_inplace(&mut a_goldilock);
    }

    println!(
        "Goldilock {}ms",
        before_goldilock.elapsed().unwrap().as_millis()
    );
}
