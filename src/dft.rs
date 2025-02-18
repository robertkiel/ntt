pub mod ntt;
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
pub mod ntt_avx2;
pub mod ntt_goldilock;

pub trait DFT<O> {
    fn forward_inplace(&self, x: &mut [O]);
    fn forward_inplace_lazy(&self, x: &mut [O]);
    fn backward_inplace(&self, x: &mut [O]);
    fn backward_inplace_lazy(&self, x: &mut [O]);
}

#[cfg(test)]
mod tests {
    use crate::dft::ntt::Table;
    use crate::dft::ntt_avx2::TableAVX2;
    use crate::dft::ntt_goldilock::TableGoldilock;
    use crate::dft::DFT;
    use rand::Rng;
    use std::time::SystemTime;

    #[test]
    fn base_version() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new_goldilock();

        let mut a = [0u64; 2u64.pow(16) as usize];

        for a_j in a.iter_mut() {
            *a_j = rng.random_range(0..table.q);
        }

        let a_cloned = a.clone();
        table.forward_inplace(&mut a);
        assert!(a
            .iter()
            .zip(a_cloned.iter())
            .any(|(a, a_cloned)| a != a_cloned));
        table.backward_inplace(&mut a);

        assert_eq!(a, a_cloned);
    }

    #[test]
    fn avx_correctness() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new_u32_compatible();
        let table_avx = TableAVX2::new();

        let mut a_avx2 = [0i64; 2u64.pow(16) as usize];

        for a_j in a_avx2.iter_mut() {
            *a_j = rng.random_range(0..table_avx.q);
        }
        let mut a = a_avx2.iter().map(|x| *x as u64).collect::<Vec<u64>>();
        let a_cloned = a.clone();

        table.forward_inplace(&mut a);
        table_avx.forward_inplace(&mut a_avx2);

        assert!(a
            .iter()
            .zip(a_avx2.iter())
            .all(|(a, a_avx2)| *a == *a_avx2 as u64));

        table.backward_inplace(&mut a);
        table_avx.backward_inplace(&mut a_avx2);

        assert!(a
            .iter()
            .zip(a_avx2.iter())
            .all(|(a, a_avx2)| *a == *a_avx2 as u64));

        assert!(a
            .iter()
            .zip(a_cloned.iter())
            .all(|(a, a_cloned)| { *a == *a_cloned }));
    }

    #[test]
    fn avx_against_base() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new_u32_compatible();
        let table_avx = TableAVX2::new();

        let mut a_avx2 = [0i64; 2u64.pow(16) as usize];

        for a_j in a_avx2.iter_mut() {
            *a_j = rng.random_range(0..table_avx.q);
        }

        let mut a = a_avx2.iter().map(|x| *x as u64).collect::<Vec<u64>>();

        // check if backward(forward(a)) == a
        let a_cloned = a.clone();

        // assert_eq!(a, a_avx2);

        let now = SystemTime::now();
        for _ in 0..10 {
            table.forward_inplace(&mut a);

            table.backward_inplace(&mut a);
        }
        println!("base {}ms", now.elapsed().unwrap().as_millis());

        assert!(a
            .iter()
            .zip(a_cloned.iter())
            .all(|(a, a_cloned)| { *a == *a_cloned }));

        let now = SystemTime::now();

        for _ in 0..10 {
            table_avx.forward_inplace(&mut a_avx2);

            table_avx.backward_inplace(&mut a_avx2);
        }
        println!("AVX2 {}ms", now.elapsed().unwrap().as_millis());

        assert!(a_cloned
            .iter()
            .zip(a_avx2.iter())
            .all(|(a, a_avx2)| { *a == *a_avx2 as u64 }));

        panic!()
    }

    #[test]
    fn goldilock_correctness() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new_goldilock();
        let table_goldilock = TableGoldilock::new();

        let mut a_goldilock = [0u64; 2u64.pow(16) as usize];

        for a_j in a_goldilock.iter_mut() {
            *a_j = rng.random_range(0..table_goldilock.q);
        }

        let a_cloned = a_goldilock.clone();

        let mut a = a_goldilock.clone();

        table.forward_inplace(&mut a);
        table_goldilock.forward_inplace(&mut a_goldilock);

        assert_eq!(a, a_goldilock);

        table.backward_inplace(&mut a);
        table_goldilock.backward_inplace(&mut a_goldilock);

        assert_eq!(a, a_goldilock);
        assert_eq!(a, a_cloned);
    }

    #[test]
    fn goldilock_against_base() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new_goldilock();
        let table_goldilock = TableGoldilock::<u64>::new();

        let mut a_goldilock = [0u64; 2u64.pow(16) as usize];

        for a_j in a_goldilock.iter_mut() {
            *a_j = rng.random_range(0..table_goldilock.q);
        }

        let mut a = a_goldilock.clone();

        // check if backward(forward(a)) == a
        let a_cloned = a.clone();

        let now = SystemTime::now();
        for _ in 0..10 {
            table.forward_inplace(&mut a);

            table.backward_inplace(&mut a);
        }
        println!("base {}ms", now.elapsed().unwrap().as_millis());

        let now = SystemTime::now();

        for _ in 0..10 {
            table_goldilock.forward_inplace(&mut a_goldilock);

            table_goldilock.backward_inplace(&mut a_goldilock);
        }
        println!("goldilock {}ms", now.elapsed().unwrap().as_millis());

        assert_eq!(a_cloned, a);
    }
}
