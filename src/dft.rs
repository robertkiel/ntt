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

    #[test]
    fn avx_against_naive() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new_u32_compatible();
        let table_avx = TableAVX2::new();

        let mut a_avx2 = [0u32; 2u64.pow(16) as usize];

        for a_j in a_avx2.iter_mut() {
            *a_j = rng.random_range(0..table_avx.q);
        }

        let mut a = a_avx2.iter().map(|x| *x as u64).collect::<Vec<u64>>();

        // check if backward(forward(a)) == a
        let a_cloned = a.clone();

        table.forward_inplace(&mut a);
        table_avx.forward_inplace(&mut a_avx2);

        assert!(a
            .iter()
            .zip(a_avx2.iter())
            .all(|(a, a_avx2)| { *a as u32 == *a_avx2 }));

        table.backward_inplace(&mut a);
        table_avx.backward_inplace(&mut a_avx2);

        assert!(a
            .iter()
            .zip(a_avx2.iter())
            .all(|(a, a_avx2)| { *a as u32 == *a_avx2 }));

        assert_eq!(a_cloned, a);
    }

    #[test]
    fn goldilock_against_naive() {
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
}
