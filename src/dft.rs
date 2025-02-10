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
    use crate::dft::ntt_avx2::TableAVX;
    use crate::dft::DFT;
    use rand::Rng;

    #[test]
    fn test_float_case() {
        let mut rng = rand::rng();
        let table = Table::<u64>::new_u32_compatible();
        let table_avx = TableAVX::new();

        let mut a_float = [0u32; 2u64.pow(16) as usize];

        for i in 0..a_float.len() {
            a_float[i] = rng.random_range(0..table_avx.q);
        }

        let mut a = a_float
            .iter()
            .cloned()
            .map(|x| x as u64)
            .collect::<Vec<u64>>();

        let a_clone = a.clone();
        table.forward_inplace(&mut a);
        table_avx.forward_inplace(&mut a_float);

        println!("a {:?}", a.iter().copied().take(4).collect::<Vec<u64>>());
        println!(
            "a_float {:?}",
            a_float.iter().copied().take(4).collect::<Vec<u32>>()
        );

        println!("Going backwards");

        table.backward_inplace(&mut a);
        table_avx.backward_inplace(&mut a_float);

        println!("a {:?}", a.iter().copied().take(4).collect::<Vec<u64>>());
        println!(
            "a_float {:?}",
            a_float.iter().copied().take(4).collect::<Vec<u32>>()
        );
    }
}
