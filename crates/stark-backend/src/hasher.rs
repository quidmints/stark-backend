use std::{fmt::Debug, marker::PhantomData};

use derivative::Derivative;
use p3_field::Field;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
#[cfg(feature = "multi-field-transcript")]
use {
    super::multi_field_packing::{checked_num_packed_f_elms, pack_f_to_sf},
    itertools::Itertools,
    p3_field::{PrimeField, PrimeField32},
    p3_symmetric::CryptographicPermutation,
};

/// Trait abstracting Merkle tree hash operations. This trait is used as part of the definition of
/// the [`StarkProtocolConfig`](crate::StarkProtocolConfig).
///
/// The `MerkleHasher` is parameterized over the base field `F` used as leaves,
/// and provides a `Digest` type for internal nodes.

/// [sbf-meter] CU accounting for OpenVM's hashing, mirroring the instrument already used on the
/// risc0 side so the two verifiers can be compared on one axis.
///
/// ⭐ THE NUMBER THIS EXISTS FOR: OpenVM verifies a real proof in 316,632,277 CU at poseidon2, and
/// RISC Zero in 406,353,121 — so OpenVM's verifier is the leaner of the two. But risc0 drops to
/// 12,834,936 by putting blake3 on the `sol_blake3` syscall, a 31.7x gain, and whether OpenVM would
/// gain the SAME factor depends entirely on WHAT FRACTION OF ITS BILL IS HASHING. At 92% blake3
/// leaves it at ~27M and it loses; at 97% it lands at ~11.7M and it wins. Nobody has measured it,
/// and the prover work needed to give OpenVM blake3 is only worth doing on the winning side of that
/// line.
///
/// Three words at a fixed runtime-heap address, because the SBF loader rejects writable statics:
/// [hash_cu, hash_calls, compress_calls].
#[cfg(target_os = "solana")]
pub mod meter {
    extern "C" {
        fn sol_remaining_compute_units() -> u64;
    }
    const BASE: *mut u64 = 0x300004000 as *mut u64;
    #[inline(always)]
    pub fn now() -> u64 { unsafe { sol_remaining_compute_units() } }
    #[inline(always)]
    pub fn account(start: u64, is_compress: bool) {
        unsafe {
            *BASE += start.saturating_sub(now());
            *BASE.add(1) += 1;
            if is_compress { *BASE.add(2) += 1; }
        }
    }
    /// Phase spans start at word 4: [shape, zerocheck_logup, gkr, stacked, whir], CU each.
    #[inline(always)]
    pub fn phase(slot: usize, start: u64) {
        unsafe { *BASE.add(4 + slot) += start.saturating_sub(now()); }
    }
    pub fn read() -> [u64; 12] { unsafe { core::array::from_fn(|i| *BASE.add(i)) } }
    pub fn reset() { unsafe { for i in 0..12 { *BASE.add(i) = 0; } } }
}
#[cfg(not(target_os = "solana"))]
pub mod meter {
    #[inline(always)] pub fn now() -> u64 { 0 }
    #[inline(always)] pub fn account(_s: u64, _c: bool) {}
    #[inline(always)] pub fn phase(_slot: usize, _start: u64) {}
    pub fn read() -> [u64; 12] { [0; 12] }
    pub fn reset() {}
}

pub trait MerkleHasher: 'static + Clone + Send + Sync {
    type F: Field;
    type Digest: 'static + Copy + Send + Sync;

    /// Hash a slice of field elements into a digest (used for leaf hashing).
    fn hash_slice(&self, vals: &[Self::F]) -> Self::Digest;

    /// Compress two digests into one (used for internal Merkle tree nodes).
    fn compress(&self, left: Self::Digest, right: Self::Digest) -> Self::Digest;

    /// Compress a vector of digests (must be power-of-two length) into a single digest
    /// by repeated binary compression.
    fn tree_compress(&self, mut hashes: Vec<Self::Digest>) -> Self::Digest {
        debug_assert!(hashes.len().is_power_of_two());
        while hashes.len() > 1 {
            let mut next = Vec::with_capacity(hashes.len() / 2);
            for pair in hashes.chunks_exact(2) {
                next.push(self.compress(pair[0], pair[1]));
            }
            hashes = next;
        }
        hashes.pop().unwrap()
    }
}

/// [MerkleHasher] implementation built from a cryptographic hash function and a compression
/// function. This struct is intended for use by the protocol verifier. Prover backends may use
/// independent but functionally equivalent implementations.
///
/// The compression function does not need to be collision-resistant and only needs to be a pseudo-cryptographic function: <https://eprint.iacr.org/2026/089>.
#[derive(Derivative, derive_new::new)]
#[derivative(
    Clone(bound = "H: Clone, C: Clone"),
    Debug(bound = "H: Debug, C: Debug")
)]
pub struct Hasher<F, Digest, H, C> {
    hash: H,
    compress: C,
    _phantom: PhantomData<(F, Digest)>,
}

impl<F, Digest, H, C> MerkleHasher for Hasher<F, Digest, H, C>
where
    F: Field,
    Digest: 'static + Copy + Send + Sync,
    H: 'static + Send + Sync + CryptographicHasher<F, Digest>,
    C: 'static + Send + Sync + PseudoCompressionFunction<Digest, 2>,
{
    type F = F;
    type Digest = Digest;

    fn hash_slice(&self, vals: &[Self::F]) -> Self::Digest {
        let __t = meter::now();
        let r = self.hash.hash_slice(vals);
        meter::account(__t, false);
        r
    }

    fn compress(&self, left: Self::Digest, right: Self::Digest) -> Self::Digest {
        let __t = meter::now();
        let r = self.compress.compress([left, right]);
        meter::account(__t, true);
        r
    }
}

/// A padding-free, overwrite-mode sponge that operates natively over `PF` but accepts elements
/// of `F: PrimeField32`, packing them at base-2^(F::bits()).
///
/// This differs from Plonky3's `MultiField32PaddingFreeSponge` which packs at base-2^32.
/// For BabyBear (31-bit field), base-2^31 packing is injective into BN254, whereas
/// base-2^32 packing of 8 elements can exceed the BN254 modulus.
#[cfg(feature = "multi-field-transcript")]
#[derive(Clone, Debug)]
pub struct MultiFieldHasher<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize> {
    permutation: P,
    num_f_elms: usize,
    _phantom: PhantomData<(F, PF)>,
}

#[cfg(feature = "multi-field-transcript")]
impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    MultiFieldHasher<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField,
{
    pub fn new(permutation: P) -> Self {
        let num_f_elms = checked_num_packed_f_elms::<F, PF>();

        Self {
            permutation,
            num_f_elms,
            _phantom: PhantomData,
        }
    }
}

#[cfg(feature = "multi-field-transcript")]
impl<F, PF, P, const WIDTH: usize, const RATE: usize, const OUT: usize>
    CryptographicHasher<F, [PF; OUT]> for MultiFieldHasher<F, PF, P, WIDTH, RATE, OUT>
where
    F: PrimeField32,
    PF: PrimeField + Default + Copy,
    P: CryptographicPermutation<[PF; WIDTH]>,
{
    fn hash_iter<I>(&self, input: I) -> [PF; OUT]
    where
        I: IntoIterator<Item = F>,
    {
        let mut state = [PF::default(); WIDTH];
        for block_chunk in &input.into_iter().chunks(RATE) {
            for (chunk_id, chunk) in (&block_chunk.chunks(self.num_f_elms))
                .into_iter()
                .enumerate()
            {
                state[chunk_id] = pack_f_to_sf(&chunk.collect_vec());
            }
            state = self.permutation.permute(state);
        }

        state[..OUT].try_into().unwrap()
    }
}
