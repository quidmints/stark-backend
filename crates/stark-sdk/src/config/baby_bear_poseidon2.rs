use std::{
    io::{self, Read, Write},
    marker::PhantomData,
    sync::OnceLock,
};

use openvm_stark_backend::{
    codec::{
        decode_extension_field32, decode_prime_field32, encode_extension_field32,
        encode_prime_field32, DecodableConfig, EncodableConfig,
    },
    hasher::Hasher,
    p3_symmetric::{self, PaddingFreeSponge, Permutation, TruncatedPermutation},
    prover::{Coordinator, CpuColMajorBackend, ReferenceDevice},
    transcript::duplex_sponge,
    FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams, TranscriptLog,
};
use p3_baby_bear::{default_babybear_poseidon2_16, BabyBear, Poseidon2BabyBear};
use p3_field::{extension::BinomialExtensionField, integers::QuotientMap, PrimeCharacteristicRing, PrimeField32};

const RATE: usize = 8;
/// permutation width
const WIDTH: usize = 16; // rate + capacity
pub const CHUNK: usize = 8;
pub const DIGEST_SIZE: usize = CHUNK;

type Perm = Poseidon2BabyBear<WIDTH>;
// Generic over P: CryptographicPermutation<[F; WIDTH]>
type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_SIZE>;
type Compress<P> = TruncatedPermutation<P, 2, CHUNK, WIDTH>;
type PermHasher<P> = Hasher<F, Digest, Hash<P>, Compress<P>>;
/// With `blake3-hash` on, the SAME config hashes with blake3 instead of poseidon2.
#[cfg(feature = "blake3-hash")]
type Blake3Hasher = Hasher<F, Digest, Blake3F, Blake3F>;

// ⭐⭐ A REAL BLAKE3 HASHER FOR OPENVM, KEEPING `Digest = [BabyBear; 8]`.
//
// The owner has asked for OpenVM's blake3 number repeatedly and I kept explaining why a
// `BabyBearBlake3Config` is a large campaign — new engine, transcript, generic vk/proof types, a
// prover. All true, and all AVOIDABLE for the purpose of getting the NUMBER: the digest type does
// not have to change. blake3 emits 32 bytes, which is exactly 8 u32 words, which map onto 8
// BabyBear elements. So the hash BODY can be swapped underneath the existing config and every
// concrete type downstream — `VmStarkVerifyingKey`, the engines, the codecs — keeps compiling.
//
// 🔑 AND BECAUSE PROVER AND VERIFIER SHARE THIS CONFIG, PROOFS ACTUALLY VERIFY. This is not a
// timing harness that fakes a hash and measures an early rejection; it is the same protocol with a
// different hash, end to end.
//
// ⚠️ THE ONE PLACE IT DOES NOT REACH, STATED: OpenVM's aggregation proves "I verified the previous
// layer" IN-CIRCUIT against poseidon2 chips, so a blake3 FRI commitment cannot be checked by that
// circuit. ⇒ this measures an APP proof, not the aggregated internal-recursive one. The ratio it
// yields is what the risc0-vs-OpenVM comparison actually needs.
//
// ⚠️ `from_wrapped_u32` REDUCES rather than rejecting, so every 4-byte word is a valid element.
// That costs under one bit per limb against blake3's 256 — ~248 bits survive, far above the ~100
// the protocol targets.
#[cfg(feature = "blake3-hash")]
#[derive(Clone, Copy, Debug)]
pub struct Blake3F;

#[cfg(feature = "blake3-hash")]
#[inline]
fn blake3_bytes(data: &[u8]) -> [u8; 32] {
    #[cfg(target_os = "solana")]
    {
        extern "C" {
            fn sol_blake3(vals: *const u8, val_len: u64, hash_result: *mut u8) -> u64;
        }
        let parts: [&[u8]; 1] = [data];
        let mut out = [0u8; 32];
        unsafe { sol_blake3(parts.as_ptr() as *const u8, 1, out.as_mut_ptr()) };
        out
    }
    #[cfg(not(target_os = "solana"))]
    {
        *blake3::hash(data).as_bytes()
    }
}

#[cfg(feature = "blake3-hash")]
#[inline]
fn bytes_to_digest(b: [u8; 32]) -> Digest {
    // BabyBear's modulus. Reducing explicitly (rather than reaching for a `from_wrapped_*` helper
    // that this Plonky3 version does not expose) keeps the mapping obvious and total: every 4-byte
    // word becomes a valid canonical element.
    const P: u32 = 0x78000001;
    core::array::from_fn(|i| {
        let w = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        F::from_int(w % P)
    })
}

#[cfg(feature = "blake3-hash")]
impl p3_symmetric::CryptographicHasher<F, Digest> for Blake3F {
    fn hash_iter<I>(&self, input: I) -> Digest
    where
        I: IntoIterator<Item = F>,
    {
        let mut buf = Vec::new();
        for f in input {
            buf.extend_from_slice(&f.as_canonical_u32().to_le_bytes());
        }
        bytes_to_digest(blake3_bytes(&buf))
    }
}

#[cfg(feature = "blake3-hash")]
impl p3_symmetric::PseudoCompressionFunction<Digest, 2> for Blake3F {
    fn compress(&self, input: [Digest; 2]) -> Digest {
        let mut buf = [0u8; 64];
        for (k, d) in input.iter().enumerate() {
            for (i, f) in d.iter().enumerate() {
                buf[k * 32 + i * 4..k * 32 + i * 4 + 4]
                    .copy_from_slice(&f.as_canonical_u32().to_le_bytes());
            }
        }
        bytes_to_digest(blake3_bytes(&buf))
    }
}

// Defined below
type SC = BabyBearPoseidon2Config;

// Convenience type aliases
pub type F = BabyBear;
pub type EF = BinomialExtensionField<BabyBear, 4>;
pub const D_EF: usize = 4;
pub type Digest = [F; DIGEST_SIZE];
pub type DuplexSponge = duplex_sponge::DuplexSponge<F, Perm, WIDTH, RATE>;
pub type DuplexSpongeRecorder = duplex_sponge::DuplexSpongeRecorder<F, Perm, WIDTH, RATE>;
pub type DuplexSpongeValidator = duplex_sponge::DuplexSpongeValidator<F, Perm, WIDTH, RATE>;

/// ⭐ ONE ALIAS DECIDES THE WHOLE CONFIG'S HASH. Everything downstream — the engines, the codecs,
/// `VmStarkVerifyingKey` — is written against `Self::Hasher`, so swapping this swaps the protocol's
/// hash without touching a single concrete type.
#[cfg(not(feature = "blake3-hash"))]
pub type ConfigHasher = PermHasher<Perm>;
#[cfg(feature = "blake3-hash")]
pub type ConfigHasher = Blake3Hasher;

#[derive(Clone, Debug, derive_new::new)]
pub struct BabyBearPoseidon2Config {
    params: SystemParams,
    hasher: ConfigHasher,
}

impl StarkProtocolConfig for BabyBearPoseidon2Config {
    type F = F;
    type EF = EF;
    type Digest = Digest;
    type Hasher = ConfigHasher;

    fn params(&self) -> &SystemParams {
        &self.params
    }

    fn hasher(&self) -> &Self::Hasher {
        &self.hasher
    }
}

impl BabyBearPoseidon2Config {
    pub fn new_from_perm(params: SystemParams, perm: Perm) -> Self {
        #[cfg(not(feature = "blake3-hash"))]
        let hasher = Hasher::new(
            PaddingFreeSponge::new(perm.clone()),
            TruncatedPermutation::new(perm),
        );
        // The permutation is still threaded through for the Fiat-Shamir transcript, which stays
        // a DuplexSponge over poseidon2 — only the MERKLE hash moves to blake3, and that is where
        // the thousands of calls are.
        #[cfg(feature = "blake3-hash")]
        let hasher = {
            let _ = perm;
            Hasher::new(Blake3F, Blake3F)
        };
        Self { params, hasher }
    }

    pub fn default_from_params(params: SystemParams) -> Self {
        let perm = default_babybear_poseidon2_16();
        Self::new_from_perm(params, perm)
    }
}

impl EncodableConfig for BabyBearPoseidon2Config {
    fn encode_base_field<W: Write>(val: &F, writer: &mut W) -> io::Result<()> {
        encode_prime_field32(val, writer)
    }

    fn encode_extension_field<W: Write>(val: &EF, writer: &mut W) -> io::Result<()> {
        encode_extension_field32::<Self::F, _, _>(val, writer)
    }

    fn encode_digest<W: Write>(digest: &Self::Digest, writer: &mut W) -> io::Result<()> {
        for val in digest {
            encode_prime_field32(val, writer)?;
        }
        Ok(())
    }
}

impl DecodableConfig for BabyBearPoseidon2Config {
    fn decode_base_field<R: Read>(reader: &mut R) -> io::Result<F> {
        decode_prime_field32(reader)
    }

    fn decode_extension_field<R: Read>(reader: &mut R) -> io::Result<EF> {
        decode_extension_field32::<F, _, _>(reader)
    }

    fn decode_digest<R: Read>(reader: &mut R) -> io::Result<Digest> {
        let mut result = Digest::default();
        for val in &mut result {
            *val = decode_prime_field32(reader)?;
        }
        Ok(result)
    }
}

pub struct BabyBearPoseidon2RefEngine<TS = DuplexSponge> {
    device: ReferenceDevice<SC>,
    _transcript: PhantomData<TS>,
}

impl<TS> StarkEngine for BabyBearPoseidon2RefEngine<TS>
where
    TS: FiatShamirTranscript<SC> + From<Perm>,
{
    type SC = SC;
    type PB = CpuColMajorBackend<SC>;
    type PD = ReferenceDevice<SC>;
    type TS = TS;

    fn new(params: SystemParams) -> Self {
        let config = BabyBearPoseidon2Config::default_from_params(params);
        Self {
            device: ReferenceDevice::new(config),
            _transcript: PhantomData,
        }
    }

    fn config(&self) -> &SC {
        self.device.config()
    }

    fn device(&self) -> &Self::PD {
        &self.device
    }

    fn initial_transcript(&self) -> Self::TS {
        TS::from(default_babybear_poseidon2_16())
    }

    fn prover_from_transcript(
        &self,
        transcript: TS,
    ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
        Coordinator::new(CpuColMajorBackend::new(), self.device.clone(), transcript)
    }
}

// ---- Optimized CPU engine (behind `cpu-backend` feature) ----

#[cfg(feature = "cpu-backend")]
mod cpu_engine {
    use openvm_cpu_backend::{CpuBackend, CpuDevice};

    use super::*;

    /// SIMD-optimized transcript backed by Plonky3's `DuplexChallenger`.
    ///
    /// Produces identical Fiat-Shamir challenges as the standard `DuplexSponge` transcript,
    /// but with ~4x faster proof-of-work grinding on aarch64 NEON (8x on x86 AVX2).
    #[derive(Clone, Debug)]
    pub struct CpuTranscript {
        inner: openvm_stark_backend::p3_challenger::DuplexChallenger<
            BabyBear,
            Poseidon2BabyBear<WIDTH>,
            WIDTH,
            RATE,
        >,
    }

    impl From<Perm> for CpuTranscript {
        fn from(perm: Perm) -> Self {
            Self {
                inner: openvm_stark_backend::p3_challenger::DuplexChallenger::new(perm),
            }
        }
    }

    impl FiatShamirTranscript<BabyBearPoseidon2Config> for CpuTranscript {
        #[inline]
        fn observe(&mut self, value: BabyBear) {
            openvm_stark_backend::p3_challenger::CanObserve::observe(&mut self.inner, value);
        }

        #[inline]
        fn sample(&mut self) -> BabyBear {
            openvm_stark_backend::p3_challenger::CanSample::sample(&mut self.inner)
        }

        fn observe_commit(&mut self, digest: [BabyBear; RATE]) {
            for x in digest {
                openvm_stark_backend::p3_challenger::CanObserve::observe(&mut self.inner, x);
            }
        }

        fn grind(&mut self, bits: usize) -> BabyBear {
            openvm_stark_backend::p3_challenger::GrindingChallenger::grind(&mut self.inner, bits)
        }
    }

    /// Row-major CPU engine for BabyBear + Poseidon2.
    ///
    /// Default transcript is [`CpuTranscript`], which uses Plonky3's `DuplexChallenger`
    /// for SIMD-optimized proof-of-work grinding (~4x faster on NEON, ~8x on AVX2).
    pub struct BabyBearPoseidon2CpuEngine<TS = CpuTranscript> {
        device: CpuDevice<SC>,
        _transcript: PhantomData<TS>,
    }

    impl<TS> StarkEngine for BabyBearPoseidon2CpuEngine<TS>
    where
        TS: FiatShamirTranscript<SC> + From<Poseidon2BabyBear<WIDTH>>,
    {
        type SC = SC;
        type PB = CpuBackend<SC>;
        type PD = CpuDevice<SC>;
        type TS = TS;

        fn new(params: SystemParams) -> Self {
            let config = BabyBearPoseidon2Config::default_from_params(params);
            Self {
                device: CpuDevice::new(config),
                _transcript: PhantomData,
            }
        }

        fn config(&self) -> &SC {
            self.device.config()
        }

        fn device(&self) -> &Self::PD {
            &self.device
        }

        fn initial_transcript(&self) -> Self::TS {
            TS::from(default_babybear_poseidon2_16())
        }

        fn prover_from_transcript(
            &self,
            transcript: TS,
        ) -> Coordinator<Self::SC, Self::PB, Self::PD, Self::TS> {
            Coordinator::new(CpuBackend::new(), self.device.clone(), transcript)
        }
    }
}

#[cfg(feature = "cpu-backend")]
pub use cpu_engine::{BabyBearPoseidon2CpuEngine, CpuTranscript};

// Fixed Poseidon2 configuration
// [sbf] WAS `static PERM: OnceLock<..>`, CACHED ONCE AND RETURNED BY REFERENCE. A `OnceLock`
// is writable by construction — it has to be, to record that it was initialised — so it lands
// in `.bss` and the Solana loader refuses the whole ELF: "read-write data not supported".
//
// ⚠️ THE TRADE IS REAL AND IS NOT FREE: the permutation is now BUILT ON EVERY CALL rather than
// once. Three of the four callers already `.clone()`d it, so they pay nothing; the fourth used
// it through a temporary and still can. What it costs on chain is unmeasured, and it is the
// first thing to measure if this route is ever taken seriously — a verifier that rebuilds its
// permutation per hash would be a bad trade even with a loading ELF.
pub fn poseidon2_perm() -> Poseidon2BabyBear<WIDTH> {
    default_babybear_poseidon2_16()
}

pub fn poseidon2_compress_with_capacity(
    left: [F; CHUNK],
    right: [F; CHUNK],
) -> ([F; CHUNK], [F; CHUNK]) {
    let mut state = [F::ZERO; WIDTH];
    state[..CHUNK].copy_from_slice(&left);
    state[CHUNK..].copy_from_slice(&right);
    poseidon2_perm().permute_mut(&mut state);
    (
        state[..CHUNK].try_into().unwrap(),
        state[CHUNK..].try_into().unwrap(),
    )
}

pub fn default_duplex_sponge() -> DuplexSponge {
    DuplexSponge::from(poseidon2_perm())
}

pub fn default_duplex_sponge_recorder() -> DuplexSpongeRecorder {
    DuplexSpongeRecorder::from(poseidon2_perm())
}

pub fn default_duplex_sponge_validator(
    logs: TranscriptLog<F, [F; WIDTH]>,
) -> DuplexSpongeValidator {
    DuplexSpongeValidator::new(poseidon2_perm(), logs)
}

#[cfg(test)]
mod poseidon2_constant_tests {
    use p3_baby_bear::{
        BABYBEAR_RC16_EXTERNAL_FINAL, BABYBEAR_RC16_EXTERNAL_INITIAL, BABYBEAR_RC16_INTERNAL,
    };
    use zkhash::{
        ark_ff::PrimeField as _, fields::babybear::FpBabyBear as HorizenBabyBear,
        poseidon2::poseidon2_instance_babybear::RC16,
    };

    use super::*;

    fn horizen_to_p3(horizen_babybear: HorizenBabyBear) -> BabyBear {
        BabyBear::from_u64(horizen_babybear.into_bigint().0[0])
    }

    #[allow(clippy::type_complexity)]
    pub fn horizen_round_consts_16() -> ((Vec<[BabyBear; 16]>, Vec<[BabyBear; 16]>), Vec<BabyBear>)
    {
        let p3_rc16: Vec<Vec<BabyBear>> = RC16
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|babybear| horizen_to_p3(*babybear))
                    .collect()
            })
            .collect();

        let rounds_f = 8;
        let rounds_p = 13;
        let rounds_f_beginning = rounds_f / 2;
        let p_end = rounds_f_beginning + rounds_p;
        let initial: Vec<[BabyBear; 16]> = p3_rc16[..rounds_f_beginning]
            .iter()
            .cloned()
            .map(|round| round.try_into().unwrap())
            .collect();
        let terminal: Vec<[BabyBear; 16]> = p3_rc16[p_end..]
            .iter()
            .cloned()
            .map(|round| round.try_into().unwrap())
            .collect();
        let internal_round_constants: Vec<BabyBear> = p3_rc16[rounds_f_beginning..p_end]
            .iter()
            .map(|round| round[0])
            .collect();
        ((initial, terminal), internal_round_constants)
    }

    /// Uses HorizenLabs Poseidon2 round constants, but plonky3 Mat4 and also
    /// with a p3 Monty reduction factor.
    #[test]
    fn test_horizen_p3_rc_equality() {
        let ((external_initial, external_terminal), internal_constants) = horizen_round_consts_16();
        assert_eq!(external_initial, BABYBEAR_RC16_EXTERNAL_INITIAL.to_vec());
        assert_eq!(external_terminal, BABYBEAR_RC16_EXTERNAL_FINAL.to_vec());
        assert_eq!(internal_constants, BABYBEAR_RC16_INTERNAL.to_vec());
    }
}
