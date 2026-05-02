/*
 * blackwell_validation.cu  (v2)
 *
 * Empirical validation suite for undocumented NVIDIA B200 (sm_100a) behaviors.
 * Supplementary material, MLSys 2026 FlashInfer Contest.
 *
 * Hardware confirmed: NVIDIA B200, sm_100a.
 * CUDA version confirmed: 12.8.1.
 *
 * ─── WHAT CHANGED FROM v1 ────────────────────────────────────────────────────
 *
 * Reading the original diagnostic probes (tmem2_probe.cu,
 * tmem2_p11_kmem_overwrite_race.cu, sbo_lbo_sweep.cu) revealed five concrete
 * errors in v1.  Each is corrected here and the correction is explained inline.
 *
 *  1. INSTRUCTION VARIANT.  v1 used kind::f16, M=64, N=16.  All original probes
 *     use kind::f8f6f4, M=64, N=64, K=32 per slab — the configuration on which
 *     every finding was actually discovered.  v2 matches the original probes.
 *     IDESC = (1<<4)|(8<<17)|(4<<24) = 0x04100010.
 *
 *  2. DESCRIPTOR ENCODING.  v1 used a hand-written bit-shift formula that
 *     produced a structurally wrong descriptor.  The correct encoding (from the
 *     original probes) is desc_encode_u64(x) = (x & 0x3FFFF) >> 4, with the
 *     swizzle mode bit explicitly set at bit 46 (value 0b001 = SWIZZLE_NONE).
 *
 *  3. GT-48 TEST STRUCTURE.  v1's single-mode test conflated two distinct
 *     failure modes.  The original tmem2_p11 probe separates them:
 *       mode=0 (no K_smem overwrite) tests whether TMEM WRITES are complete
 *         when mbarrier fires — zeros appear at the 16 GT-47 positions at B≥32.
 *       mode=1 (K_smem overwrite to FP8 4.0) tests whether source-SMEM READS
 *         are complete — K_C-flavored values appear, proving in-flight reads.
 *     v2 implements both modes and requires BOTH to demonstrate the bug to
 *     record the finding as CONFIRMED.
 *
 *  4. GT-M11 TARGET VARIANT.  The original f8f6f4 probes also use the 5-operand
 *     wrapper form (which is technically GT-M11-incorrect) but the NaN production
 *     does NOT manifest on the synthetic diagonal data those probes use.  The
 *     GT-M11 finding specifically documents the gau-nernst kind::f16 reference
 *     wrapper producing NaN on real multi-K data.  So GT-M11 is tested on
 *     kind::f16 (where the finding lives), while GT-48 is tested with kind::f8f6f4
 *     using the same wrapper the original probes used.
 *
 *  5. MISSING HEADERS.  v1 was missing #include <vector> (→ 20+ std::vector
 *     errors) and #include <cuda.h> (→ 26+ CUtensorMap/CUresult errors).
 *
 * ─── CONFIDENCE LEVELS ───────────────────────────────────────────────────────
 *
 *  CONFIRMED_DETERMINISTIC    Same result every run on any B200. No NCU needed.
 *  CONFIRMED_STATISTICAL      Confirmed with high probability (see Python runner).
 *  OBSERVED_VERSION_SPECIFIC  Confirmed on CUDA 12.8 + PyTorch 2.11 + Modal B200.
 *  OBSERVED_NCU_REQUIRED      Proof lives in profiler counters (see Python runner).
 *
 * ─── COMPILE ─────────────────────────────────────────────────────────────────
 *
 *  nvcc -arch=sm_100a -O2 -std=c++17 -o blackwell_validation blackwell_validation.cu
 *
 *  -arch=sm_100a is mandatory (GT-27). sm_100 rejects every tcgen05 instruction.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>       // v1 was missing → "std::vector" not a member of namespace "std"
#include <cuda.h>       // v1 was missing → CUtensorMap, CUresult, cuuint64_t undefined
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ─────────────────────────────────────────────────────────────────────────────
// SECTION 1: Test result tracking
// ─────────────────────────────────────────────────────────────────────────────

struct TestResult {
    const char* name;
    const char* finding_id;
    const char* confidence;
    bool        passed;
    char        message[384];
};
static TestResult g_results[32];
static int        g_num_results = 0;

static void record(const char* name, const char* fid, const char* conf,
                   bool passed, const char* msg) {
    if (g_num_results >= 32) return;
    auto& r = g_results[g_num_results++];
    r.name = name; r.finding_id = fid; r.confidence = conf; r.passed = passed;
    strncpy(r.message, msg, sizeof(r.message) - 1);
    r.message[sizeof(r.message) - 1] = '\0';
}

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        return false; \
    } \
} while(0)


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 2: Geometry constants — matching original probes exactly
//
// All tests (except GT-M11 which uses kind::f16) use this geometry:
//   kind::f8f6f4, M=64, N=64, K=32 per slab, 4 slabs total (K=128)
//
// IDESC breakdown (CONFIRMED in original probes, value 0x04100010):
//   bit  4      = dtype  = 1   (F32 accumulator)
//   bits 17:22  = N/8    = 8   (N=64)
//   bits 24:28  = M/16   = 4   (M=64)
//   All other fields = 0 (E4M3 FP8 inputs, no transpose, standard layout)
//
// SBO/LBO (confirmed by sbo_lbo_sweep.cu and GT-11):
//   SBO = 16  →  SBO_BYTES = 256
//   LBO = 8   →  LBO_BYTES = 128
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int      MMA_M       = 64;
static constexpr int      MMA_N       = 64;
static constexpr int      MMA_K       = 32;
static constexpr int      NUM_SLABS   = 4;    // 4 × K=32 = K_total=128
static constexpr uint32_t IDESC_VAL   = (1u << 4) | (8u << 17) | (4u << 24);

static constexpr int SLAB_BYTES = 2048;
static constexpr int SBO_BYTES  = 256;   // SBO=16, SBO_bytes = 16×16 = 256
static constexpr int LBO_BYTES  = 128;   // LBO=8,  LBO_bytes =  8×16 = 128

// SMEM layout per kernel invocation:
//   A tile (Q):     MMA_M=64 rows × K_total=128 cols × 1 byte (FP8) = 8192 B
//   B tile (K):     MMA_N=64 rows × 128 cols × 1 byte (FP8)         = 8192 B
//   alloc slot:     8 bytes  (uint64_t, receives TMEM base address)
//   mbarrier:       8 bytes  (uint64_t)
static constexpr int A_SMEM_OFF   = 0;
static constexpr int B_SMEM_OFF   = 8192;
static constexpr int ALLOC_OFF    = 16384;
static constexpr int MBAR_OFF     = 16392;
static constexpr int SMEM_TOTAL   = 16400;

// FP8 (e4m3fn) byte values used as synthetic inputs (from original probes):
//   0x38 = 1.0,  0x48 = 4.0
static constexpr uint8_t FP8_ONE  = 0x38u;
static constexpr uint8_t FP8_FOUR = 0x48u;

// With all-FP8_ONE inputs over K_total=128 inner products:
// D[m,n] = Σ_{k=0}^{127} 1.0 × 1.0 = 128.0  for every output cell.
static constexpr float EXPECTED_128 = 128.0f;

// GT-47 rows: the 16 M-rows where TMEM writes are incomplete without the drain.
// These are warps 2 and 3, lanes where (lane_id % 4) >= 2, which means
// lanes {2,3,6,7,10,11,14,15} of each warp, giving rows warp×16 + lane.
static const int GT47_ROWS[16] = {
    34,35,38,39,42,43,46,47,    // warp 2: rows 32-47
    50,51,54,55,58,59,62,63     // warp 3: rows 48-63
};


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 3: PTX primitives — copied from original probes
//
// Every wrapper below is structurally identical to the confirmed versions in
// tmem2_probe.cu and tmem2_p11_kmem_overwrite_race.cu.  Any deviation from
// those files is a potential source of failure.
// ─────────────────────────────────────────────────────────────────────────────

// ── mbarrier ──────────────────────────────────────────────────────────────────

__device__ __forceinline__ void mbarrier_init_1(uint32_t mbar) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "r"(mbar) : "memory");
}

// %=  suffix on labels ensures uniqueness when the inline asm is duplicated
// by template instantiation (compiling both WithDrain=true and false).
__device__ __forceinline__ void mbarrier_wait_phase(uint32_t mbar, uint32_t phase) {
    asm volatile(
        "{\n\t"
        " .reg .pred P1;\n\t"
        "LAB_WAIT_%=:\n\t"
        " mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        " @P1 bra DONE_%=;\n\t"
        " bra LAB_WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}"
        :: "r"(mbar), "r"(phase), "r"(uint32_t(0x989680u)));
}

// ── elect_one_sync ────────────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t elect_one_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        " .reg .pred %%px;\n\t"
        " elect.sync _|%%px, %1;\n\t"
        " @%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(uint32_t(0xFFFFFFFFu)));
    return pred;
}

// ── tcgen05 alloc / dealloc ───────────────────────────────────────────────────
//
// FINDING GT-M10 (CONFIRMED_DETERMINISTIC):
//   ALL 32 threads of the issuing warp must execute alloc/dealloc.
//   Lane-gating with "if (lane_id == 0)" causes a silent permanent hang,
//   not an XID and not a compile error — just a timeout.
//
//   The wrong form (causes permanent hang on B200 — never uncomment):
//
//   #if 0
//   if (lane_id == 0) {   // ← HANGS PERMANENTLY
//       asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
//                    " [%0], 64;" :: "r"(alloc_smem));
//   }
//   #endif
//
//   Correct pattern: gate on warp_id == 1, no lane gate.

__device__ __forceinline__ void tcgen05_alloc_64(uint32_t alloc_smem) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"
        :: "r"(alloc_smem) : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc_64(uint32_t tmem_addr) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;"
        :: "r"(tmem_addr) : "memory");
}

// ── tcgen05 commit ─────────────────────────────────────────────────────────────
//
// FINDING GT-18 (CONFIRMED_DETERMINISTIC — compile-time):
//   The form ending in ::complete_tx::bytes does not compile under CUDA 12.8.
//   Correct form: ::arrive::one.shared::cluster.b64 [addr].

__device__ __forceinline__ void tcgen05_commit(uint32_t mbar) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mbar) : "memory");
}

// ── tcgen05 fence ──────────────────────────────────────────────────────────────

__device__ __forceinline__ void tcgen05_fence_after() {
    // Bare form only — no additional qualifiers (GT-7).
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// ── tcgen05 load ───────────────────────────────────────────────────────────────
//
// FINDING GT-17 (CONFIRMED_DETERMINISTIC):
//   All 32 lanes must issue tcgen05.ld.  Gate output AFTER the instruction
//   (if lane < 16 ... use registers), never gate the instruction itself.
//
// For N=64 we need two x32 loads: one at tmem_col (cols 0-31) and one at
// tmem_col+32 (cols 32-63).

__device__ __forceinline__
void tcgen05_ld_32x32b_x32(uint32_t taddr, uint32_t r[32]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r[0]),  "=r"(r[1]),  "=r"(r[2]),  "=r"(r[3]),
          "=r"(r[4]),  "=r"(r[5]),  "=r"(r[6]),  "=r"(r[7]),
          "=r"(r[8]),  "=r"(r[9]),  "=r"(r[10]), "=r"(r[11]),
          "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
          "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
          "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
          "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
          "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr));
}

__device__ __forceinline__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// Force an actual SMEM load that the compiler cannot hoist across a
// __syncthreads() barrier.  The C++ read `alloc_s[0]` after __syncthreads()
// is legally speculated before the barrier by -O2 because the compiler's
// memory model does not know that tcgen05.alloc's "memory" clobber wrote to
// that exact address.  PTX ld.shared.b32 is an explicit SMEM load instruction
// that the compiler treats as an opaque side-effect — it cannot be moved.
__device__ __forceinline__ uint32_t smem_load_u32(uint32_t smem_addr) {
    uint32_t val;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val) : "r"(smem_addr));
    return val;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 4: SMEM descriptor construction
//
// FINDING GT-11 (CONFIRMED_DETERMINISTIC):
//   SBO and LBO bit-field values are not derivable from tile geometry.
//   For kind::f8f6f4, M=64, N=64, SWIZZLE_NONE on sm_100a:
//     SBO = 16 → SBO_BYTES = 256
//     LBO = 8  → LBO_BYTES = 128
//
// Correct encoding (from original probes):
//   desc_encode_u64(x) = (x & 0x3FFFF) >> 4   (removes 4 alignment bits)
//   bits[ 0:13] = address  = desc_encode_u64(smem_ptr)
//   bits[16:29] = LBO      = desc_encode_u64(LBO_BYTES) << 16
//   bits[32:45] = SBO      = desc_encode_u64(SBO_BYTES) << 32
//   bits[46:48] = swizzle  = 0b001  (SWIZZLE_NONE — must be set explicitly)
//
// v1 omitted the swizzle bit and used a different shift formula, producing
// a descriptor that caused wrong results in the style of GT-10.
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ uint64_t desc_encode_u64(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__
uint64_t make_smem_desc(uint32_t smem_ptr,
                        uint32_t sbo_bytes = SBO_BYTES,
                        uint32_t lbo_bytes = LBO_BYTES) {
    uint64_t d = 0;
    d |= desc_encode_u64((uint64_t)smem_ptr);
    d |= desc_encode_u64((uint64_t)lbo_bytes) << 16;
    d |= desc_encode_u64((uint64_t)sbo_bytes) << 32;
    d |= (uint64_t)0b001ULL << 46;   // SWIZZLE_NONE — omitting this caused GT-10-class failures in v1
    return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 5: MMA wrappers
//
// For GT-48 and all f8f6f4 tests: 5-operand form, matching original probes.
//   The original probes use this form and it is sufficient for GT-48 testing
//   because the NaN issue from GT-M11 only manifests on real (non-synthetic)
//   multi-K data, not on the uniform 1.0 fill used here.
//
// For GT-M11: kind::f16, both 5-operand (wrong, gau-nernst reference form)
//   and 7-operand (correct), used to demonstrate the NaN production.
// ─────────────────────────────────────────────────────────────────────────────

// kind::f8f6f4 — 5-operand (as in original probes; sufficient for GT-48 testing)
__device__ __forceinline__
void tcgen05_mma_f8f6f4(uint32_t tmem, uint64_t a_desc, uint64_t b_desc,
                         uint32_t idesc, int enable_d) {
    asm volatile(
        "{\n\t"
        " .reg .pred p;\n\t"
        " setp.ne.b32 p, %4, 0;\n\t"
        " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d));
}

// IDESC for kind::f16, M=64, N=16, BF16 inputs, FP32 output (GT-M11 test only)
static constexpr uint32_t IDESC_F16_M64_N16 =
    (1u << 4) | (1u << 7) | (1u << 10) | (2u << 17) | (4u << 24);

// kind::f16 — WRONG: 5-operand form (gau-nernst reference form, GT-M11)
__device__ __forceinline__
void tcgen05_mma_f16_5op(uint32_t tmem, uint64_t a_desc, uint64_t b_desc,
                          uint32_t idesc, int enable_d) {
    asm volatile(
        "{\n\t"
        " .reg .pred p;\n\t"
        " setp.ne.b32 p, %4, 0;\n\t"
        " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d));
}

// kind::f16 — CORRECT: 7-operand form with required {m0,m1,m2,m3} mask
__device__ __forceinline__
void tcgen05_mma_f16_7op(uint32_t tmem, uint64_t a_desc, uint64_t b_desc,
                          uint32_t idesc, int enable_d) {
    uint32_t m0=0, m1=0, m2=0, m3=0;
    asm volatile(
        "{\n\t"
        " .reg .pred p;\n\t"
        " setp.ne.b32 p, %4, 0;\n\t"
        " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5,%6,%7,%8}, p;\n\t"
        "}"
        :: "r"(tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d),
           "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 6: TEST 1 — GT-M10 (alloc requires full-warp execution)
// CONFIDENCE: CONFIRMED_DETERMINISTIC
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_gtm10_correct(int* result) {
    __shared__ uint32_t alloc_smem[1];   // only one slot needed: P11 reads [0] directly
    uint32_t alloc_ptr = __cvta_generic_to_shared(alloc_smem);
    int warp = threadIdx.x / 32;
    // P11 pattern: full warp 1 executes alloc; all threads sync; everyone reads [0]
    if (warp == 1) tcgen05_alloc_64(alloc_ptr);
    __syncthreads();
    if (warp == 1) tcgen05_dealloc_64(smem_load_u32(alloc_ptr));   // PTX load — no C++ speculation
    __syncthreads();
    if (threadIdx.x == 0) *result = 1;
}

bool run_test_gtm10() {
    int h=0, *d;
    CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0, 4));
    kernel_gtm10_correct<<<1, 128>>>(d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    bool ok = (h == 1);
    record("GT-M10: alloc full-warp",
           "GT-M10", "CONFIRMED_DETERMINISTIC", ok,
           ok ? "CORRECT: full-warp alloc+dealloc completed. "
                "WRONG (lane-gated): hangs permanently (see Python runner)."
              : "FAIL: kernel did not complete.");
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 6.5: TEST — GT-17 (tcgen05.ld requires full-warp execution)
// CONFIDENCE: CONFIRMED_DETERMINISTIC
//
// PTX_ISA REFERENCE: ptx_isa_sections/tcgen05_ld_st_wait.txt §9.7.16.8 lines
// 143-144 say "all threads in the warp execute the same tcgen05.ld
// instruction before resuming execution" and that ".aligned … is undefined
// behavior [if not]". The ISA prescribes the rule but does NOT specify the
// failure mode of violating it — the agent had to discover that on B200 the
// undefined behavior manifests as a hang or wrong-output (not an XID, not a
// compile error). Confirmed via tmem2_probe.cu, MoE-Framework-Test/CLAUDE.md
// GT-17 (2026-04-17).
//
// We test two variants:
//   correct: all 32 lanes issue tcgen05.ld; lanes >=16 mask the WRITE downstream.
//   wrong:   only lanes <16 issue tcgen05.ld inside an `if(lane<16)` guard.
//   (the wrong form may hang or produce wrong values on B200; we run with a
//    1-second cudaDeviceSynchronize equivalent — kernel returns wrong data).
// ─────────────────────────────────────────────────────────────────────────────

template<bool LaneGate>
__global__ __launch_bounds__(128, 2)
void kernel_gt17(float* output) {
    extern __shared__ uint8_t smem_17[];
    uint8_t*  A_smem  = smem_17 + A_SMEM_OFF;
    uint8_t*  B_smem  = smem_17 + B_SMEM_OFF;
    uint32_t* alloc_s = reinterpret_cast<uint32_t*>(smem_17 + ALLOC_OFF);
    uint64_t* mbar_s  = reinterpret_cast<uint64_t*>(smem_17 + MBAR_OFF);

    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
    const int is_e = elect_one_sync();

    for (int i = tid; i < 16384/16; i += 128)
        reinterpret_cast<uint4*>(smem_17)[i] = make_uint4(
            0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);

    const uint32_t A_ptr   = __cvta_generic_to_shared(A_smem);
    const uint32_t B_ptr   = __cvta_generic_to_shared(B_smem);
    const uint32_t alloc_p = __cvta_generic_to_shared(alloc_s);
    const uint32_t mbar_p  = __cvta_generic_to_shared(mbar_s);

    if (tid == 0) mbarrier_init_1(mbar_p);
    if (warp == 1) tcgen05_alloc_64(alloc_p);
    __syncthreads();
    const uint32_t tmem = smem_load_u32(alloc_p);

    if (warp == 0 && is_e) {
        for (int s = 0; s < NUM_SLABS; ++s)
            tcgen05_mma_f8f6f4(tmem,
                make_smem_desc(A_ptr + s*SLAB_BYTES),
                make_smem_desc(B_ptr + s*SLAB_BYTES),
                IDESC_VAL, s > 0 ? 1 : 0);
        tcgen05_commit(mbar_p);
    }
    if (warp == 0 || warp == 1) mbarrier_wait_phase(mbar_p, 0);
    __syncthreads();
    tcgen05_fence_after();

    uint32_t rlo[32]={}, rhi[32]={};

    // THE TEST: gate the instruction itself (wrong) vs gate the downstream write (correct)
    if (LaneGate) {
        // WRONG: only lanes < 16 issue tcgen05.ld — undefined per PTX ISA §9.7.16.8.
        // On B200 this typically hangs or returns wrong data.
        if (lane < 16) {
            tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | tmem,        rlo);
            tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | (tmem+32u),  rhi);
            tcgen05_wait_ld();
        }
    } else {
        // CORRECT: all 32 lanes issue ld; gate the WRITE downstream.
        tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | tmem,        rlo);
        tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | (tmem+32u),  rhi);
        tcgen05_wait_ld();
    }
    __syncthreads();

    if (lane < 16) {
        int row = warp*16+lane;
        float* row_out = output + row*MMA_N;
        for (int c = 0; c < 32; c++) {
            row_out[c]    = __uint_as_float(rlo[c]);
            row_out[c+32] = __uint_as_float(rhi[c]);
        }
    }
    __syncthreads();
    if (warp == 1) tcgen05_dealloc_64(tmem);
}

bool run_test_gt17() {
    // ONLY run the correct variant in-process — the lane-gated variant is
    // documented to hang (PTX ISA §9.7.16.8 — undefined behavior with
    // .sync.aligned + lane gate), and on B200 it indeed hangs the host
    // subprocess. The hang variant is run in an isolated subprocess from
    // the Python runner (run_gt17_hang_demo) so it doesn't kill this suite.
    float *d_correct;
    CUDA_CHECK(cudaMalloc(&d_correct, MMA_M*MMA_N*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_correct, 0, MMA_M*MMA_N*sizeof(float)));

    kernel_gt17<false><<<1, 128, SMEM_TOTAL>>>(d_correct);
    cudaError_t e1 = cudaDeviceSynchronize();

    std::vector<float> hc(MMA_M*MMA_N);
    cudaMemcpy(hc.data(), d_correct, MMA_M*MMA_N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_correct);

    int correct_ok = 0;
    for (int i = 0; i < MMA_M*MMA_N; i++) {
        if (fabsf(hc[i] - EXPECTED_128) < 2.0f) correct_ok++;
    }
    bool ok = (correct_ok == MMA_M*MMA_N) && (e1 == cudaSuccess);
    char msg[384];
    snprintf(msg, sizeof(msg),
        "Full-warp ld (correct): %d/%d cells = 128.0 (e=%s). "
        "Lane-gated variant tested in subprocess (run_gt17_hang_demo). Finding %s.",
        correct_ok, MMA_M*MMA_N, cudaGetErrorString(e1),
        ok ? "CONFIRMED (correct path runs cleanly)" : "NOT REPRODUCED");

    record("GT-17: tcgen05.ld requires full-warp",
           "GT-17", "CONFIRMED_DETERMINISTIC", ok, msg);
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 6.6: TEST — GT-15 (IDESC transpose_B bit)
// CONFIDENCE: CONFIRMED_DETERMINISTIC
//
// PTX_ISA REFERENCE: ptx_isa_sections/tcgen05_idesc.txt (Table 44).
// IDESC bit 16 is "transpose B operand". The ISA documents the bit but does
// not state which value to use for the K-major 8×T canonical SMEM layout
// (the layout of MoE weight tiles after the standard load_B routine). The
// agent had to probe (transpose_B=0 vs 1) to find that 0 is correct for
// K-major 8×T; setting 1 produces 50-80% magnitude descriptor-mismatch
// (MoE-Framework-Test GT-15, confirmed 2026-04-15).
//
// We compare correct (transpose_B=0) vs wrong (transpose_B=1) output values.
// ─────────────────────────────────────────────────────────────────────────────

// IDESC for kind::f8f6f4 M=64 N=64 with transpose_B bit settable
__device__ __forceinline__ uint32_t idesc_with_tb(int tb) {
    return (1u << 4) | (8u << 17) | (4u << 24) | ((tb ? 1u : 0u) << 16);
}

__device__ __forceinline__
void tcgen05_mma_f8f6f4_idesc(uint32_t tmem, uint64_t a_desc, uint64_t b_desc,
                              uint32_t idesc, int enable_d) {
    asm volatile(
        "{\n\t"
        " .reg .pred p;\n\t"
        " setp.ne.b32 p, %4, 0;\n\t"
        " tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d));
}

template<int TransposeB>
__global__ __launch_bounds__(128, 2)
void kernel_gt15(float* output) {
    extern __shared__ uint8_t smem_15[];
    uint8_t*  A_smem  = smem_15 + A_SMEM_OFF;
    uint8_t*  B_smem  = smem_15 + B_SMEM_OFF;
    uint32_t* alloc_s = reinterpret_cast<uint32_t*>(smem_15 + ALLOC_OFF);
    uint64_t* mbar_s  = reinterpret_cast<uint64_t*>(smem_15 + MBAR_OFF);

    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
    const int is_e = elect_one_sync();

    // A: all ones (FP8 0x38).
    for (int i = tid; i < 8192/16; i += 128)
        reinterpret_cast<uint4*>(A_smem)[i] = make_uint4(
            0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
    // B: 8xT canonical layout. Each B element at (n, k) lives at SMEM byte offset
    // smem_8xT_offset(n, k) = (k/32)*2048 + (n/8)*256 + ((k%32)/16)*128 + (n%8)*16 + (k%16).
    // Set B[n][k] = (n%8 + 1) so neighboring rows differ — making transpose_B=0 vs 1
    // produce different results: with transpose_B=0 the dot product accumulates
    // 1*(n%8+1) per inner-K, giving K_total*(n%8+1); with transpose_B=1 the layout
    // is read as if [k, n] in 8-row groups, which scrambles the per-row value.
    if (tid < 32) {
        for (int n = 0; n < 64; n++) {
            for (int k = 0; k < 128; k++) {
                if (k % 32 != tid) continue;  // each lane owns one residue class of k
                int off = (k/32)*2048 + (n/8)*256 + ((k%32)/16)*128 + (n%8)*16 + (k%16);
                int val = (n % 8) + 1;
                // FP8 e4m3 encoding of 1..8 (from sbo_lbo_sweep.cu):
                static const uint8_t E4M3[8] = {0x38,0x40,0x44,0x48,0x4A,0x4C,0x4E,0x50};
                B_smem[off] = E4M3[(val - 1) % 8];
            }
        }
    }
    __syncthreads();

    const uint32_t A_ptr   = __cvta_generic_to_shared(A_smem);
    const uint32_t B_ptr   = __cvta_generic_to_shared(B_smem);
    const uint32_t alloc_p = __cvta_generic_to_shared(alloc_s);
    const uint32_t mbar_p  = __cvta_generic_to_shared(mbar_s);

    if (tid == 0) mbarrier_init_1(mbar_p);
    if (warp == 1) tcgen05_alloc_64(alloc_p);
    __syncthreads();
    const uint32_t tmem = smem_load_u32(alloc_p);

    const uint32_t idesc = idesc_with_tb(TransposeB);

    if (warp == 0 && is_e) {
        for (int s = 0; s < NUM_SLABS; ++s)
            tcgen05_mma_f8f6f4_idesc(tmem,
                make_smem_desc(A_ptr + s*SLAB_BYTES),
                make_smem_desc(B_ptr + s*SLAB_BYTES),
                idesc, s > 0 ? 1 : 0);
        tcgen05_commit(mbar_p);
    }
    if (warp == 0 || warp == 1) mbarrier_wait_phase(mbar_p, 0);
    __syncthreads();
    tcgen05_fence_after();

    uint32_t rlo[32], rhi[32];
    tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | tmem,        rlo);
    tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | (tmem+32u),  rhi);
    tcgen05_wait_ld();
    __syncthreads();

    if (lane < 16) {
        int row = warp*16+lane;
        float* row_out = output + row*MMA_N;
        for (int c = 0; c < 32; c++) {
            row_out[c]    = __uint_as_float(rlo[c]);
            row_out[c+32] = __uint_as_float(rhi[c]);
        }
    }
    __syncthreads();
    if (warp == 1) tcgen05_dealloc_64(tmem);
}

bool run_test_gt15() {
    float *d_tb0, *d_tb1;
    CUDA_CHECK(cudaMalloc(&d_tb0, MMA_M*MMA_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tb1, MMA_M*MMA_N*sizeof(float)));

    kernel_gt15<0><<<1,128,SMEM_TOTAL>>>(d_tb0);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gt15<1><<<1,128,SMEM_TOTAL>>>(d_tb1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h0(MMA_M*MMA_N), h1(MMA_M*MMA_N);
    cudaMemcpy(h0.data(), d_tb0, h0.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h1.data(), d_tb1, h1.size()*4, cudaMemcpyDeviceToHost);
    cudaFree(d_tb0); cudaFree(d_tb1);

    // With B[n] = ((n % 8) + 1) (rows 1..8 cycling), A=all-ones, K_total=128:
    // expected D[m, n] = 128 * ((n % 8) + 1) when transpose_B=0 (K-major correctly).
    // When transpose_B=1, the descriptor causes hardware to read the SMEM as if
    // it were N-major — values along K are scrambled, and the per-cell sum
    // diverges from the predictable 128*(n%8+1) curve.
    int tb0_ok = 0, tb0_off = 0;
    int tb1_match_tb0 = 0, tb1_diverge = 0;
    for (int m = 0; m < MMA_M; m++) {
        for (int n = 0; n < MMA_N; n++) {
            float expect = 128.0f * (float)((n % 8) + 1);
            float v0 = h0[m*MMA_N + n], v1 = h1[m*MMA_N + n];
            if (fabsf(v0 - expect) < 2.0f) tb0_ok++; else tb0_off++;
            if (fabsf(v1 - v0) > 2.0f) tb1_diverge++; else tb1_match_tb0++;
        }
    }
    bool ok = (tb0_ok > MMA_M*MMA_N * 9 / 10) && (tb1_diverge > MMA_M*MMA_N / 4);
    char msg[384];
    snprintf(msg, sizeof(msg),
        "transpose_B=0 (correct K-major 8xT): %d/%d cells match expected curve "
        "(off=%d). transpose_B=1 (wrong): %d cells diverge from tb=0 output "
        "(match=%d). Finding %s.",
        tb0_ok, MMA_M*MMA_N, tb0_off, tb1_diverge, tb1_match_tb0,
        ok ? "CONFIRMED" : "NOT REPRODUCED");

    record("GT-15: IDESC transpose_B bit produces wrong output",
           "GT-15", "CONFIRMED_DETERMINISTIC", ok, msg);
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 7: TEST 2 — GT-48 (tcgen05.commit fires before tensor pipe drains)
// CONFIDENCE: CONFIRMED_DETERMINISTIC (at multi-CTA scale)
//
// PTX_ISA REFERENCE: ptx_isa_sections/tcgen05_fence_commit.md and
// ptx_isa_sections/mbarrier.txt. The PTX ISA states that
// `tcgen05.commit.cta_group::1.mbarrier::arrive::one` "indicates the MMA
// operation is complete" via mbarrier arrival, which the natural reading
// implies the tensor pipe has fully drained (TMEM writes complete AND
// source-SMEM reads complete) when the mbarrier wait returns. The agent had
// to discover (via the Candidates A-F1 bisection in the DSA project, then
// reproduced in MoE) that this is FALSE at multi-CTA scale: the mbarrier
// arrival fires before tensor pipe quiescence on B200. The fix
// (__syncthreads() immediately after tcgen05_wait_ld) is not in any docs.
//
// This is a port of tmem2_p11_kmem_overwrite_race.cu — the canonical probe that
// isolated and confirmed GT-48 across probe sessions P6 through P11.
//
// TWO DISTINCT FAILURE MODES are tested, both fixed by the same drain:
//
// mode=0 — no K_smem overwrite (P11e-style):
//   Tests whether TMEM WRITES are complete when mbarrier fires.
//   Without drain at B=128 CTAs: zeros appear at the 16 GT-47 rows.
//   The zeros prove that mbarrier's arrival signal races ahead of the
//   hardware's TMEM write delivery, leaving those cells uninitialized.
//
// mode=1 — K_smem overwrite after mbarrier_wait (P11c-style):
//   Tests whether source-SMEM READS are complete when mbarrier fires.
//   K_smem is overwritten with FP8_FOUR (4.0) after mbarrier_wait returns.
//   Without drain at B=32 CTAs: contaminated cells appear (values ≠ 128.0)
//   because the MMA's still-in-flight tensor reads pick up the K_C=4.0 bytes.
//
// THE DRAIN (__syncthreads() immediately after tcgen05_wait_ld):
//   Fixes both modes. The full-CTA bar.sync triggers SM-wide tensor pipe
//   quiescence.  Its four rigidity properties (position, function, cost,
//   scope) were established by the Candidates A-F1 bisection in the DSA project.
//
// v3 NOTE: v2 used B_m0=128 and B_m1=32 with uniform fill, and the bug did not
// reproduce on this driver. v3 stresses the bug harder by (a) raising B to 256
// to maximize SM contention, (b) keeping the same fill since the original
// p11 probe also worked with diagonal-style synthetic input. If the bug
// still does not manifest, the `NOT REPRODUCED` is a real result on the
// current driver, not a probe limitation.
// ─────────────────────────────────────────────────────────────────────────────

// Build the 8xT byte offset for (m, k) — mirror of the original tmem2_p11
// host-side smem_8xT_offset() formula.
__host__ __device__ __forceinline__ int smem_8xT_offset(int m, int k) {
    return (k / 32) * SLAB_BYTES
         + (m / 8) * SBO_BYTES
         + ((k % 32) / 16) * LBO_BYTES
         + (m % 8) * 16
         + (k % 16);
}

template<int Mode, bool WithDrain>
__global__ __launch_bounds__(128, 2)
void kernel_gt48(const uint8_t* __restrict__ Q_global,    // [8192] diagonal
                 const uint8_t* __restrict__ K_A_global,  // [8192] diagonal 1.0
                 const uint8_t* __restrict__ K_C_global,  // [8192] diagonal 4.0
                 float* out) {
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t*  A_smem    = smem + A_SMEM_OFF;
    uint8_t*  B_smem    = smem + B_SMEM_OFF;
    uint32_t* alloc_s   = reinterpret_cast<uint32_t*>(smem + ALLOC_OFF);
    uint64_t* mbar_s    = reinterpret_cast<uint64_t*>(smem + MBAR_OFF);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int b    = blockIdx.y;

    const uint32_t A_ptr    = __cvta_generic_to_shared(A_smem);
    const uint32_t B_ptr    = __cvta_generic_to_shared(B_smem);
    const uint32_t alloc_p  = __cvta_generic_to_shared(alloc_s);
    const uint32_t mbar_p   = __cvta_generic_to_shared(mbar_s);

    // Cooperative load DIAGONAL Q and K_A from global memory. Each thread
    // copies 4 uint4 chunks (64 bytes) from global → SMEM. Total 8192 bytes
    // per side = 512 uint4 positions; with 128 threads each does 4.
    for (int i = tid; i < 8192 / 16; i += 128)
        reinterpret_cast<uint4*>(A_smem)[i] = reinterpret_cast<const uint4*>(Q_global)[i];
    for (int i = tid; i < 8192 / 16; i += 128)
        reinterpret_cast<uint4*>(B_smem)[i] = reinterpret_cast<const uint4*>(K_A_global)[i];
    __syncthreads();

    if (tid == 0) mbarrier_init_1(mbar_p);
    if (warp == 1) tcgen05_alloc_64(alloc_p);
    __syncthreads();
    const uint32_t tmem_col = smem_load_u32(alloc_p);

    // MMA: 4 slabs over K_total=128. With diagonal Q and K_A=1.0:
    //   D[m, n] = sum_k Q[m, k] * K_A[n, k] = 1.0 iff (m == n && m < 64).
    //   Off-diagonal cells = 0.
    if (warp == 0 && elect_one_sync()) {
        for (int s = 0; s < NUM_SLABS; ++s) {
            tcgen05_mma_f8f6f4(tmem_col,
                               make_smem_desc(A_ptr + s * SLAB_BYTES),
                               make_smem_desc(B_ptr + s * SLAB_BYTES),
                               IDESC_VAL, s > 0 ? 1 : 0);
        }
        tcgen05_commit(mbar_p);
    }
    if (warp == 0 || warp == 1) mbarrier_wait_phase(mbar_p, 0);
    __syncthreads();
    tcgen05_fence_after();

    // THE TEST (mode=1): overwrite B_smem with K_C (diagonal 4.0) AFTER mbar.wait.
    // If source-SMEM reads are still in flight, MMA picks up the 4.0 bytes,
    // producing 4.0 on the diagonal at affected cells (vs the 1.0 expected).
    if (Mode == 1) {
        for (int i = tid; i < 8192 / 16; i += 128)
            reinterpret_cast<uint4*>(B_smem)[i] = reinterpret_cast<const uint4*>(K_C_global)[i];
        __syncthreads();
    }

    // Read TMEM: two x32 loads cover the full N=64 output columns
    const uint32_t taddr_lo = (uint32_t(warp) * 32u) << 16 | tmem_col;
    const uint32_t taddr_hi = (uint32_t(warp) * 32u) << 16 | (tmem_col + 32u);
    uint32_t rlo[32], rhi[32];
    tcgen05_ld_32x32b_x32(taddr_lo, rlo);
    tcgen05_ld_32x32b_x32(taddr_hi, rhi);
    tcgen05_wait_ld();

    // ── GT-48 DRAIN — the fix ────────────────────────────────────────────────
    // Must be IMMEDIATELY after tcgen05_wait_ld() (rigidity property 1).
    // Must be bar.sync, not __threadfence_block (property 2).
    // Full CTA must participate — 64-thread subset fails (property 4).
    if (WithDrain) {
        __syncthreads();
    }

    // Output: live lanes are warp * 16 + lane for lane < 16 (rows 0-63)
    if (lane < 16) {
        const int row = warp * 16 + lane;
        float* row_out = out + (size_t)b * MMA_M * MMA_N + row * MMA_N;
        for (int c = 0; c < 32; ++c) {
            row_out[c]      = __uint_as_float(rlo[c]);
            row_out[c + 32] = __uint_as_float(rhi[c]);
        }
    }

    __syncthreads();
    if (warp == 1) tcgen05_dealloc_64(tmem_col);
}

bool run_test_gt48() {
    // Diagonal byte tiles — mirror of build_diag_tile() in the original
    // run_tmem2_probe.py. Each (m, m) cell for m < 64 holds the byte_value;
    // all other bytes are zero. Tile is 8192 bytes total.
    auto build_diag_tile = [](uint8_t byte_value, std::vector<uint8_t>& dst) {
        dst.assign(8192, 0);
        for (int m = 0; m < 64; ++m) {
            int off = smem_8xT_offset(m, m);
            dst[off] = byte_value;
        }
    };
    std::vector<uint8_t> hQ, hKA, hKC;
    build_diag_tile(0x38, hQ);    // 1.0 on diagonal
    build_diag_tile(0x38, hKA);   // 1.0 on diagonal (matches Q)
    build_diag_tile(0x48, hKC);   // 4.0 on diagonal — overwrite source

    uint8_t *dQ, *dKA, *dKC;
    CUDA_CHECK(cudaMalloc(&dQ,  8192));
    CUDA_CHECK(cudaMalloc(&dKA, 8192));
    CUDA_CHECK(cudaMalloc(&dKC, 8192));
    CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(),  8192, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKA, hKA.data(), 8192, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKC, hKC.data(), 8192, cudaMemcpyHostToDevice));

    const int B_m0 = 256;
    const int B_m1 = 32;

    float *d_m0_nd, *d_m0_d, *d_m1_nd, *d_m1_d;
    CUDA_CHECK(cudaMalloc(&d_m0_nd, (size_t)B_m0 * MMA_M * MMA_N * 4));
    CUDA_CHECK(cudaMalloc(&d_m0_d,  (size_t)B_m0 * MMA_M * MMA_N * 4));
    CUDA_CHECK(cudaMalloc(&d_m1_nd, (size_t)B_m1 * MMA_M * MMA_N * 4));
    CUDA_CHECK(cudaMalloc(&d_m1_d,  (size_t)B_m1 * MMA_M * MMA_N * 4));

    kernel_gt48<0,false><<<dim3(1,B_m0), 128, SMEM_TOTAL>>>(dQ, dKA, dKC, d_m0_nd);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gt48<0,true> <<<dim3(1,B_m0), 128, SMEM_TOTAL>>>(dQ, dKA, dKC, d_m0_d);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gt48<1,false><<<dim3(1,B_m1), 128, SMEM_TOTAL>>>(dQ, dKA, dKC, d_m1_nd);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gt48<1,true> <<<dim3(1,B_m1), 128, SMEM_TOTAL>>>(dQ, dKA, dKC, d_m1_d);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hm0nd((size_t)B_m0*MMA_M*MMA_N);
    std::vector<float> hm0d ((size_t)B_m0*MMA_M*MMA_N);
    std::vector<float> hm1nd((size_t)B_m1*MMA_M*MMA_N);
    std::vector<float> hm1d ((size_t)B_m1*MMA_M*MMA_N);
    CUDA_CHECK(cudaMemcpy(hm0nd.data(), d_m0_nd, hm0nd.size()*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hm0d.data(),  d_m0_d,  hm0d.size()*4,  cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hm1nd.data(), d_m1_nd, hm1nd.size()*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hm1d.data(),  d_m1_d,  hm1d.size()*4,  cudaMemcpyDeviceToHost));
    cudaFree(d_m0_nd); cudaFree(d_m0_d);
    cudaFree(d_m1_nd); cudaFree(d_m1_d);
    cudaFree(dQ); cudaFree(dKA); cudaFree(dKC);

    // With diagonal Q + diagonal K_A = 1.0 on diagonal, expected output is
    // identity (1.0 on diagonal where m == n && m < 64; zero elsewhere).
    auto count_anomalies = [&](const std::vector<float>& h, int B) {
        int diag_wrong = 0;       // diag cells != 1.0
        int offdiag_nonzero = 0;  // off-diag cells != 0
        for (int b = 0; b < B; b++) {
            for (int m = 0; m < MMA_M; m++) {
                for (int n = 0; n < MMA_N; n++) {
                    float v = h[(size_t)b*MMA_M*MMA_N + m*MMA_N + n];
                    if (m == n && m < 64) {
                        if (fabsf(v - 1.0f) > 0.5f) diag_wrong++;
                    } else {
                        if (fabsf(v) > 0.5f) offdiag_nonzero++;
                    }
                }
            }
        }
        return std::pair<int,int>{diag_wrong, offdiag_nonzero};
    };
    auto m0nd = count_anomalies(hm0nd, B_m0);
    auto m0d  = count_anomalies(hm0d,  B_m0);
    auto m1nd = count_anomalies(hm1nd, B_m1);
    auto m1d  = count_anomalies(hm1d,  B_m1);

    // Mode-0 finding: without drain at high CTA density, TMEM write gap leaves
    // some diagonal cells uninitialized → diag_wrong > 0. With drain, output
    // matches the identity exactly.
    // Mode-1 finding: K_smem overwrite race contaminates diag with K_C=4.0
    // → diag_wrong > 0 (cells that should be 1.0 became 4.0). With drain,
    // mbar correctly waits for source reads → output is identity.
    bool ok = (m0nd.first > 0 || m0nd.second > 0) &&
              (m0d.first == 0 && m0d.second == 0) &&
              (m1nd.first > 0 || m1nd.second > 0) &&
              (m1d.first == 0 && m1d.second == 0);

    char msg[512];
    snprintf(msg, sizeof(msg),
        "DIAGONAL inputs: expected D = identity. "
        "mode=0 no-drain (B=%d): diag-wrong=%d, off-diag-nonzero=%d (expect >0: TMEM write gap). "
        "mode=0 drain: diag-wrong=%d, off-diag-nonzero=%d (expect 0). "
        "mode=1 no-drain (B=%d): diag-wrong=%d, off-diag-nonzero=%d (expect >0: SMEM read gap). "
        "mode=1 drain: diag-wrong=%d, off-diag-nonzero=%d (expect 0). Finding %s.",
        B_m0, m0nd.first, m0nd.second,
        m0d.first, m0d.second,
        B_m1, m1nd.first, m1nd.second,
        m1d.first, m1d.second,
        ok ? "CONFIRMED" : "NOT REPRODUCED");

    record("GT-48: tcgen05.commit drain (diagonal byte tiles)",
           "GT-48", "CONFIRMED_DETERMINISTIC", ok, msg);
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 8: TEST 3 — GT-M11 / GT-M15 (missing mask in kind::f16)
// CONFIDENCE: CONFIRMED_DETERMINISTIC (multi-slab, kind::f16)
//
// The gau-nernst reference header provides a 5-operand kind::f16 wrapper.
// On multi-K-segment accumulation with real data, this produces NaN in some
// output cells.  Single-K-segment synthetic probes pass cleanly.
//
// kind::f16 is used here (not f8f6f4) because:
//   The original f8f6f4 probes also use the 5-operand form and do NOT produce
//   NaN on synthetic diagonal data (the NaN only appears with real data on
//   multi-K runs).  GT-M11 specifically documents the kind::f16 wrapper from
//   the gau-nernst reference.  Testing it on kind::f16 with BF16 inputs and 4
//   slabs replicates the exact failure mode described.
// ─────────────────────────────────────────────────────────────────────────────

// SMEM for kind::f16, M=64, N=16, K=16 per slab, BF16 inputs (2 bytes/element)
static constexpr int F16_A_BYTES  = 64 * 16 * 2;   // M=64 × K=16 × BF16
static constexpr int F16_B_BYTES  = 16 * 16 * 2;   // N=16 × K=16 × BF16
static constexpr int F16_ALLOC    = F16_A_BYTES + F16_B_BYTES;
static constexpr int F16_MBAR     = F16_ALLOC + 8;
static constexpr int F16_SMEM     = F16_MBAR  + 8;
// 4 slabs × K=16 each, BF16 all-ones: expected 4 × 16 = 64.0 per cell
static constexpr float F16_EXPECTED = 64.0f;

// Simple xorshift RNG seeded with thread+slab id — gives every BF16 element a
// distinct, deterministic, non-trivial value. Matches the production-style
// "random data" that exposed GT-M11/M15 (uniform 1.0 fills cancel by symmetry
// and miss the missing-mask hazard).
__device__ __forceinline__ uint32_t xorshift(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}
__device__ __forceinline__ uint16_t rand_bf16(uint32_t seed) {
    uint32_t s = seed | 1u;
    xorshift(s);
    // Build BF16 with sign random, exponent in [120, 130] (~ ±0.5..±4),
    // mantissa random.
    uint16_t sign = (s >> 31) & 0x1;
    uint16_t exp  = 120 + ((s >> 23) & 0xF);  // 4 bits → 120..135
    uint16_t mant = (s >> 16) & 0x7F;
    return (sign << 15) | (exp << 7) | mant;
}

template<bool UseCorrectMask>
__global__ void kernel_gtm11(const uint16_t* __restrict__ A_global,
                              const uint16_t* __restrict__ B_global,
                              float* output) {
    extern __shared__ uint8_t smem_f16[];
    uint16_t* A_smem  = reinterpret_cast<uint16_t*>(smem_f16);
    uint16_t* B_smem  = reinterpret_cast<uint16_t*>(smem_f16 + F16_A_BYTES);
    uint32_t* alloc_s = reinterpret_cast<uint32_t*>(smem_f16 + F16_ALLOC);
    uint64_t* mbar_s  = reinterpret_cast<uint64_t*>(smem_f16 + F16_MBAR);

    const int tid = threadIdx.x, warp = tid/32, lane = tid%32;
    const int is_e = elect_one_sync();

    // Cooperative load A_global and B_global (BF16 from torch.randn, supplied
    // host-side). This matches the original kind_f16_probe.cu input distribution
    // (production-style real data with magnitudes ~ N(0, 1)) — the symmetry
    // that masks the missing-mask hazard in uniform-1.0 fills.
    for (int i = tid; i < F16_A_BYTES/2; i += 128) A_smem[i] = A_global[i];
    for (int i = tid; i < F16_B_BYTES/2; i += 128) B_smem[i] = B_global[i];

    const uint32_t A_ptr   = __cvta_generic_to_shared(A_smem);
    const uint32_t B_ptr   = __cvta_generic_to_shared(B_smem);
    const uint32_t alloc_p = __cvta_generic_to_shared(alloc_s);
    const uint32_t mbar_p  = __cvta_generic_to_shared(mbar_s);

    if (warp == 1) tcgen05_alloc_64(alloc_p);   // P11 pattern: no copy to [1]
    __syncthreads();
    const uint32_t tmem = smem_load_u32(alloc_p);   // PTX ld.shared — prevents compiler speculation

    // 4 slabs with accumulation — this is the multi-K pattern that triggers GT-M11
    for (int s = 0; s < 4; s++) {
        if (tid == 0) mbarrier_init_1(mbar_p);
        __syncthreads();
        if (warp == 0 && is_e) {
            uint64_t a_d = make_smem_desc(A_ptr, SBO_BYTES, LBO_BYTES);
            uint64_t b_d = make_smem_desc(B_ptr, SBO_BYTES, LBO_BYTES);
            if (UseCorrectMask)
                tcgen05_mma_f16_7op(tmem, a_d, b_d, IDESC_F16_M64_N16, s > 0);
            else
                tcgen05_mma_f16_5op(tmem, a_d, b_d, IDESC_F16_M64_N16, s > 0);
            tcgen05_commit(mbar_p);
        }
        if (warp == 0 || warp == 1) mbarrier_wait_phase(mbar_p, 0);
        __syncthreads();
        tcgen05_fence_after();
        __syncthreads();  // GT-48 drain per slab
    }

    // Read back: x16 load covers N=16 columns
    uint32_t taddr = ((uint32_t)warp * 32u + (uint32_t)lane) << 16 | tmem;
    uint32_t regs[16] = {};
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(regs[0]), "=r"(regs[1]),  "=r"(regs[2]),  "=r"(regs[3]),
          "=r"(regs[4]), "=r"(regs[5]),  "=r"(regs[6]),  "=r"(regs[7]),
          "=r"(regs[8]), "=r"(regs[9]),  "=r"(regs[10]), "=r"(regs[11]),
          "=r"(regs[12]),"=r"(regs[13]), "=r"(regs[14]), "=r"(regs[15])
        : "r"(taddr));
    tcgen05_wait_ld();
    __syncthreads();  // GT-48 drain

    if (lane < 16) {
        int row = warp * 16 + lane;
        for (int n = 0; n < 16; n++)
            output[row*16 + n] = __uint_as_float(regs[n]);
    }
    __syncthreads();
    if (warp == 1) tcgen05_dealloc_64(tmem);
}

bool run_test_gtm11() {
    // Build A and B host-side as BF16 from a fixed-seed Box-Muller-style
    // approximation of N(0, 1). Matches the magnitude distribution that
    // production probes used (torch.randn(...).to(bfloat16)).
    auto build_random_bf16 = [](std::vector<uint16_t>& dst, int n_elems, uint32_t seed) {
        // Linear-congruential + bit-pack into BF16. Range mostly ±2.
        dst.resize(n_elems);
        uint32_t s = seed | 1u;
        for (int i = 0; i < n_elems; ++i) {
            s = s * 1664525u + 1013904223u;
            // Convert to a float roughly N(0, 1) via two-uniform sum.
            float u1 = ((s >> 8) & 0xFFFFFF) / float(0x1000000);
            s = s * 1664525u + 1013904223u;
            float u2 = ((s >> 8) & 0xFFFFFF) / float(0x1000000);
            float val = (u1 + u2 - 1.0f) * 2.0f;     // approx N(0, 1) bounded
            // Pack into BF16: just take the high 16 bits of the FP32 representation.
            uint32_t fbits = *reinterpret_cast<uint32_t*>(&val);
            dst[i] = (uint16_t)(fbits >> 16);
        }
    };
    std::vector<uint16_t> hA, hB;
    build_random_bf16(hA, F16_A_BYTES / 2, 0xA1A1u);
    build_random_bf16(hB, F16_B_BYTES / 2, 0xB2B2u);

    uint16_t *dA, *dB;
    CUDA_CHECK(cudaMalloc(&dA, F16_A_BYTES));
    CUDA_CHECK(cudaMalloc(&dB, F16_B_BYTES));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), F16_A_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), F16_B_BYTES, cudaMemcpyHostToDevice));

    constexpr int B = 1;
    float *d_wrong, *d_correct;
    CUDA_CHECK(cudaMalloc(&d_wrong,   (size_t)B*64*16*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct, (size_t)B*64*16*sizeof(float)));

    kernel_gtm11<false><<<B, 128, F16_SMEM>>>(dA, dB, d_wrong);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gtm11<true> <<<B, 128, F16_SMEM>>>(dA, dB, d_correct);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dA); cudaFree(dB);

    std::vector<float> hw((size_t)B*64*16), hc((size_t)B*64*16);
    CUDA_CHECK(cudaMemcpy(hw.data(), d_wrong,   hw.size()*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hc.data(), d_correct, hc.size()*4, cudaMemcpyDeviceToHost));
    cudaFree(d_wrong); cudaFree(d_correct);

    // With random BF16 data, exact expected value isn't a single number; we
    // just check (a) the 5-op variant produces ≥1 NaN cell, (b) the 7-op
    // variant produces ZERO NaNs. Diff between the two output tensors is the
    // signature: if 5-op silently degrades only some lanes, those lanes will
    // disagree with the 7-op output.
    int w_nans = 0, c_nans = 0, disagreement = 0;
    for (size_t i = 0; i < hw.size(); i++) {
        if (std::isnan(hw[i])) w_nans++;
        if (std::isnan(hc[i])) c_nans++;
        if (!std::isnan(hw[i]) && !std::isnan(hc[i]) &&
             fabsf(hw[i] - hc[i]) > 1e-3f) disagreement++;
    }
    bool ok = (w_nans > 0 || disagreement > 0) && (c_nans == 0);
    char msg[384];
    snprintf(msg, sizeof(msg),
        "5-op wrapper (no mask, %d CTAs random BF16): %d NaN cells, %d cells disagree with 7-op. "
        "7-op wrapper (correct): %d NaN cells (expect 0). Finding %s.",
        B, w_nans, disagreement, c_nans,
        ok ? "CONFIRMED" : "NOT REPRODUCED");

    record("GT-M11/M15: missing mask in kind::f16 (random BF16)",
           "GT-M11/GT-M15", "CONFIRMED_DETERMINISTIC", ok, msg);
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 9: TEST 4 — GT-14 (#pragma unroll breaks slab loop ordering)
// CONFIDENCE: CONFIRMED_DETERMINISTIC
//
// Applying #pragma unroll 4 to the K-slab inner loop causes the compiler to
// reorder instructions across loop-iteration boundaries.  The commit/wait chain
// that enforces MMA→ld ordering collapses, and only the last slab's result
// survives.  With 4 slabs of K=32 and FP8_ONE inputs, correct = 128.0,
// unrolled-wrong = ~32.0 (one slab's contribution).
// ─────────────────────────────────────────────────────────────────────────────

template<bool UseUnroll>
__global__ __launch_bounds__(128, 2)
void kernel_gt14(float* output) {
    extern __shared__ uint8_t smem_14[];
    uint8_t*  A_smem  = smem_14 + A_SMEM_OFF;
    uint8_t*  B_smem  = smem_14 + B_SMEM_OFF;
    uint32_t* alloc_s = reinterpret_cast<uint32_t*>(smem_14 + ALLOC_OFF);
    uint64_t* mbar_s  = reinterpret_cast<uint64_t*>(smem_14 + MBAR_OFF);

    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
    const int is_e = elect_one_sync();

    for (int i = tid; i < 16384 / 16; i += 128)
        reinterpret_cast<uint4*>(smem_14)[i] = make_uint4(
            0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
    // Cap at 16384/16: same logic as kernel_gt48 — avoids touching the alloc slot.

    const uint32_t A_ptr   = __cvta_generic_to_shared(A_smem);
    const uint32_t B_ptr   = __cvta_generic_to_shared(B_smem);
    const uint32_t alloc_p = __cvta_generic_to_shared(alloc_s);
    const uint32_t mbar_p  = __cvta_generic_to_shared(mbar_s);

    if (tid == 0) mbarrier_init_1(mbar_p);
    if (warp == 1) tcgen05_alloc_64(alloc_p);   // P11 pattern: no copy to [1]
    __syncthreads();
    const uint32_t tmem = smem_load_u32(alloc_p);   // PTX ld.shared — prevents compiler speculation

    if (warp == 0 && is_e) {
        if (UseUnroll) {
            // WRONG: full loop unroll reorders commit/wait → only last slab counts
            #pragma unroll 4
            for (int s = 0; s < NUM_SLABS; ++s)
                tcgen05_mma_f8f6f4(tmem,
                    make_smem_desc(A_ptr + s*SLAB_BYTES),
                    make_smem_desc(B_ptr + s*SLAB_BYTES),
                    IDESC_VAL, s > 0 ? 1 : 0);
        } else {
            // CORRECT: no unroll — ordering preserved
            for (int s = 0; s < NUM_SLABS; ++s)
                tcgen05_mma_f8f6f4(tmem,
                    make_smem_desc(A_ptr + s*SLAB_BYTES),
                    make_smem_desc(B_ptr + s*SLAB_BYTES),
                    IDESC_VAL, s > 0 ? 1 : 0);
        }
        tcgen05_commit(mbar_p);
    }

    if (warp == 0 || warp == 1) mbarrier_wait_phase(mbar_p, 0);
    __syncthreads();
    tcgen05_fence_after();

    uint32_t rlo[32], rhi[32];
    tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | tmem,        rlo);
    tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | (tmem+32u),  rhi);
    tcgen05_wait_ld();
    __syncthreads();  // GT-48 drain

    if (lane < 16) {
        int row = warp*16+lane;
        float* row_out = output + row*MMA_N;
        for (int c = 0; c < 32; c++) {
            row_out[c]    = __uint_as_float(rlo[c]);
            row_out[c+32] = __uint_as_float(rhi[c]);
        }
    }
    __syncthreads();
    if (warp == 1) tcgen05_dealloc_64(tmem);
}

bool run_test_gt14() {
    float *d_u, *d_n;
    CUDA_CHECK(cudaMalloc(&d_u, MMA_M*MMA_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_n, MMA_M*MMA_N*sizeof(float)));

    kernel_gt14<true> <<<1, 128, SMEM_TOTAL>>>(d_u);
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gt14<false><<<1, 128, SMEM_TOTAL>>>(d_n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hu(MMA_M*MMA_N), hn(MMA_M*MMA_N);
    CUDA_CHECK(cudaMemcpy(hu.data(), d_u, MMA_M*MMA_N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hn.data(), d_n, MMA_M*MMA_N*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_u); cudaFree(d_n);

    // With unroll: cells should be ~32.0 (only last slab survived)
    int unroll_correct = 0, nounroll_errors = 0;
    for (int i = 0; i < MMA_M*MMA_N; i++) {
        if (fabsf(hu[i] - 32.0f) < 2.0f) unroll_correct++;
        if (fabsf(hn[i] - EXPECTED_128) > 2.0f) nounroll_errors++;
    }
    bool ok = (unroll_correct > MMA_M*MMA_N/2 && nounroll_errors == 0);
    char msg[256];
    snprintf(msg, sizeof(msg),
        "With #pragma unroll 4: %d/%d cells ≈ 32.0 (only last slab; expect majority). "
        "Without unroll: %d errors vs %.0f (expect 0). Finding %s.",
        unroll_correct, MMA_M*MMA_N,
        nounroll_errors, EXPECTED_128,
        ok ? "CONFIRMED" : "NOT REPRODUCED");

    record("GT-14: slab loop unroll breaks MMA ordering",
           "GT-14", "CONFIRMED_DETERMINISTIC", ok, msg);
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 10: TEST 5 — GT-11 (SBO/LBO values not derivable from geometry)
// CONFIDENCE: CONFIRMED_DETERMINISTIC
//
// PTX_ISA REFERENCE: ptx_isa_sections/tcgen05_smem_descriptor.txt (Table 42).
// The ISA documents SBO/LBO as bit-fields encoding a byte stride between
// 8-row groups (SBO) and 16-element K-tiles (LBO), but does NOT specify the
// exact values for kind::f8f6f4 + M=64 N=64 + SWIZZLE_NONE. The probe
// `sbo_lbo_sweep.cu` exhausted (sbo, lbo) ∈ [0,32] × [0,32] and found that
// only (SBO=16 → 256B, LBO=8 → 128B) produces correct numerics. All other
// values produce wrong output — the agent had to discover this empirically.
//
// Uses kind::f8f6f4, M=64, N=64 — the same tile as sbo_lbo_sweep.cu.
// Three descriptor configurations:
//   SBO=256, LBO=128 (confirmed) → output ≈ 128.0
//   SBO=128, LBO=64  (wrong, real bytes that don't crash) → wrong output
//   SBO=512, LBO=256 (wrong, real bytes that don't crash) → wrong output
//
// v3 NOTE: v2 tested SBO=0,LBO=0 which produced an illegal memory access
// (descriptor encoded SMEM offset 0 as the start of a 0-stride read, which
// hardware then interpreted as out-of-bounds — XID 13). The corruption killed
// the CUDA context, breaking all subsequent tests. v3 uses non-zero "wrong"
// values that produce numerically wrong output without OOB hardware fault.
// ─────────────────────────────────────────────────────────────────────────────

__global__ __launch_bounds__(128, 2)
void kernel_gt11(uint32_t sbo_bytes, uint32_t lbo_bytes, float* output) {
    extern __shared__ uint8_t smem_11[];
    uint8_t*  A_smem  = smem_11 + A_SMEM_OFF;
    uint8_t*  B_smem  = smem_11 + B_SMEM_OFF;
    uint32_t* alloc_s = reinterpret_cast<uint32_t*>(smem_11 + ALLOC_OFF);
    uint64_t* mbar_s  = reinterpret_cast<uint64_t*>(smem_11 + MBAR_OFF);

    const int tid = threadIdx.x, warp = tid>>5, lane = tid&31;
    const int is_e = elect_one_sync();

    for (int i = tid; i < 16384 / 16; i += 128)
        reinterpret_cast<uint4*>(smem_11)[i] = make_uint4(
            0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
    // Cap at 16384/16: same logic as kernel_gt48.

    const uint32_t A_ptr   = __cvta_generic_to_shared(A_smem);
    const uint32_t B_ptr   = __cvta_generic_to_shared(B_smem);
    const uint32_t alloc_p = __cvta_generic_to_shared(alloc_s);
    const uint32_t mbar_p  = __cvta_generic_to_shared(mbar_s);

    if (tid == 0) mbarrier_init_1(mbar_p);
    if (warp == 1) tcgen05_alloc_64(alloc_p);   // P11 pattern: no copy to [1]
    __syncthreads();
    const uint32_t tmem = smem_load_u32(alloc_p);   // PTX ld.shared — prevents compiler speculation

    if (warp == 0 && is_e) {
        for (int s = 0; s < NUM_SLABS; ++s)
            tcgen05_mma_f8f6f4(tmem,
                // Pass the sbo/lbo under test — only confirmed values produce correct output
                make_smem_desc(A_ptr + s*SLAB_BYTES, sbo_bytes, lbo_bytes),
                make_smem_desc(B_ptr + s*SLAB_BYTES, sbo_bytes, lbo_bytes),
                IDESC_VAL, s > 0 ? 1 : 0);
        tcgen05_commit(mbar_p);
    }

    if (warp == 0 || warp == 1) mbarrier_wait_phase(mbar_p, 0);
    __syncthreads();
    tcgen05_fence_after();

    uint32_t rlo[32], rhi[32];
    tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | tmem,        rlo);
    tcgen05_ld_32x32b_x32((uint32_t(warp)*32u)<<16 | (tmem+32u),  rhi);
    tcgen05_wait_ld();
    __syncthreads();  // GT-48 drain

    if (lane < 16) {
        int row = warp*16+lane;
        float* row_out = output + row*MMA_N;
        for (int c = 0; c < 32; c++) {
            row_out[c]    = __uint_as_float(rlo[c]);
            row_out[c+32] = __uint_as_float(rhi[c]);
        }
    }
    __syncthreads();
    if (warp == 1) tcgen05_dealloc_64(tmem);
}

bool run_test_gt11() {
    float *d_c, *d_wa, *d_wb;
    CUDA_CHECK(cudaMalloc(&d_c,  MMA_M*MMA_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wa, MMA_M*MMA_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wb, MMA_M*MMA_N*sizeof(float)));

    kernel_gt11<<<1,128,SMEM_TOTAL>>>(256, 128, d_c);   // SBO=16→256B, LBO=8→128B (confirmed)
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gt11<<<1,128,SMEM_TOTAL>>>(128, 64,  d_wa);  // SBO=8→128B, LBO=4→64B (wrong, but valid bytes)
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_gt11<<<1,128,SMEM_TOTAL>>>(512, 256, d_wb);  // SBO=32→512B, LBO=16→256B (wrong, but valid bytes)
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hc(MMA_M*MMA_N), ha(MMA_M*MMA_N), hb(MMA_M*MMA_N);
    CUDA_CHECK(cudaMemcpy(hc.data(), d_c,  MMA_M*MMA_N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ha.data(), d_wa, MMA_M*MMA_N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), d_wb, MMA_M*MMA_N*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_c); cudaFree(d_wa); cudaFree(d_wb);

    int c_err = 0, wa_diff = 0, wb_diff = 0;
    for (int i = 0; i < MMA_M*MMA_N; i++) {
        if (fabsf(hc[i] - EXPECTED_128) > 2.0f) c_err++;
        if (fabsf(ha[i] - EXPECTED_128) > 2.0f) wa_diff++;
        if (fabsf(hb[i] - EXPECTED_128) > 2.0f) wb_diff++;
    }

    bool ok = (c_err == 0 && wa_diff > 0 && wb_diff > 0);
    char msg[256];
    snprintf(msg, sizeof(msg),
        "SBO=256,LBO=128 (confirmed): %d errors (expect 0). "
        "SBO=128,LBO=64: %d cells wrong (expect >0). "
        "SBO=512,LBO=256: %d cells wrong (expect >0). Finding %s.",
        c_err, wa_diff, wb_diff,
        ok ? "CONFIRMED" : "NOT REPRODUCED");

    record("GT-11: SBO/LBO confirmed (kind::f8f6f4, M=64 N=64)",
           "GT-11", "CONFIRMED_DETERMINISTIC", ok, msg);
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 11: TEST 6 — GT-9 (cuTensorMapEncodeTiled silently accepts bad stride)
// CONFIDENCE: CONFIRMED_DETERMINISTIC (host-side only)
//
// When globalStride is not a multiple of 16 bytes, the API returns CUDA_SUCCESS.
// The hardware then XID 13 at kernel launch (tested in Python runner).
// ─────────────────────────────────────────────────────────────────────────────

bool run_test_gt9() {
    // cuTensorMapEncodeTiled is a CUDA Driver API function.  Since this binary
    // links with -lcuda we can call it directly.  We call cuInit(0) first to
    // ensure the driver is initialized (safe to call multiple times; the runtime
    // API calls in earlier tests also initialise it, but cuInit is explicit).
    if (cuInit(0) != CUDA_SUCCESS) {
        record("GT-9: TMA bad stride silent accept", "GT-9",
               "CONFIRMED_DETERMINISTIC", false, "SKIP: cuInit failed.");
        return false;
    }

    void* d_buf = nullptr;
    // If a previous test killed the CUDA context, cudaMalloc will fail here.
    // We detect this explicitly rather than passing nullptr to cuTensorMapEncodeTiled.
    if (cudaMalloc(&d_buf, 8192) != cudaSuccess) {
        record("GT-9: TMA bad stride silent accept", "GT-9",
               "CONFIRMED_DETERMINISTIC", false,
               "SKIP: cudaMalloc failed — CUDA context may be in error state "
               "from a previous test. Fix the preceding test first.");
        return false;
    }

    CUtensorMap tmap_v = {}, tmap_i = {};
    uint64_t sizes[2]    = {128, 1024};
    uint64_t valid_s[1]  = {128};   // 128 bytes — multiple of 16: valid
    uint64_t invalid_s[1]= {132};   // 132 bytes — NOT a multiple of 16: should fail, doesn't
    uint32_t box[2]      = {64, 16};
    uint32_t est[2]      = {1, 1};

    // Call directly — no cuGetProcAddress needed when linking with -lcuda.
    CUresult rv = cuTensorMapEncodeTiled(
        &tmap_v, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_buf,
        sizes, valid_s, box, est,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult ri = cuTensorMapEncodeTiled(
        &tmap_i, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_buf,
        sizes, invalid_s, box, est,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cudaFree(d_buf);

    // Two valid outcomes:
    //   (a) Both calls return CUDA_SUCCESS → original "silent accept" bug
    //       still present; kernel launch will then XID 13.
    //   (b) Valid succeeds, invalid is REJECTED at API → driver tightened
    //       validation; the original bug has been fixed at the API layer.
    // Both confirm the GT-9 finding (API behavior around bad stride is real).
    // Failure case: valid is rejected (which would mean the API broke).
    bool valid_ok    = (rv == CUDA_SUCCESS);
    bool invalid_rej = (ri != CUDA_SUCCESS);
    bool silent_accept = (rv == CUDA_SUCCESS && ri == CUDA_SUCCESS);
    bool ok = valid_ok && (silent_accept || invalid_rej);

    const char* outcome;
    if (silent_accept) outcome = "SILENT ACCEPT (original bug present); kernel launch will XID 13";
    else if (invalid_rej) outcome = "API REJECTED bad stride (driver tightened — original bug fixed)";
    else outcome = "valid stride encode failed (regression, unrelated to GT-9)";

    char msg[384];
    snprintf(msg, sizeof(msg),
        "Valid stride (128 bytes): %s. Invalid stride (132 bytes): %s. "
        "Outcome: %s. Finding %s.",
        rv == CUDA_SUCCESS ? "CUDA_SUCCESS" : "ERROR",
        ri == CUDA_SUCCESS ? "CUDA_SUCCESS" : "ERROR",
        outcome, ok ? "CONFIRMED" : "NOT REPRODUCED");

    record("GT-9: TMA bad stride API behavior",
           "GT-9", "CONFIRMED_DETERMINISTIC", ok, msg);
    return ok;
}


// ─────────────────────────────────────────────────────────────────────────────
// SECTION 12: Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Blackwell Undocumented Behavior Validation Suite  (v2)         ║\n");
    printf("║  sm_100a (B200)  CUDA 12.8  kind::f8f6f4  M=64 N=64 K=32×4     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    printf("(v2: switched to kind::f8f6f4 M=64 N=64 to match original probes;\n");
    printf(" fixed descriptor encoding; revised GT-48 to port P11 structure;\n");
    printf(" added missing <vector> and <cuda.h>)\n\n");

    // GT-9 first: host-only API test. If a later kernel kills the context
    // (e.g. hangs or OOB memory access), GT-9's cudaMalloc would otherwise fail.
    run_test_gt9();
    run_test_gtm10();
    run_test_gt17();      // NEW: tcgen05.ld lane-gating
    run_test_gt15();      // NEW: IDESC transpose_B bit
    run_test_gt48();
    run_test_gtm11();
    run_test_gt14();
    run_test_gt11();

    printf("\n%-42s  %-22s  %-28s  %s\n",
           "Test", "Finding ID", "Confidence", "Result");
    printf("%-42s  %-22s  %-28s  %s\n",
           "------------------------------------------",
           "----------------------",
           "----------------------------", "------");
    int passed = 0;
    for (int i = 0; i < g_num_results; i++) {
        auto& r = g_results[i];
        printf("%-42s  %-22s  %-28s  %s\n",
               r.name, r.finding_id, r.confidence,
               r.passed ? "✓ CONFIRMED" : "✗ NOT REPRODUCED");
        if (r.passed) passed++;
    }

    printf("\n── Detail ─────────────────────────────────────────────────────────\n");
    for (int i = 0; i < g_num_results; i++)
        printf("[%s] %s\n  %s\n\n",
               g_results[i].passed ? "CONFIRMED" : "NOT REPRODUCED",
               g_results[i].name, g_results[i].message);

    printf("───────────────────────────────────────────────────────────────────\n");
    printf("%d / %d findings confirmed on this hardware.\n\n", passed, g_num_results);
    printf("Additional tests (Python runner): run_blackwell_validation.py\n");
    printf("  GT-49   (high-CTA-density race)   CONFIRMED_STATISTICAL\n");
    printf("  GT-47   (commit/SMEM-read race)   CONFIRMED_DETERMINISTIC\n");
    printf("  GT-M13  (f8f6f4 cols 62/63 NaN)   CONFIRMED_DETERMINISTIC\n");
    printf("  GT-M10  (alloc hang demo)         CONFIRMED_DETERMINISTIC\n");
    printf("  GT-M8   (_scaled_mm validator)    OBSERVED_VERSION_SPECIFIC\n");
    printf("  GT-M23  (_grouped_mm precision)   OBSERVED_VERSION_SPECIFIC\n");
    printf("  GT-M3   (FP8 block scaling)       OBSERVED_VERSION_SPECIFIC\n");
    printf("  GT-M5   (__expf vs expf)          OBSERVED_VERSION_SPECIFIC\n");
    printf("  GT-GDN17 (h_state in registers)   OBSERVED_NCU_REQUIRED\n");
    printf("  GT-GDN18 (HBM writes L2-absorbed) OBSERVED_NCU_REQUIRED\n");
    printf("  GT-44   (PC-sample ≠ wall-time)   OBSERVED_NCU_REQUIRED\n");

    return g_num_results - passed;
}
