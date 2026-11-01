// NR 310.9.16 projection: residual initializes z0, f16 rounding per32 k and ordered split-K folds.
#include "vitfuse_common.cuh"
    struct __align__(8) vitfuse_ConvParams
    {
        const unsigned char* input;    // +0
        const void* unused8;           // +8
        unsigned char* output;         // +16
        const unsigned char* weights;  // +24
        const void* unused32;          // +32
        const void* unused40;          // +40
        const int* waitFlags;          // +48, chained: producer tile flags
        int* signalFlags;              // +56, chained/publish: this kernel's tile flags
        int height, width;             // +64
    };
    static_assert(sizeof(vitfuse_ConvParams) == 72, "parameter block size");

    // One CTA (4 warps) computes 128 tokens x 128 output channels of an FP8 1x1 conv over
    // K input channels of a C-channel input (`input` and `weights` already offset to the
    // first of them). Warp w: token blocks 4(w&1)..+3 of the tile, output slices
    // 2(w>>1), +1. Lane 4g+t: rows g, g+8 of each block; stored positions 4t..4t+3 and
    // 16+4t.. of each slice (16 channels). Shared: A[k][128 tokens], B[k][128 channels],
    // reordered so each thread's 8 tokens / 16 channels are contiguous.
    template <int K, int C, int N>
    __device__ __forceinline__ void vitfuse_ConvTile(const unsigned char* input, const unsigned char* weights,
                                                float (&acc)[4][16], int tokenTile, int channelTile, int blocks16, const unsigned char* residual, const unsigned char* gamma)
    {
        __shared__ __align__(16) float vitfuse_proj_As[32][128];
        __shared__ __align__(16) float vitfuse_proj_Bs[32][128];
        const int lane = threadIdx.x, warp = threadIdx.y, thread = warp * 32 + lane;
        const int g = lane >> 2, t = lane & 3, wr = warp & 3, wc = warp >> 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 16; j+=2) {
                int block=tokenTile*8+wr*2+i/2;
                int row=g+8*(i&1), sl=j/8, jj=j&7;
                int chan=(channelTile*4+wc*2+sl)*32+16*(jj/4)+vitfuse_channel(4*t+(jj&3));
                float v0=0.f,v1=0.f;
                if(residual&&block<blocks16) {
                    unsigned word=*reinterpret_cast<const unsigned short*>(residual+block*16*N+vitfuse_aoff(row,vitfuse_stored(chan)));
                    float2 g=__half22float2(*reinterpret_cast<const __half2*>(gamma+2*chan));
                    v0=vitfuse_half(vitfuse_decode(word)*g.x);v1=vitfuse_half(vitfuse_decode(word>>8)*g.y);
                }
                acc[i][j]=v0;acc[i][j+1]=v1;
            }
        }
        for (int slice = 0; slice < K / 32; ++slice) {
            __syncthreads();
            if (thread < 128) {   // A: 8 token blocks x 512 bytes; thread loads 32 bytes of block thread/16.
                const int block = thread >> 4, part = thread & 15;
                const int globalBlock = tokenTile * 8 + block;
                uint4 words[2] = {};
                if (globalBlock < blocks16) {
                    const uint4* source = reinterpret_cast<const uint4*>(
                        input + static_cast<size_t>(globalBlock) * 16 * C + slice * 512 + part * 32);
                    words[0] = source[0]; words[1] = source[1];
                }
                const unsigned* w = reinterpret_cast<const unsigned*>(words);
#pragma unroll
                for (int word = 0; word < 8; ++word) {
                    const int offset = part * 32 + word * 4;       // byte offset in the 512-byte slice
                    const int fragLane = offset >> 4, reg = (offset >> 2) & 3;
                    const int row = (fragLane >> 2) + 8 * (reg & 1);
                    const int kBase = 4 * (fragLane & 3) + 16 * (reg >> 1);
                    const int slot = (block >> 2)*64 + ((block & 3)>>1)*32 + (row & 7)*4 + (block & 1)*2 + (row>>3);
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) { vitfuse_proj_As[kBase + byte][slot] = vitfuse_decode(w[word] >> (8 * byte)); }
                }
            }
            if (thread < 128) {   // B: 128 channels x 32 k = 4096 contiguous bytes; thread loads 32 bytes.
                const uint4* source = reinterpret_cast<const uint4*>(
                    weights + static_cast<size_t>(slice) * N * 32 +
                    static_cast<size_t>(channelTile) * 128 * 32 + thread * 32);
                const uint4 words[2] = {source[0], source[1]};
                const unsigned* w = reinterpret_cast<const unsigned*>(words);
#pragma unroll
                for (int word = 0; word < 8; ++word) {
                    const int offset = thread * 32 + word * 4;      // byte offset in the 4096 bytes
                    const int group = offset >> 11, rest = offset & 2047;
                    const int load = rest >> 9, fragLane = (rest >> 4) & 31, q = (rest >> 2) & 3;
                    const int nTile = 2 * load + (q >> 1), kBase = 4 * (fragLane & 3) + 16 * (q & 1);
                    const int n = group * 64 + nTile * 8 + (fragLane >> 2);   // channel within the tile
                    const int sl = n >> 5, h2 = (n >> 4) & 1, hi = (n >> 3) & 1, tt = (n >> 1) & 3, lo = n & 1;
                    const int slot = (sl>>1)*64 + ((sl&1)*2+h2)*16 + tt*4 + hi*2+lo;
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) { vitfuse_proj_Bs[kBase + byte][slot] = vitfuse_decode(w[word] >> (8 * byte)); }
                }
            }
            __syncthreads();
            const float4* aRow = reinterpret_cast<const float4*>(&vitfuse_proj_As[0][wr*32+g*4]);
            const float4* bRow = reinterpret_cast<const float4*>(&vitfuse_proj_Bs[0][wc*64+t*4]);
#pragma unroll 4
            for (int k = 0; k < 32; ++k) {
                const float4 a0 = aRow[k * 32];
                const float4 b0 = bRow[k * 32], b1 = bRow[k * 32 + 4], b2 = bRow[k * 32 + 8], b3 = bRow[k * 32 + 12];
                const float a[4] = {a0.x, a0.y, a0.z, a0.w};
                const float b[16] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w,
                                     b2.x, b2.y, b2.z, b2.w, b3.x, b3.y, b3.z, b3.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 16; ++j) { acc[i][j] = fmaf(a[i], b[j], acc[i][j]); }
                }
            }
#pragma unroll
            for (int i=0;i<4;i++) {
#pragma unroll
                for (int j=0;j<16;j++) acc[i][j]=vitfuse_half(acc[i][j]);
            }
        }
    }

    struct __align__(8) vitfuse_ProjectionParams
    {
        const unsigned char* input;     // +0, 1024 channels
        const unsigned char* residual;  // +8, 1024 channels, same layout as the output
        unsigned char* output;          // +16
        const unsigned char* weights;   // +24, then f16 gamma[1024] (residual scale)
        int* sliceFlags;                // +32, per tile: last K quarter (z) folded in
        __half* partial;                // +40, f16 running sum, one per output byte
        const int* waitFlags;           // +48: attention head/tile flags
        int* signalFlags;               // +56, chained: this kernel's tile flags
        int height, width;              // +64
    };
    static_assert(sizeof(vitfuse_ProjectionParams) == 72, "parameter block size");

    // Split K over blockIdx.z (quarters of 256 input channels), folded in z order as NVIDIA
    // does: z=0 stores residual + its sum as f16, z=1 and 2 add theirs in f16, z=3 adds its
    // own and writes FP8. sliceFlags[tile] = z once quarter z is in.
    template <bool Wait, bool Signal>
    __device__ __forceinline__ void vitfuse_Projection(const vitfuse_ProjectionParams& p)
    {
        constexpr int K = 256, C = 1024, N = 1024;
        const int tokens = p.height * p.width;
        const int tiles = (tokens - 1) / 128 + 1;
        const int channelTile = blockIdx.x / tiles, tokenTile = blockIdx.x % tiles;
        const int blocks16 = (((tokens - 1) + 32) & ~31) / 16;
        const int z = blockIdx.z, lane = threadIdx.x, warp = threadIdx.y, wr = warp & 3, wc = warp >> 2;
        const bool leader = threadIdx.x == 0 && threadIdx.y == 0;
        if (Wait) {   // This K quarter consumes eight attention heads; wait on each.
            if (leader) {
                for (int y = 0; y < 4; y += 2) {
                    const int first = tokenTile * 128 + y * 32;
                    const int last = min(((first + 32) >> 7) + 1, tiles);
                    for (int tile = first >> 7; tile < last; ++tile) {
                        for (int c = 0; c < 8; ++c) { vitfuse_wait(p.waitFlags + tile * 32 + z * 8 + c); }
                    }
                }
                __threadfence();
            }
            __syncthreads();
        }
        float acc[4][16];
        vitfuse_ConvTile<K, C, N>(p.input + z * (K / 32) * 512, p.weights + static_cast<size_t>(z) * K * N, acc,
                             tokenTile, channelTile, blocks16, z==0?p.residual:nullptr, p.weights+C*N);
        const int flag = tokenTile * 8 + channelTile;
        if (z > 0) {
            if (leader) {
                while (*reinterpret_cast<const volatile int*>(p.sliceFlags + flag) < z - 1) {}
                __threadfence();
            }
            __syncthreads();
        }
#pragma unroll
        for (int bi = 0; bi < 2; ++bi) {
            const int globalBlock = tokenTile * 8 + wr * 2 + bi;
            if (globalBlock >= blocks16) { continue; }
#pragma unroll
            for (int sl = 0; sl < 2; ++sl) {
                const size_t base = static_cast<size_t>(globalBlock) * 16 * N + (channelTile * 4 + wc * 2 + sl) * 512 + lane * 16;
                float v[16];
#pragma unroll
                for (int q = 0; q < 16; ++q) {   // lane byte q: word q/4 as in StoreTile
                    v[q] = acc[bi * 2 + ((q >> 2) & 1)][sl * 8 + 4 * (q >> 3) + (q & 3)];
                }
                uint4* partial = reinterpret_cast<uint4*>(p.partial + base);
                if (z > 0) {
                    const uint4 s0 = __ldcg(partial), s1 = __ldcg(partial + 1);
                    const unsigned sw[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        const __half h = __ushort_as_half(static_cast<unsigned short>(sw[q >> 1] >> (16 * (q & 1))));
                        v[q] = vitfuse_half(__half2float(h) + vitfuse_half(v[q]));
                    }
                }
                if (z < 3) {
                    unsigned hw[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        hw[i] = __half_as_ushort(__float2half_rn(v[2 * i])) |
                                (static_cast<unsigned>(__half_as_ushort(__float2half_rn(v[2 * i + 1]))) << 16);
                    }
                    __stcg(partial, make_uint4(hw[0], hw[1], hw[2], hw[3]));
                    __stcg(partial + 1, make_uint4(hw[4], hw[5], hw[6], hw[7]));
                } else {
                    unsigned ow[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        ow[i] = vitfuse_encode(v[4 * i]) | (vitfuse_encode(v[4 * i + 1]) << 8) |
                                (vitfuse_encode(v[4 * i + 2]) << 16) | (vitfuse_encode(v[4 * i + 3]) << 24);
                    }
                    *reinterpret_cast<uint4*>(p.output + base) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
                }
            }
        }
        __threadfence();
        __syncthreads();
        if (leader) {
            *reinterpret_cast<volatile int*>(p.sliceFlags + flag) = z;
            if (Signal && z == 3) { *reinterpret_cast<volatile int*>(p.signalFlags + flag) = 0; }
        }
    }


extern "C" __global__ void __launch_bounds__(256,2) cc_vit_1d_projection_chained_fp8(vitfuse_ProjectionParams p) {vitfuse_Projection<true,true>(p);}
extern "C" __global__ void __launch_bounds__(256,2) cc_vit_1d_projection_wait_fp8(vitfuse_ProjectionParams p) {vitfuse_Projection<true,false>(p);}

