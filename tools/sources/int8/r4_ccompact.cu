#define TRP_SKIP_EXPAND
#define TRP_SKIP_QKV
// 288-token research schedule; original grid/block, external 128-token flags.
// Research: native Pascal (sm_61) replacements for NR 310.9.16 ViT kernels. Same entry
// names, parameter blocks, data layouts and flag protocol as NVIDIA's kernels; FP8 is
// decoded once per CTA into shared memory and multiplied with f32 FMA.
//
// Layouts (verified against NVIDIA's kernels on an RTX 4080 Super, bench/layouts.py):
//   activations: 16-token blocks of tokens*channels bytes; per 32-channel slice 512 bytes
//     in mma A-fragment order: lane 4g+t holds rows g / g+8, stored k 4t..4t+3 / +16.
//   kernel outputs: written from accumulators, so stored k 4t+i of a 16-channel half
//     holds channel 2t + (i&1) + 8(i>>1).
//   weights: 32-k slices of N*32 bytes; per 64 output channels 2048 bytes; load j of lane
//     4g+t holds n-tiles 2j, 2j+1 (8 columns each) as b0 (k 4t..4t+3), b1 (k 4t+16..).
#include <cuda_fp16.h>

// Model weights remain immutable while launches are in flight. Bump epoch only
// after quiescing this module to invalidate changed content at a reused pointer.
// Clear keys and ready stamps, also while quiescent, to reclaim all sixteen slots.
extern "C" {
__device__ unsigned long long int8_c_keys[8];
__device__ unsigned long long int8_c_ready[8][32];
__device__ unsigned int8_c_epoch = 1;
__device__ unsigned int8_c_safe[8][32];
__device__ __align__(16) unsigned int8_c_words[8][32][32][8][128];
__device__ float int8_c_scale[8][32][128];
}


namespace
{
    __device__ __forceinline__ float int8_E4M3(unsigned bits)
    {
        return __uint_as_float(((bits & 0x80u) << 24) | ((bits & 0x7Fu) << 20)) * 1.329227995784916e36f;
    }

    __device__ __forceinline__ unsigned int8_EncodeE4M3(float value)
    {
        const unsigned sign = (__float_as_uint(value) >> 24) & 0x80u;
        if (value != value) { return sign | 0x7Fu; }
        const float magnitude = fminf(fabsf(value), 448.0f);
        unsigned code;
        if (magnitude < 0.015625f) {
            code = static_cast<unsigned>(__float2int_rn(magnitude * 512.0f));
        } else {
            unsigned bits = __float_as_uint(magnitude);
            bits += 0x7FFFFu + ((bits >> 20) & 1u);
            code = (bits >> 20) - (120u << 3);
        }
        return sign | code;
    }

    __device__ __forceinline__ float int8_Half(float value) { return __half2float(__float2half_rn(value)); }

    // NVIDIA's MpCubicSiluActivation in f16 arithmetic (each step rounded to f16):
    // y * 1.7890625 * (1/2 + c/4 - c|c|/32), c = clamp(y, -4, 4).
    __device__ __forceinline__ float int8_CubicSilu(float y)
    {
        const float c = fminf(fmaxf(y, -4.0f), 4.0f);
        const float p = int8_Half(fmaf(-0.055908203125f, fabsf(c), 0.447265625f));
        const float t = int8_Half(fmaf(c, p, 0.89453125f));
        return int8_Half(y * t);
    }

    struct __align__(8) int8_Conv1x1Params
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
    static_assert(sizeof(int8_Conv1x1Params) == 72, "parameter block size");

    // One CTA (4 warps) computes 128 tokens x 128 output channels of an FP8 1x1 conv over
    // K input channels of a C-channel input (`input` and `weights` already offset to the
    // first of them). Warp w: token blocks 4(w&1)..+3 of the tile, output slices
    // 2(w>>1), +1. Lane 4g+t: rows g, g+8 of each block; stored positions 4t..4t+3 and
    // 16+4t.. of each slice (16 channels). Shared: A[k][128 tokens], B[k][128 channels],
    // reordered so each thread's 8 tokens / 16 channels are contiguous.

    __device__ __forceinline__ int int8_Fixed16(unsigned code,int exponent) {
        const float inv=__uint_as_float(unsigned(148-exponent)<<23);
        return __float_as_int(fmaf(int8_E4M3(code),inv,12582912.0f))-0x4b400000;
    }
    __device__ __forceinline__ int int8_Fixed(unsigned code,int maximumExponent) {
        const float inv=__uint_as_float(unsigned(140-maximumExponent)<<23);
        return __float_as_int(fmaf(int8_E4M3(code),inv,12582912.0f))-0x4b400000;
    }
    template <int K, int C, int N>
    __device__ __forceinline__ void int8_Fp32Tile(const unsigned char* input, const unsigned char* weights,
                                                float (&acc)[6][16], int tokenTile, int channelTile, int blocks16)
    {
        __shared__ __align__(16) float As[32][96];
        __shared__ __align__(16) float Bs[32][128];
        const int lane = threadIdx.x, warp = threadIdx.y, thread = warp * 32 + lane;
        const int g = lane >> 2, t = lane & 3, wr = warp & 1, wc = warp >> 1;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
#pragma unroll
            for (int j = 0; j < 16; ++j) { acc[i][j] = 0.0f; }
        }
        for (int slice = 0; slice < K / 32; ++slice) {
            __syncthreads();
            {   // A: 8 token blocks x 512 bytes; thread loads 32 bytes of block thread/16.
                const int block = thread >> 4, part = thread & 15;
                const int globalBlock = tokenTile * 6 + block;
                uint4 words[2] = {};
                if (block<6 && globalBlock < blocks16) {
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
                    const int slot = (block/3)*48+(row&7)*2+(block%3)*16+(row>>3);
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) { if(block<6)As[kBase + byte][slot] = int8_E4M3(w[word] >> (8 * byte)); }
                }
            }
            {   // B: 128 channels x 32 k = 4096 contiguous bytes; thread loads 32 bytes.
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
                    for (int byte = 0; byte < 4; ++byte) { Bs[kBase + byte][slot] = int8_E4M3(w[word] >> (8 * byte)); }
                }
            }
            __syncthreads();
            const float2* aRow=reinterpret_cast<const float2*>(&As[0][wr*48+g*2]);
            const float4* bRow = reinterpret_cast<const float4*>(&Bs[0][wc*64+t*4]);
#pragma unroll 4
            for (int k = 0; k < 32; ++k) {
                const float2 a0=aRow[k*48],a1=aRow[k*48+8],a2=aRow[k*48+16];
                const float4 b0 = bRow[k * 32], b1 = bRow[k * 32 + 4], b2 = bRow[k * 32 + 8], b3 = bRow[k * 32 + 12];
                const float a[6]={a0.x,a0.y,a1.x,a1.y,a2.x,a2.y};
                const float b[16] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w,
                                     b2.x, b2.y, b2.z, b2.w, b3.x, b3.y, b3.z, b3.w};
#pragma unroll
                for (int i = 0; i < 6; ++i) {
#pragma unroll
                    for (int j = 0; j < 16; ++j) { acc[i][j] = fmaf(a[i], b[j], acc[i][j]); }
                }
            }
        }
    }




    template<int N>
    __device__ __forceinline__ int int8_c_prepare(const unsigned char* key,
            const unsigned char* tileWeights, int tile, int tokenTile) {
        __shared__ int int8_c_slot;
        __shared__ unsigned long long int8_c_stamp;
        __shared__ bool int8_c_build;
        const int thread=threadIdx.y*32+threadIdx.x;
        if(thread==0) {
            int found=-1;
            const unsigned long long k=(unsigned long long)key;
            for(int i=0;i<8;++i) {
                unsigned long long old=*reinterpret_cast<volatile unsigned long long*>(int8_c_keys+i); if(old==0)old=atomicCAS(int8_c_keys+i,0ull,k);
                if(old==0 || old==k) {found=i;break;}
            }
            unsigned signature=2166136261u;
            const int probe[4]={0,2044,16*N*32+1024,31*N*32+4092};
#pragma unroll
            for(int i=0;i<4;++i)signature=(signature^*reinterpret_cast<const volatile unsigned*>(tileWeights+probe[i]))*16777619u;
            int8_c_slot=found;
            int8_c_stamp=((unsigned long long)int8_c_epoch<<32)|(signature|1u);
            int8_c_build=found>=0 && tokenTile==0 &&
                *reinterpret_cast<volatile unsigned long long*>(&int8_c_ready[found][tile])!=int8_c_stamp;
        }
        __syncthreads();
        const int slot=int8_c_slot;
        if(slot<0)return -1;
        if(int8_c_build) {
            // For each channel tile, tokenTile0 precedes every consumer in ordered dispatch.
            if(thread==0)*reinterpret_cast<volatile unsigned long long*>(&int8_c_ready[slot][tile])=0;

            __shared__ int int8_c_coldexp[128];
            __shared__ unsigned int8_c_sums[256];
            __shared__ unsigned int8_c_inexact[256];
            if(thread<128) {
                const int col=thread;
                const int n=(col/16)*16+((col%16)/4)*2+((col%4)/2)*8+(col&1);
                unsigned maximum=0;
                for(int ss=0;ss<32;++ss) {
#pragma unroll
                    for(int kk=0;kk<8;++kk) {
                        int at=(n/16)*512+(n&7)*64+(kk&3)*16+((n>>3)&1)*8+(kk>>2)*4;
                        unsigned code=*reinterpret_cast<const unsigned*>(tileWeights+(size_t)ss*N*32+at);
#pragma unroll
                        for(int bb=0;bb<4;++bb)maximum=max(maximum,(code>>(8*bb))&127u);
                    }
                }
                int8_c_coldexp[col]=max(int(maximum>>3),1);
            }
            __syncthreads();
            unsigned int8_sumabs=0, int8_inexact=0;
            for(int task=thread;task<32*128;task+=256) {
                const int slice=task/128,col=task%128;
                const int n=(col/16)*16+((col%16)/4)*2+((col%4)/2)*8+(col&1);
                unsigned code[8];int maximum=0;
#pragma unroll
                for(int kk=0;kk<8;++kk) {
                    int at=(n/16)*512+(n&7)*64+(kk&3)*16+((n>>3)&1)*8+(kk>>2)*4;
                    if(kk<4){uint2 pair=*reinterpret_cast<const uint2*>(tileWeights+(size_t)slice*N*32+at);code[kk]=pair.x;code[kk+4]=pair.y;}
#pragma unroll
                    for(int bb=0;bb<4;++bb)maximum=max(maximum,int((code[kk]>>(8*bb))&127));
                }
                const int exponent=int8_c_coldexp[col];
                if(slice==0)int8_c_scale[slot][tile][col]=__uint_as_float(unsigned(exponent+114)<<23);
#pragma unroll
                for(int kk=0;kk<8;++kk) {
                    unsigned packed=0;
#pragma unroll
                    for(int bb=0;bb<4;++bb) {int v=int8_Fixed(code[kk]>>(8*bb),exponent); int8_inexact|=(float(v)*__uint_as_float(unsigned(exponent+114)<<23)!=int8_E4M3(code[kk]>>(8*bb))); int8_sumabs+=unsigned(abs(v));packed|=(unsigned(v)&255u)<<(8*bb);}
                    int8_c_words[slot][tile][slice][kk][col]=packed;
                }
            }
            int8_c_sums[thread]=int8_sumabs;
            int8_c_inexact[thread]=int8_inexact;
            __syncthreads();
            if(thread==0) {
                unsigned maximum=0;
                for(int col=0;col<128;++col)maximum=max(maximum,int8_c_sums[col]+int8_c_sums[col+128]);
                unsigned inexact=0;
                for(int i=0;i<256;++i)inexact|=int8_c_inexact[i];
                int8_c_safe[slot][tile]=(maximum<=69905u && !inexact);
            }
            __threadfence();
            __syncthreads();
            if(thread==0)*reinterpret_cast<volatile unsigned long long*>(&int8_c_ready[slot][tile])=int8_c_stamp;
        } else if(thread==0) {
            while(*reinterpret_cast<volatile unsigned long long*>(&int8_c_ready[slot][tile])!=int8_c_stamp){}
            __threadfence();
        }
        __syncthreads();
        return __ldcg(&int8_c_safe[slot][tile])?slot:-1;
    }

    template <int K, int C, int N>
    __device__ __forceinline__ void int8_UncachedTile(const unsigned char* input, const unsigned char* weights,
                                                float (&acc)[6][8], int tokenTile, int channelTile, int blocks16)
    {
        __shared__ __align__(16) unsigned As[8][96];
        __shared__ __align__(16) unsigned int8_Ahi[8][96];
        __shared__ __align__(16) unsigned Bs[8][128];
        __shared__ float int8_Sa[96],int8_Sb[128];
        const int lane = threadIdx.x, warp = threadIdx.y, thread = warp * 32 + lane;
        const int g = lane >> 2, t = lane & 3, wr = warp & 1, wc = warp >> 1;
#pragma unroll
        for (int i = 0; i < 6; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { acc[i][j] = 0.0f; }
        }
        for (int slice = 0; slice < K / 32; ++slice) {
            __syncthreads();

            if(thread<96) {
                const int block=(thread/48)*3+(thread%48)/16;
                const int row=(thread%16)/2+(thread&1)*8;
                const int globalBlock=tokenTile*6+block;
                unsigned code[8];int maximum=0;
#pragma unroll
                for(int kk=0;kk<8;++kk) {
                    int at=(row&7)*64+(kk&3)*16+(row>>3)*4+(kk>>2)*8;
                    code[kk]=globalBlock<blocks16?*reinterpret_cast<const unsigned*>(input+(size_t)globalBlock*16*C+slice*512+at):0;
#pragma unroll
                    for(int bb=0;bb<4;++bb)maximum=max(maximum,int((code[kk]>>(8*bb))&127));
                }
                const int exponent=max(maximum>>3,1);
                int8_Sa[thread]=__uint_as_float(unsigned(exponent+106)<<23);
#pragma unroll
                for(int kk=0;kk<8;++kk) {
                    unsigned packed=0, high=0;
#pragma unroll
                    for(int bb=0;bb<4;++bb) {
                        int fixed=int8_Fixed16(code[kk]>>(8*bb),exponent);
                        if(bb<2)packed|=(unsigned(fixed)&65535u)<<(16*bb);
                        else high|=(unsigned(fixed)&65535u)<<(16*(bb-2));
                    }
                    As[kk][thread]=packed;
                    int8_Ahi[kk][thread]=high;
                }
            }
            if(thread<128) {
                const int n=(thread/16)*16+((thread%16)/4)*2+((thread%4)/2)*8+(thread&1);
                unsigned code[8];int maximum=0;
#pragma unroll
                for(int kk=0;kk<8;++kk) {
                    int at=(n/16)*512+(n&7)*64+(kk&3)*16+((n>>3)&1)*8+(kk>>2)*4;
                    if(kk<4){uint2 pair=*reinterpret_cast<const uint2*>(weights+(size_t)slice*N*32+(size_t)channelTile*128*32+at);code[kk]=pair.x;code[kk+4]=pair.y;}
#pragma unroll
                    for(int bb=0;bb<4;++bb)maximum=max(maximum,int((code[kk]>>(8*bb))&127));
                }
                const int exponent=max(maximum>>3,1);
                int8_Sb[thread]=__uint_as_float(unsigned(exponent+114)<<23);
#pragma unroll
                for(int kk=0;kk<8;++kk) {
                    unsigned packed=0;
#pragma unroll
                    for(int bb=0;bb<4;++bb)packed|=(unsigned(int8_Fixed(code[kk]>>(8*bb),exponent))&255u)<<(8*bb);
                    Bs[kk][thread]=packed;
                }
            }
            __syncthreads();
#pragma unroll
            for(int phase=0;phase<6/3;++phase) {
                int dot[3][8]={};
#pragma unroll 1
                for(int kk=0;kk<8;++kk) {
                    unsigned av[3],ah[3],bv[8];
#pragma unroll
                    for(int i=0;i<3;++i) {
                        int r=phase*3+i;
                        av[i]=As[kk][wr*48+g*2+(r&1)+(r>>1)*16];
                            ah[i]=int8_Ahi[kk][wr*48+g*2+(r&1)+(r>>1)*16];
                    }
#pragma unroll
                    for(int j=0;j<8;++j)bv[j]=Bs[kk][wc*32+t*4+(j&3)+(j>>2)*16];
#pragma unroll
                    for(int i=0;i<3;++i) {
#pragma unroll
                        for(int j=0;j<8;++j){asm("dp2a.lo.s32.s32 %0,%1,%2,%3;":"=r"(dot[i][j]):"r"(av[i]),"r"(bv[j]),"r"(dot[i][j]));
                        asm("dp2a.hi.s32.s32 %0,%1,%2,%3;":"=r"(dot[i][j]):"r"(ah[i]),"r"(bv[j]),"r"(dot[i][j]));}
                    }
                }
#pragma unroll
                for(int i=0;i<3;++i) {
                    int r=phase*3+i;
                    const float sa=int8_Sa[wr*48+g*2+(r&1)+(r>>1)*16];
#pragma unroll
                    for(int j=0;j<8;++j)acc[r][j]=fmaf(float(dot[i][j]),sa*int8_Sb[wc*32+t*4+(j&3)+(j>>2)*16],acc[r][j]);
                }
            }
        }
    }


    template <int K, int C, int N>
    __device__ __forceinline__ bool int8_Conv1x1Tile(const unsigned char* input,const unsigned char* weights,
        float (&acc)[6][8],int tokenTile,int channelTile,int blocks16,int int8_slot) {
        __shared__ __align__(16) unsigned As[8][96],int8_Ahi[8][96],Bs[8][128];
        __shared__ float int8_Sa[96],int8_Sb[128];
        __shared__ int int8_exp[96];
        const int lane=threadIdx.x,warp=threadIdx.y,thread=warp*32+lane;
        const int g=lane>>2,t=lane&3,wr=warp&1,wc=warp>>1;
        const int int8_tile=blockIdx.z*8+channelTile;
        { constexpr int half=0;
            __syncthreads();
            __shared__ int int8_badinput;
            if(thread==0)int8_badinput=0;
            if(thread<128)int8_Sb[thread]=__ldcg(&int8_c_scale[int8_slot][int8_tile][thread]);
            __syncthreads();
            int dot[6][8]={};
            for(int ss=0;ss<32;++ss) {
                const int slice=half*16+ss;
                __syncthreads();
                if(thread<96) {
                    const int block=(thread/48)*3+(thread%48)/16;
                    const int row=(thread%16)/2+(thread&1)*8;
                    const int globalBlock=tokenTile*6+block; constexpr int exponent=12;
                    unsigned int8_bad=0;
#pragma unroll
                    for(int kk=0;kk<8;++kk) {
                        const int at=(row&7)*64+(kk&3)*16+(row>>3)*4+(kk>>2)*8;
                        const unsigned code=globalBlock<blocks16?*reinterpret_cast<const unsigned*>(input+(size_t)globalBlock*16*C+slice*512+at):0;
                        int8_bad|=((code&0x7f7f7f7fu)+0x18181818u)&0x80808080u;
                        unsigned lo=0,hi=0;
#pragma unroll
                        for(int bb=0;bb<4;++bb) {
                            const int fixed=int8_Fixed16(code>>(bb*8),exponent);
                            if(bb<2)lo|=(unsigned(fixed)&65535u)<<(16*bb);
                            else hi|=(unsigned(fixed)&65535u)<<(16*(bb-2));
                        }
                        As[kk][thread]=lo;int8_Ahi[kk][thread]=hi;
                    }
                    if(int8_bad)atomicExch(&int8_badinput,1);
                }
#pragma unroll
                for(int word=thread;word<1024;word+=256)Bs[word/128][word%128]=__ldcg(&int8_c_words[int8_slot][int8_tile][slice][word/128][word%128]);
                __syncthreads();
                if(int8_badinput)return false;
#pragma unroll 1
                for(int kk=0;kk<8;++kk) {
                    unsigned av[6],ah[6],bv[8];
#pragma unroll
                    for(int i=0;i<6;++i) {
                        int row=wr*48+g*2+(i&1)+(i>>1)*16;
                        av[i]=As[kk][row];ah[i]=int8_Ahi[kk][row];
                    }
#pragma unroll
                    for(int j=0;j<8;++j)bv[j]=Bs[kk][wc*32+t*4+(j&3)+(j>>2)*16];
#pragma unroll
                    for(int i=0;i<6;++i) {
#pragma unroll
                        for(int j=0;j<8;++j) {
                            asm("dp2a.lo.s32.s32 %0,%1,%2,%3;":"=r"(dot[i][j]):"r"(av[i]),"r"(bv[j]),"r"(dot[i][j]));
                            asm("dp2a.hi.s32.s32 %0,%1,%2,%3;":"=r"(dot[i][j]):"r"(ah[i]),"r"(bv[j]),"r"(dot[i][j]));
                        }
                    }
                }
            }
            // Producer verified max-column sum(abs(Wq)) <=69905; |Aq|<=30720.
            // Thus every partial dot and final dot is within INT32_MAX for all inputs.
#pragma unroll
            for(int i=0;i<6;++i) {
                constexpr float sa=0.001953125f;
#pragma unroll
                for(int j=0;j<8;++j)acc[i][j]=float(dot[i][j])*(sa*int8_Sb[wc*32+t*4+(j&3)+(j>>2)*16]);
            }
        }
        return true;
    }

    // Writes acc (after `epilogue`) as FP8 in NVIDIA's accumulator output layout.
    template <int N, class Epilogue>
    __device__ __forceinline__ void int8_Fp32StoreTile(const int8_Conv1x1Params& p, const float (&acc)[6][16], int tokenTile,
                                              int channelTile, int blocks16, Epilogue epilogue)
    {
        const int lane = threadIdx.x, warp = threadIdx.y, wr = warp & 1, wc = warp >> 1;
#pragma unroll
        for (int bi = 0; bi < 3; ++bi) {
            const int globalBlock = tokenTile * 6 + wr * 3 + bi;
            if (globalBlock >= blocks16) { continue; }
#pragma unroll
            for (int sl = 0; sl < 2; ++sl) {
                unsigned words[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {   // a0: row g idx 0-3, a1: row g+8 idx 0-3, a2/a3: idx 4-7
                    const int row = bi * 2 + (r & 1), base = sl * 8 + 4 * (r >> 1);
                    words[r] = int8_EncodeE4M3(epilogue(acc[row][base])) | (int8_EncodeE4M3(epilogue(acc[row][base + 1])) << 8) |
                               (int8_EncodeE4M3(epilogue(acc[row][base + 2])) << 16) | (int8_EncodeE4M3(epilogue(acc[row][base + 3])) << 24);
                }
                const int slice = channelTile * 4 + wc * 2 + sl;
                *reinterpret_cast<uint4*>(p.output + static_cast<size_t>(globalBlock) * 16 * N + slice * 512 + lane * 16) =
                    make_uint4(words[0], words[1], words[2], words[3]);
            }
        }
    }

    // Writes acc (after `epilogue`) as FP8 in NVIDIA's accumulator output layout.
    template <int N, class Epilogue>
    __device__ __forceinline__ void int8_StoreTile(const int8_Conv1x1Params& p, const float (&acc)[6][8], int tokenTile,
                                              int channelTile, int blocks16, Epilogue epilogue)
    {
        const int lane = threadIdx.x, warp = threadIdx.y, wr = warp & 1, wc = warp >> 1;
#pragma unroll
        for (int bi = 0; bi < 3; ++bi) {
            const int globalBlock = tokenTile * 6 + wr * 3 + bi;
            if (globalBlock >= blocks16) { continue; }
#pragma unroll
            for (int sl = 0; sl < 1; ++sl) {
                unsigned words[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {   // a0: row g idx 0-3, a1: row g+8 idx 0-3, a2/a3: idx 4-7
                    const int row = bi * 2 + (r & 1), base = sl * 8 + 4 * (r >> 1);
                    words[r] = int8_EncodeE4M3(epilogue(acc[row][base])) | (int8_EncodeE4M3(epilogue(acc[row][base + 1])) << 8) |
                               (int8_EncodeE4M3(epilogue(acc[row][base + 2])) << 16) | (int8_EncodeE4M3(epilogue(acc[row][base + 3])) << 24);
                }
                const int slice = channelTile * 4 + wc + sl;
                *reinterpret_cast<uint4*>(p.output + static_cast<size_t>(globalBlock) * 16 * N + slice * 512 + lane * 16) =
                    make_uint4(words[0], words[1], words[2], words[3]);
            }
        }
    }

    __device__ __forceinline__ void int8_WaitFlag(const int* flag)
    {
        while (*reinterpret_cast<const volatile int*>(flag) < 0) {}
    }

    template <bool Wait, bool Signal>
    __device__ __forceinline__ void int8_FfnExpand(const int8_Conv1x1Params& p)
    {
        constexpr int K = 1024, N = 4096;
        const int tokens = p.height * p.width;
        const int tiles = (tokens - 1) / 96 + 1;
        const int channelTile = blockIdx.x / tiles, tokenTile = blockIdx.x % tiles;
        const int blocks16 = (((tokens - 1) + 32) & ~31) / 16;
        if (Wait) {   // Producer: 1024 channels in 8 tiles of 128 per 128-token tile.
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                const int first=(tokenTile*96)>>7,last=min(((tokenTile*96+95)>>7)+1,(tokens+127)/128);
                for (int tile = first; tile < last; ++tile) {
                    for (int c = 0; c < K / 128; ++c) { int8_WaitFlag(p.waitFlags + tile * 8 + c); }
                }
                __threadfence();
            }
            __syncthreads();
        }
        float acc[6][16];
        int8_Fp32Tile<K, K, N>(p.input, p.weights, acc, tokenTile, channelTile, blocks16);
        int8_Fp32StoreTile<N>(p, acc, tokenTile, channelTile, blocks16, [](float y) { return int8_CubicSilu(int8_Half(y)); });
        if (Signal) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                for(int tile=(tokenTile*96)/128;tile<=min((tokenTile*96+95)/128,(tokens-1)/128);++tile) {
                    const int first=(tile*128)/96,last=(min(tokens,(tile+1)*128)-1)/96;
                    int* flag=p.signalFlags+tile*32+channelTile;
                    if(atomicSub(flag,1)==-(last-first+1)) {
                        __threadfence();
                        *reinterpret_cast<volatile int*>(flag)=0;
                    }
                }
            }
        }
    }

    struct __align__(8) int8_ContractParams
    {
        const unsigned char* input;     // +0, 4096 channels
        const unsigned char* residual;  // +8, 1024 channels, same layout as the output
        unsigned char* output;          // +16
        const unsigned char* weights;   // +24, then f16 gamma[1024] (residual scale)
        int* sliceFlags;                // +32, per tile: last K quarter (z) folded in
        __half* partial;                // +40, f16 running sum, one per output byte
        const int* waitFlags;           // +48, chained: ffn_expand tile flags
        int* signalFlags;               // +56, chained: this kernel's tile flags
        int height, width;              // +64
    };
    static_assert(sizeof(int8_ContractParams) == 72, "parameter block size");

    // Split K over blockIdx.z (quarters of 1024 input channels), folded in z order as NVIDIA
    // does: z=0 stores residual + its sum as f16, z=1 and 2 add theirs in f16, z=3 adds its
    // own and writes FP8. sliceFlags[tile] = z once quarter z is in.
    template <bool Wait, bool Signal>
    __device__ __forceinline__ void int8_FfnContract(const int8_ContractParams& p)
    {
        constexpr int K = 1024, C = 4096, N = 1024;
        const int tokens = p.height * p.width;
        const int tiles = (tokens - 1) / 96 + 1;
        const int channelTile = blockIdx.x / tiles, tokenTile = blockIdx.x % tiles;
        const int blocks16 = (((tokens - 1) + 32) & ~31) / 16;
        const int z = blockIdx.z, lane = threadIdx.x, warp = threadIdx.y, wr = warp & 1, wc = warp >> 1;
        const bool leader = threadIdx.x == 0 && threadIdx.y == 0;
        if (Wait) {   // NVIDIA waits on the first ffn_expand channel tile of this quarter; we
                      // read all 8 at once, so wait on each.
            if (leader) {
                for(int tile=(tokenTile*96)/128;tile<=min((tokenTile*96+95)/128,(tokens-1)/128);++tile) {
                    for(int c=0;c<8;++c)int8_WaitFlag(p.waitFlags+tile*32+z*8+c);
                }
                __threadfence();
            }
            __syncthreads();
        }
        const int int8_slot=int8_c_prepare<N>(p.weights,p.weights+(size_t)z*K*N+(size_t)channelTile*4096,z*8+channelTile,tokenTile);
        float acc[6][8];
        if(int8_slot<0)int8_UncachedTile<K, C, N >(p.input + z * (K / 32) * 512, p.weights + static_cast<size_t>(z) * K * N, acc,
                             tokenTile, channelTile, blocks16);
        else if(!int8_Conv1x1Tile<K, C, N >(p.input + z * (K / 32) * 512, p.weights + static_cast<size_t>(z) * K * N, acc,
                             tokenTile, channelTile, blocks16,int8_slot))
            int8_UncachedTile<K,C,N>(p.input+z*(K/32)*512,p.weights+(size_t)z*K*N,acc,tokenTile,channelTile,blocks16);
        const int flag = tokenTile * 8 + channelTile;
        if (z > 0) {
            if (leader) {
                while (*reinterpret_cast<const volatile int*>(p.sliceFlags + flag) < z - 1) {}
                __threadfence();
            }
            __syncthreads();
        }
#pragma unroll
        for (int bi = 0; bi < 3; ++bi) {
            const int globalBlock = tokenTile * 6 + wr * 3 + bi;
            if (globalBlock >= blocks16) { continue; }
#pragma unroll
            for (int sl = 0; sl < 1; ++sl) {
                const size_t base = static_cast<size_t>(globalBlock) * 16 * N + (channelTile * 4 + wc + sl) * 512 + lane * 16;
                float v[16];
#pragma unroll
                for (int q = 0; q < 16; ++q) {   // lane byte q: word q/4 as in int8_StoreTile
                    v[q] = acc[bi * 2 + ((q >> 2) & 1)][sl * 8 + 4 * (q >> 3) + (q & 3)];
                }
                uint4* partial = reinterpret_cast<uint4*>(p.partial + base);
                if (z == 0) {   // + residual * gamma, the f16 per-channel scale after the weights
                    const uint4 r = *reinterpret_cast<const uint4*>(p.residual + base);
                    const unsigned rw[4] = {r.x, r.y, r.z, r.w};
                    const unsigned* gamma = reinterpret_cast<const unsigned*>(p.weights + C * N) +
                                            (channelTile * 4 + wc + sl) * 16 + (lane & 3);
#pragma unroll
                    for (int q = 0; q < 16; ++q) {   // channel 16(q>>3) + 8((q>>1)&1) + 2t + (q&1) of the slice
                        const unsigned pair = __ldg(gamma + 8 * (q >> 3) + 4 * ((q >> 1) & 1));
                        const float g = __half2float(__ushort_as_half(static_cast<unsigned short>(pair >> (16 * (q & 1)))));
                        v[q] = int8_Half(v[q] + int8_Half(int8_E4M3(rw[q >> 2] >> (8 * (q & 3))) * g));
                    }
                } else {
                    const uint4 s0 = __ldcg(partial), s1 = __ldcg(partial + 1);
                    const unsigned sw[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        const __half h = __ushort_as_half(static_cast<unsigned short>(sw[q >> 1] >> (16 * (q & 1))));
                        v[q] = int8_Half(__half2float(h) + int8_Half(v[q]));
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
                        ow[i] = int8_EncodeE4M3(v[4 * i]) | (int8_EncodeE4M3(v[4 * i + 1]) << 8) |
                                (int8_EncodeE4M3(v[4 * i + 2]) << 16) | (int8_EncodeE4M3(v[4 * i + 3]) << 24);
                    }
                    *reinterpret_cast<uint4*>(p.output + base) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
                }
            }
        }
        __threadfence();
        __syncthreads();
        if (leader) {
            *reinterpret_cast<volatile int*>(p.sliceFlags + flag) = z;
            if(Signal&&z==3) {
                for(int tile=(tokenTile*96)/128;tile<=min((tokenTile*96+95)/128,(tokens-1)/128);++tile) {
                    const int first=(tile*128)/96,last=(min(tokens,(tile+1)*128)-1)/96;
                    int* flag=p.signalFlags+tile*8+channelTile;
                    if(atomicSub(flag,1)==-(last-first+1)) {
                        __threadfence();
                        *reinterpret_cast<volatile int*>(flag)=0;
                    }
                }
            }
        }
    }

    struct __align__(8) int8_QkvParams
    {
        const unsigned char* input;    // +0, 1024 channels
        unsigned char* q;              // +8, 32 heads of 32 channels, laid out as 1024 channels
        unsigned char* k;              // +16
        unsigned char* v;              // +24
        const unsigned char* weights;  // +32: f32 Q scale per head [32], then FP8 weights (N = 3072)
        int* pairFlags;                // +40, per (token tile, head pair); NVIDIA's split-K flags
        const __half* partial;         // +48, NVIDIA's split-K f16 scratch (unused here)
        const int* waitFlags;          // +56, chained: ffn_contract tile flags
        int* signalFlags;              // +64, chained: this kernel's (token tile, head pair) flags
        int height, width;             // +72
    };
    static_assert(sizeof(int8_QkvParams) == 80, "parameter block size");

    // CTA x: token tile x % tiles, head pair x / tiles. Weight columns 96 * head + [Q 32 |
    // K 32 | V 32]. Q and K are divided by their L2 norm over the head (clamped at the f16
    // eps NVIDIA uses), Q then scaled by sqrt(32) and its head's scale, all in f16 steps.
    // NVIDIA splits K over z and folds the halves in f16 through `partial`; here z picks the
    // head of the pair and each CTA reads all of K. The second CTA of a pair to finish signals.
    // Warp w: token blocks 2w, 2w+1 of the tile; lane 4g+t: rows g, g+8, stored positions
    // 4t..4t+3 and 16+4t.. of the Q, K and V slices.
    template <bool Wait, bool Signal>
    __device__ __forceinline__ void int8_Qkv(const int8_QkvParams& p)
    {
        constexpr int K = 1024, N = 3072, Cols = 96;
        __shared__ __align__(16) float As[32][96];
        __shared__ __align__(16) float Bs[32][Cols];
        const int tokens = p.height * p.width;
        const int tiles = (tokens - 1) / 96 + 1;
        const int pair = blockIdx.x / tiles, tokenTile = blockIdx.x % tiles;
        const int blocks16 = (((tokens - 1) + 32) & ~31) / 16;
        const int lane = threadIdx.x, warp = threadIdx.y, thread = warp * 32 + lane, g = lane >> 3, t = lane & 7;
        if (Wait) {
            if (thread == 0) {
                for(int tile=tokenTile*96/128;tile<=min((tokenTile*96+95)/128,(tokens-1)/128);++tile) {
                    for(int c=0;c<K/128;++c)int8_WaitFlag(p.waitFlags+tile*8+c);
                }
                __threadfence();
            }
            __syncthreads();
        }
        const float eps = int8_Half(__int_as_float(948045311)), root32 = int8_Half(sqrtf(32.0f));
        for (int h = blockIdx.z; h < 2; h += gridDim.z) {
            const int head = pair * 2 + h;
            const unsigned char* weights = p.weights + 128 + static_cast<size_t>(head) * Cols * 32;
            float acc[6][12];
#pragma unroll
            for (int i = 0; i < 6; ++i) {
#pragma unroll
                for (int j = 0; j < 12; ++j) { acc[i][j] = 0.0f; }
            }
            for (int slice = 0; slice < K / 32; ++slice) {
                __syncthreads();
                {   // A: 8 token blocks x 512 bytes; thread loads 32 bytes of block thread/16.
                    const int block = thread >> 4, part = thread & 15;
                    const int globalBlock = tokenTile * 6 + block;
                    uint4 words[2] = {};
                    if (block<6&&globalBlock < blocks16) {
                        const uint4* source = reinterpret_cast<const uint4*>(
                            p.input + static_cast<size_t>(globalBlock) * 16 * K + slice * 512 + part * 32);
                        words[0] = source[0]; words[1] = source[1];
                    }
                    const unsigned* w = reinterpret_cast<const unsigned*>(words);
#pragma unroll
                    for (int word = 0; word < 8; ++word) {
                        const int offset = part * 32 + word * 4;
                        const int fragLane = offset >> 4, reg = (offset >> 2) & 3;
                        const int row = (fragLane >> 2) + 8 * (reg & 1);
                        const int kBase = 4 * (fragLane & 3) + 16 * (reg >> 1);
                        const int slot = ((block*16+row)/24)*24+((block*16+row)%4)*2+(((block*16+row)%24)/4%2)+(((block*16+row)%24)/8)*8;
#pragma unroll
                        for (int byte = 0; byte < 4; ++byte) { if(block<6)As[kBase + byte][slot] = int8_E4M3(w[word] >> (8 * byte)); }
                    }
                }
                if (thread < Cols) {   // B: 96 columns x 32 k = 3072 contiguous bytes.
                    const uint4* source = reinterpret_cast<const uint4*>(weights + static_cast<size_t>(slice) * N * 32 + thread * 32);
                    const uint4 words[2] = {source[0], source[1]};
                    const unsigned* w = reinterpret_cast<const unsigned*>(words);
#pragma unroll
                    for (int word = 0; word < 8; ++word) {
                        const int offset = thread * 32 + word * 4;
                        const int chunk = offset >> 9, fragLane = (offset >> 4) & 31, q = (offset >> 2) & 3;
                        const int n = chunk * 16 + 8 * (q >> 1) + (fragLane >> 2), kBase = 4 * (fragLane & 3) + 16 * (q & 1);
                        const int c = n & 31, rest = c & 15;
                        const int slot = (n>>5)*32+((c>>4)*4+((c>>1)&3))*4+((c>>3)&1)*2+(c&1);
#pragma unroll
                        for (int byte = 0; byte < 4; ++byte) { Bs[kBase + byte][slot] = int8_E4M3(w[word] >> (8 * byte)); }
                    }
                }
                __syncthreads();
                const float2* aRow = reinterpret_cast<const float2*>(&As[0][warp*24+g*2]);
                const float4* bRow = reinterpret_cast<const float4*>(&Bs[0][t*4]);
#pragma unroll 4
                for (int k = 0; k < 32; ++k) {
                    const float2 a0=aRow[k*48],a1=aRow[k*48+4],a2=aRow[k*48+8];
                    const float a[6]={a0.x,a0.y,a1.x,a1.y,a2.x,a2.y};
                    float b[12];
#pragma unroll
                    for (int j = 0; j < 3; ++j) {
                        const float4 b4 = bRow[k*24+j*8];
                        b[4 * j] = b4.x; b[4 * j + 1] = b4.y; b[4 * j + 2] = b4.z; b[4 * j + 3] = b4.w;
                    }
#pragma unroll
                    for (int i = 0; i < 6; ++i) {
#pragma unroll
                        for (int j = 0; j < 12; ++j) { acc[i][j] = fmaf(a[i], b[j], acc[i][j]); }
                    }
                }
            }
            const float scale=int8_Half(reinterpret_cast<const float*>(p.weights)[head]);
#pragma unroll
            for(int i=0;i<6;++i) {
#pragma unroll
                for(int j=0;j<12;++j)acc[i][j]=int8_Half(acc[i][j]);
#pragma unroll
                for(int comp=0;comp<2;++comp) {
                    float squares[4];
#pragma unroll
                    for(int j=0;j<4;++j)squares[j]=int8_Half(acc[i][comp*4+j]*acc[i][comp*4+j]);
                    float sum=0;
#pragma unroll
                    for(int j=0;j<4;++j)sum+=__shfl_sync(0xFFFFFFFFu,squares[j],(lane&~7)|(t&3));
#pragma unroll
                    for(int j=0;j<4;++j)sum+=__shfl_sync(0xFFFFFFFFu,squares[j],(lane&~7)|(t&3)|4);
                    sum+=__shfl_xor_sync(0xFFFFFFFFu,sum,2);
                    sum+=__shfl_xor_sync(0xFFFFFFFFu,sum,1);
                    float r=int8_Half(rsqrtf(fmaxf(int8_Half(sum),eps)));
#pragma unroll
                    for(int j=0;j<4;++j) {
                        float v=int8_Half(acc[i][comp*4+j]*r);
                        acc[i][comp*4+j]=comp==0?int8_Half(int8_Half(v*root32)*scale):v;
                    }
                }
            }
#pragma unroll
            for(int i=0;i<6;++i) {
                int token=warp*24+g+4*i;
                int block=tokenTile*6+token/16;
                if(block>=blocks16)continue;
                int row=token&15,ol=(row&7)*4+(t&3);
#pragma unroll
                for(int comp=0;comp<2;++comp) {
                    unsigned bits=0;
#pragma unroll
                    for(int j=0;j<4;++j)bits|=int8_EncodeE4M3(acc[i][comp*4+j])<<(8*j);
                    int reg=comp==0?2*(t>>2)+(row>>3):2*(row>>3)+(t>>2);
                    unsigned char* output=comp==0?p.q:p.k;
                    *reinterpret_cast<unsigned*>(output+(size_t)block*16384+head*512+ol*16+reg*4)=bits;
                }
            }
            // V^T: per 32-token slice, 32 heads x 1024 bytes; per head two 512-byte chunks of 16
            // dims in the weights' B layout, tokens as k (in stored-position order): lane 4g+t
            // byte q holds dim 8(q>>3) + g, slice position 4t + 16((q>>2)&1) + (q&3).
            // Staged as f32 in As[dim][token] and encoded while gathering.
            __syncthreads();
#pragma unroll
            for(int i=0;i<6;++i) {
                int token=warp*24+g+4*i;
#pragma unroll
                for(int j=0;j<4;++j) {
                    int dim=16*(t>>2)+2*(t&3)+(j&1)+8*(j>>1);
                    As[dim][token]=acc[i][8+j];
                }
            }
            __syncthreads();
            // Chunk word r of lane 4g+t: dim 16*half + 8(r>>1) + g, slice positions 4t + 16(r&1)
            // + 0..3, i.e. tokens 16(r&1) + 2t + {0, 1, 8, 9}. Chunks thread and thread + 128:
            // with chunks 2*thread + c, ptxas -O3 (CUDA 12.9, sm_61 and sm_89) built nvcc's sm_61
            // PTX so that a quarter of V came out wrong; -O0 and sm_89 PTX were right.
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int chunk = thread + 128 * c, s = chunk >> 6, half = (chunk >> 5) & 1, l = chunk & 31;
                if(chunk>=192)continue;
                const int slice = tokenTile * 3 + s;
                if (slice * 2 >= blocks16) { continue; }
                unsigned words[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float* row = &As[16 * half + 8 * (r >> 1) + (l >> 2)][s * 32 + 16 * (r & 1) + 2 * (l & 3)];
                    words[r] = int8_EncodeE4M3(row[0]) | (int8_EncodeE4M3(row[1]) << 8) | (int8_EncodeE4M3(row[8]) << 16) | (int8_EncodeE4M3(row[9]) << 24);
                }
                *reinterpret_cast<uint4*>(p.v + static_cast<size_t>(slice) * 32768 + head * 1024 + half * 512 + l * 16) =
                    make_uint4(words[0], words[1], words[2], words[3]);
            }
        }
        __threadfence();
        __syncthreads();
        if(thread==0) {
            for(int tile=tokenTile*96/128;tile<=min((tokenTile*96+95)/128,(tokens-1)/128);++tile) {
                const int first=tile*128/96,last=(min(tokens,(tile+1)*128)-1)/96;
                const int expected=(last-first+1)*2,index=tile*16+pair;
                if(atomicSub(p.pairFlags+index,1)==-expected) {
                    __threadfence();
                    *reinterpret_cast<volatile int*>(p.pairFlags+index)=1;
                    if(Signal&&p.signalFlags)*reinterpret_cast<volatile int*>(p.signalFlags+index)=0;
                }
            }
        }
    }

}

// Builds can leave kernels to NVIDIA with -DTRP_SKIP_EXPAND, -DTRP_SKIP_CONTRACT, -DTRP_SKIP_QKV.
#ifndef TRP_SKIP_EXPAND
extern "C" __global__ void __launch_bounds__(128,3) cc_vit_1d_ffn_expand_fp8(int8_Conv1x1Params p) { int8_FfnExpand<false, false>(p); }
extern "C" __global__ void __launch_bounds__(128,3) cc_vit_1d_ffn_expand_publish_fp8(int8_Conv1x1Params p) { int8_FfnExpand<false, true>(p); }
extern "C" __global__ void __launch_bounds__(128,3) cc_vit_1d_ffn_expand_chained_fp8(int8_Conv1x1Params p) { int8_FfnExpand<true, true>(p); }
#endif
#ifndef TRP_SKIP_CONTRACT
extern "C" __global__ void __launch_bounds__(256,2) cc_vit_1d_ffn_contract_fp8(int8_ContractParams p) { int8_FfnContract<false, false>(p); }
extern "C" __global__ void __launch_bounds__(256,2) cc_vit_1d_ffn_contract_chained_fp8(int8_ContractParams p) { int8_FfnContract<true, true>(p); }
#endif
#ifndef TRP_SKIP_QKV
extern "C" __global__ void __launch_bounds__(128,3) cc_vit_1d_qkv_fp8(int8_QkvParams p) { int8_Qkv<false, false>(p); }
extern "C" __global__ void __launch_bounds__(128,3) cc_vit_1d_qkv_chained_fp8(int8_QkvParams p) { int8_Qkv<true, true>(p); }
#endif

