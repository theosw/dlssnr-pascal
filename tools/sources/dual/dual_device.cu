#include "dual_core.cuh"

extern "C" __device__ __noinline__ void prepost_core_pre(const unsigned char* blob,__half* state) {
using namespace prepost_fourwarp;
const int tok=(threadIdx.x&127)>>1,ch=16*((threadIdx.x&127)&1);
__shared__ Scratch prepost_scratch;Scratch& scratch=prepost_scratch;
float out[16];Core4<false>(state,blob,scratch,out,1024,nullptr,9232);

}

#include "dual_pool.cuh"
