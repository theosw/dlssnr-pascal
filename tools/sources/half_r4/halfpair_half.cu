#include "halfpair_of_views.cuh"
#include "halfpair_nf_views.cuh"
namespace swin1h_native_swin1h_r6_of {
__device__ __noinline__ void StageUpScalars(const unsigned char* input,const unsigned char* skipInput,const unsigned char* blob,
    int height,int width,int viewH,int viewW,Scratch& scratch,int X,int Y) {
    const int ch=16*((threadIdx.x&127)&1),tok=(threadIdx.x&127)>>1,H=height/2,W=width/2;
    const Packed a=Planar(input,H,W,X/2,Y/2);
    const Packed b=Planar(input+size_t(H)*W*32,H,W,X/2,Y/2);
    float up[16];Weights(blob,8192,scratch);Product(a,scratch,up);
#pragma unroll
    for(int n=0;n<16;++n)up[n]=Half(up[n]);
    float tail[16];Weights(blob,9216,scratch);Product(b,scratch,tail);
    const Packed skip=Blocked(skipInput,viewH>0?viewH:height,viewW>0?viewW:width,X,Y);
#pragma unroll
    for(int j=0;j<8;++j) {
        const unsigned sc=__ldg(reinterpret_cast<const unsigned*>(blob+10336)+8*((threadIdx.x&127)&1)+j);
        up[2*j]=Half(Half(up[2*j]+tail[2*j])+Half(Read(skip,ch+2*j)*LowHalf(sc)));
        up[2*j+1]=Half(Half(up[2*j+1]+tail[2*j+1])+Half(Read(skip,ch+2*j+1)*HighHalf(sc)));
    }
#pragma unroll
    for(int d=0;d<16;++d)(scratch.kv+4096*(threadIdx.x>>7))[(ch+d)*64+tok]=__half_as_ushort(__float2half_rn(up[d]));
}
__device__ __noinline__ void swin1h_r6_StageUpCoarse(const unsigned char* input,const unsigned char* skipInput,const unsigned char* blob,
    int height,int width,int viewH,int viewW,Scratch& scratch,int wx,int wy) {
    const int tok=(threadIdx.x&127)>>3,c0=4*((threadIdx.x&127)&7),H=height/2,W=width/2;
    const int X=wx/2+(tok&3),Y=wy/2+(tok>>2);
    float* xs=(scratch.decoded+2048*(threadIdx.x>>7));
    float lo[4],acc[4];
#pragma unroll 1
    for(int plane=0;plane<2;plane++) {
        __syncthreads();
#pragma unroll
        for(int j=0;j<4;j++) {
            const int c=c0+j,cn=c&15,stored=4*((cn&7)>>1)+(cn&1)+2*((cn>>3)&1);
            unsigned v=0;
            if(X>=0&&X<W&&Y>=0&&Y<H)v=__ldg(input+((size_t(2*plane+c/16)*H+Y)*W+X)*16+stored);
            xs[c*16+tok]=E4M3(v);
        }
        Weights(blob,8192+1024*plane,scratch);
#pragma unroll
        for(int j=0;j<4;j++)acc[j]=0;
#pragma unroll
        for(int k=0;k<32;k++) {
            const float a=xs[k*16+tok];
#pragma unroll
            for(int j=0;j<4;j++)acc[j]=fmaf(a,scratch.weights[k][c0+j],acc[j]);
        }
        if(plane==0) {
#pragma unroll
            for(int j=0;j<4;j++)lo[j]=Half(acc[j]);
        }
    }
#pragma unroll
    for(int j=0;j<4;j++)(scratch.kv+4096*(threadIdx.x>>7))[2048+(c0+j)*16+tok]=__half_as_ushort(__float2half_rn(lo[j]+acc[j]));
    __syncthreads();
    const int fine=(threadIdx.x&127)>>1,ch=16*((threadIdx.x&127)&1);
    int x,y;Pixel(fine,x,y);const int coarse=(x>>1)+4*(y>>1);
    const Packed skip=Blocked(skipInput,viewH>0?viewH:height,viewW>0?viewW:width,wx+x,wy+y);
#pragma unroll
    for(int j=0;j<16;j++) {
        const float up=__half2float(__ushort_as_half((scratch.kv+4096*(threadIdx.x>>7))[2048+(ch+j)*16+coarse]));
        const float sc=__half2float(reinterpret_cast<const __half*>(blob+10336)[ch+j]);
        (scratch.kv+4096*(threadIdx.x>>7))[(ch+j)*64+fine]=__half_as_ushort(__float2half_rn(up+Half(Read(skip,ch+j)*sc)));
    }
}

}
extern "C" __global__ void __launch_bounds__(256) cc_tinlayout_fused_swin_1h_32_1_inpview_tilesync_fp8(swin1h_native_swin1h_r6_of::Views p) {
    if(2*blockIdx.x>=gridDim.x)return;
    __shared__ swin1h_native_swin1h_r6_of::Scratch swin1h_s;
    const int H=p.auxH>0?p.auxH:p.H,W=p.auxW>0?p.auxW:p.W,wx=p.x0+8*(2*blockIdx.x+(threadIdx.x>>7)),wy=p.y0+8*blockIdx.y;
    int x,y;swin1h_native_swin1h_r6_of::Pixel((threadIdx.x&127)>>1,x,y);
    swin1h_native_swin1h_r6_of::swin1h_native_swin1h_r6<false>(p.input,H,W,wx+x,wy+y,swin1h_s);
    const swin1h_native_swin1h_r6_of::Packed in={};
    float out[16];swin1h_native_swin1h_r6_of::Core4<false,true>(in,p.blob,swin1h_s,out);swin1h_native_swin1h_r6_of::StoreBlocked(p.output,H,W,wx,wy,out);swin1h_native_swin1h_r6_of::Signal(p);
}
extern "C" __global__ void __launch_bounds__(256) cc_tinlayout_fused_swin_1h_32_1_ds_wait_fp8(swin1h_native_swin1h_r6_nf::Views p) {
    if(2*blockIdx.x>=gridDim.x)return;
    __shared__ swin1h_native_swin1h_r6_nf::Scratch swin1h_s;
    const int wx=p.x0+8*(2*blockIdx.x+(threadIdx.x>>7)),wy=p.y0+8*blockIdx.y;swin1h_native_swin1h_r6_nf::Wait(p,wx,wy);
    int x,y;swin1h_native_swin1h_r6_nf::Pixel((threadIdx.x&127)>>1,x,y);
    const swin1h_native_swin1h_r6_nf::Packed in=swin1h_native_swin1h_r6_nf::Blocked(p.input,p.H,p.W,wx+x,wy+y);
    float out[16];swin1h_native_swin1h_r6_nf::Core4(in,p.blob,swin1h_s,out);swin1h_native_swin1h_r6_nf::StoreBlocked(p.output,p.H,p.W,wx,wy,out);
    swin1h_native_swin1h_r6_nf::Downsample(p,swin1h_s,wx,wy,out);
}
extern "C" __global__ void __launch_bounds__(256) cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8(swin1h_native_swin1h_r6_of::Views p) {
    if(2*blockIdx.x>=gridDim.x)return;
    __shared__ swin1h_native_swin1h_r6_of::Scratch swin1h_s;
    const int wx=p.x0+8*(2*blockIdx.x+(threadIdx.x>>7)),wy=p.y0+8*blockIdx.y;int x,y;swin1h_native_swin1h_r6_of::Pixel((threadIdx.x&127)>>1,x,y);
    if(p.x0+16*int(blockIdx.x)>=0&&wy>=0)swin1h_native_swin1h_r6_of::swin1h_r6_StageUpCoarse(p.input,p.skip,p.blob,p.H,p.W,p.viewH,p.viewW,swin1h_s,wx,wy);
    else swin1h_native_swin1h_r6_of::StageUpScalars(p.input,p.skip,p.blob,p.H,p.W,p.viewH,p.viewW,swin1h_s,wx+x,wy+y);__syncthreads();
    float up[16];const int tok=(threadIdx.x&127)>>1,ch=16*((threadIdx.x&127)&1);
#pragma unroll
    for(int d=0;d<16;++d)up[d]=__half2float(__ushort_as_half((swin1h_s.kv+4096*(threadIdx.x>>7))[(ch+d)*64+tok]));
    swin1h_native_swin1h_r6_of::Packed in;swin1h_native_swin1h_r6_of::MergePack(up,in);
    float out[16];swin1h_native_swin1h_r6_of::Core4(in,p.blob,swin1h_s,out,2112,up);swin1h_native_swin1h_r6_of::StoreBlocked(p.output,p.H,p.W,wx,wy,out);swin1h_native_swin1h_r6_of::Signal(p);
}
extern "C" __global__ void __launch_bounds__(256) cc_tinlayout_fused_swin_1h_32_1_outview_wait_fp8(swin1h_native_swin1h_r6_of::Views p) {
    if(2*blockIdx.x>=gridDim.x)return;
    __shared__ swin1h_native_swin1h_r6_of::Scratch swin1h_s;
    const int wx=p.x0+8*(2*blockIdx.x+(threadIdx.x>>7)),wy=p.y0+8*blockIdx.y;swin1h_native_swin1h_r6_of::Wait(p,wx,wy);
    int x,y;swin1h_native_swin1h_r6_of::Pixel((threadIdx.x&127)>>1,x,y);
    const int H=p.auxH>0?p.auxH:p.H,W=p.auxW>0?p.auxW:p.W;
    const swin1h_native_swin1h_r6_of::Packed in=swin1h_native_swin1h_r6_of::Blocked(p.input,H,W,wx+x,wy+y);
    float out[16];swin1h_native_swin1h_r6_of::Core4(in,p.blob,swin1h_s,out);
    swin1h_native_swin1h_r6_of::StorePlanar(p.output,p.viewH>0?p.viewH:p.H,p.viewW>0?p.viewW:p.W,wx+x,wy+y,out);
}

