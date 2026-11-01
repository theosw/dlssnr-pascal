#include <cuda_fp16.h>
#ifndef SWINMH_HEADS
#define SWINMH_HEADS 2
#endif
#ifndef SWINMH_ROUND
#define SWINMH_ROUND 1
#endif
#ifndef SWINMH_VARIANT
#define SWINMH_VARIANT 0
#endif
#if SWINMH_VARIANT==0
#define swinmh_scratch swinmh_scratch_inp
#define swinmh_pool_used swinmh_pool_used_inp
#define swinmh_pool_owners swinmh_pool_owners_inp
#define swinmh_pool_next swinmh_pool_next_inp
#define swinmh_clocks swinmh_clocks_inp
#define SWINMH_SUFFIX inpview_tilesync_fp8
#elif SWINMH_VARIANT==1
#define swinmh_scratch swinmh_scratch_out
#define swinmh_pool_used swinmh_pool_used_out
#define swinmh_pool_owners swinmh_pool_owners_out
#define swinmh_pool_next swinmh_pool_next_out
#define swinmh_clocks swinmh_clocks_out
#define SWINMH_SUFFIX outview_wait_fp8
#elif SWINMH_VARIANT==2
#define swinmh_scratch swinmh_scratch_ds
#define swinmh_pool_used swinmh_pool_used_ds
#define swinmh_pool_owners swinmh_pool_owners_ds
#define swinmh_pool_next swinmh_pool_next_ds
#define swinmh_clocks swinmh_clocks_ds
#define SWINMH_SUFFIX ds_wait_fp8
#else
#define swinmh_scratch swinmh_scratch_up
#define swinmh_pool_used swinmh_pool_used_up
#define swinmh_pool_owners swinmh_pool_owners_up
#define swinmh_pool_next swinmh_pool_next_up
#define swinmh_clocks swinmh_clocks_up
#define SWINMH_SUFFIX upsample_tilesync_fp8
#endif
#define SWINMH_JOIN_(a,b) a##b
#define SWINMH_JOIN(a,b) SWINMH_JOIN_(a,b)
__device__ __forceinline__ float swinmh_decode(unsigned v) { return __uint_as_float(((v&128)<<24)|((v&127)<<20))*1.329227995784916e36f; }
__device__ __forceinline__ unsigned swinmh_encode(float x) {
    unsigned s=(__float_as_uint(x)>>24)&128;float a=fminf(fabsf(x),448.f);
    unsigned b=__float_as_uint(a);return s|(a<.015625f?__float2int_rn(a*512.f):((b+0x7ffff+((b>>20)&1))>>20)-(120<<3));
}
__device__ __forceinline__ float swinmh_half(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float swinmh_round(float x) { return SWINMH_ROUND?swinmh_half(x):x; }
__device__ __forceinline__ int swinmh_pos(int c) { return (c&~15)+4*((c&7)>>1)+(c&1)+2*((c>>3)&1); }
__device__ __forceinline__ int swinmh_bank(int k) { return ((k&6)<<2)|((k&16)>>2); }
__device__ __forceinline__ int swinmh_col(int t,int j) { return 16*(j>>2)+2*t+(j&1)+8*((j>>1)&1); }
__device__ __forceinline__ int swinmh_tok(int g,int i) { return 16*(i>>1)+g+8*(i&1); }
__device__ __forceinline__ int swinmh_woff(int k,int n,int N) {
    int p=swinmh_pos(k),l=4*(n&7)+((p&15)>>2),q=2*((n>>3)&1)+(p>>4&1);
    return (k>>5)*N*32+(n>>6)*2048+((n>>4)&3)*512+l*16+q*4+(p&3);
}
__device__ __forceinline__ float swinmh_silu(float x) {
    float c=fminf(fmaxf(x,-4.f),4.f);
    float p=swinmh_half(fmaf(-.055908203125f,fabsf(c),.447265625f));
    return swinmh_half(x*swinmh_half(fmaf(c,p,.89453125f)));
}
struct __align__(8) swinmh_params {
    const unsigned char* input;unsigned char* output;const unsigned char* weights;const unsigned char* residual;
    int height,width,x0,y0;const int* waitFlags;int ox,oy;int* signalFlags;unsigned char* ds;int dh,dw;
};
static_assert(sizeof(swinmh_params)==88,"params");
constexpr int swinmh_H=SWINMH_HEADS,swinmh_C=32*swinmh_H;
// Slots are acquired only by resident CTAs and released before exit.
// No window count or SM-count bound is required.
constexpr int swinmh_windows=19*(swinmh_H==2?8:swinmh_H==4?4:2);
__device__ unsigned swinmh_pool_used[swinmh_windows];
__device__ unsigned swinmh_pool_owners[19];

__device__ unsigned char swinmh_scratch[swinmh_windows*64*swinmh_C*4];
__device__ __forceinline__ int swinmh_blockoff(int x,int y,int c,int W,int C) {
    int r=(y&3)*4+(x&3),lane=(r&7)*4+((c&7)>>1),word=(r>>3)+2*((c>>4)&1),byte=(c&1)+2*((c>>3)&1);
    return ((y/4)*(W/4)+x/4)*16*C+(c/32)*512+lane*16+word*4+byte;
}
__device__ __forceinline__ unsigned swinmh_read(const unsigned char* p,int x,int y,int c,int H,int W,bool blocked) {
    if(x<0||x>=W||y<0||y>=H)return 0;
    return p[blocked?swinmh_blockoff(x,y,c,W,swinmh_C):((c/16*H+y)*W+x)*16+(swinmh_pos(c)&15)];
}

#include "swinmh_r7_qkv.cuh"
// Each warp: 64 tokens x 32 channels, eight rows and eight columns per lane.
template<int Kind,int Rows=8>
__device__ __forceinline__ void swinmh_gemm(const unsigned char* x,const unsigned char* w,float* as,float* bs,float (&a)[Rows][8],int b=0,const __half* residual=nullptr,const __half* scale=nullptr,bool cached=false) {
    constexpr int C=swinmh_C,H=swinmh_H,KC=Kind==6?2*C:C,KH=Kind==6?2*H:H;
    int l=threadIdx.x,h=threadIdx.y,t=l&3,g=l>>2,tid=h*32+l;
    #pragma unroll
    for(int i=0;i<Rows;i++)for(int j=0;j<8;j+=2) {
        float2 v=make_float2(0,0);
        if(Kind==3||Kind==4) {
            int c=h*32+swinmh_col(t,j),off=swinmh_tok(g,i)*C+c;
            if(Kind==3)v=__half22float2(*reinterpret_cast<const __half2*>(residual+off));
            else {
                unsigned p=*reinterpret_cast<const unsigned short*>(reinterpret_cast<const unsigned char*>(residual)+off);
                float2 sc=__half22float2(*reinterpret_cast<const __half2*>(scale+c));
                v.x=swinmh_half(swinmh_decode(p)*sc.x);v.y=swinmh_half(swinmh_decode(p>>8)*sc.y);
            }
        }
        a[i][j]=v.x;a[i][j+1]=v.y;
    }
    for(int sl=0;sl<KH;sl++) {
        __syncthreads();
        for(int z=tid;z<8*(Rows*8);z+=32*H) { int r=z%(Rows*8),kp=z/(Rows*8);
            unsigned word=*reinterpret_cast<const unsigned*>(x+r*KC+sl*32+kp*4);
            #pragma unroll
            for(int byte=0;byte<4;byte++)as[(kp*4+byte)*(Rows*8)+(r&7)*Rows+(r>>3)]=swinmh_decode(word>>(8*byte));
        }
        int base;
        if(Kind==0)base=h*C*128+sl*4096+b*1024;
        else if(Kind==1||Kind==3||Kind==4||Kind==6)base=sl*C*32+h*1024;
        else if(Kind==5)base=sl*2*C*32+(h+b*H)*1024;
        else base=sl*3*C*32+(h*3+b)*1024;
        if(Kind==2&&cached) {
            for(int z=l;z<256;z+=32) {
                uint2 v=swinmh_r7_cg2(swinmh_r7_data+base/2+2*z);
                float4 f=make_float4(__uint_as_float(v.x<<16),__uint_as_float(v.x&0xffff0000u),__uint_as_float(v.y<<16),__uint_as_float(v.y&0xffff0000u));
                *reinterpret_cast<float4*>(bs+(z/8)*C+h*32+(z&7)*4)=f;
            }
        } else {
        for(int word=l;word<256;word+=32) {
            unsigned v=__ldg(reinterpret_cast<const unsigned*>(w+base)+word);
            int o=4*word,j=o>>9,lane=(o>>4)&31,q=(o>>2)&3;
            int n=8*(2*j+(q>>1))+(lane>>2),k0=4*(lane&3)+16*(q&1);
            int nt=(n&7)>>1,nj=4*(n>>4)+2*((n>>3)&1)+(n&1);
            #pragma unroll
            for(int byte=0;byte<4;byte++) {
                int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                bs[k*C+h*32+((nt*8+nj)^swinmh_bank(k))]=swinmh_decode(v>>(8*byte));
            }
        }
        }
        __syncthreads();
        #pragma unroll 4
        for(int k=0;k<32;k++) {
            const float4* ap=reinterpret_cast<const float4*>(as+k*(Rows*8)+g*Rows);
            const float4* bp=reinterpret_cast<const float4*>(bs+k*C+h*32+((t*8)^swinmh_bank(k)));
            const float4* bp1=reinterpret_cast<const float4*>(bs+k*C+h*32+((t*8+4)^swinmh_bank(k)));
            float4 b0=bp[0],b1=bp1[0];
            float av[Rows];
            if constexpr(Rows==2) {
                float2 v=*reinterpret_cast<const float2*>(as+k*16+g*2);
                av[0]=v.x;av[1]=v.y;
            } else {
                float4 v0=ap[0],v1=ap[1];
                float values[8]={v0.x,v0.y,v0.z,v0.w,v1.x,v1.y,v1.z,v1.w};
                #pragma unroll
                for(int i=0;i<Rows;i++)av[i]=values[i];
            }
            float bv[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
            #pragma unroll
            for(int i=0;i<Rows;i++) {
                #pragma unroll
                for(int j=0;j<8;j++)a[i][j]=fmaf(av[i],bv[j],a[i][j]);
            }
        }
        #pragma unroll
        for(int i=0;i<Rows;i++)for(int j=0;j<8;j++)a[i][j]=swinmh_round(a[i][j]);
    }
}
template<int Rows>
__device__ __forceinline__ void swinmh_down(float (&up)[Rows][8],float (&mid)[Rows][8],const unsigned char* w,float* bs,int b) {
    constexpr int C=swinmh_C;
    int l=threadIdx.x,h=threadIdx.y,t=l&3;
    #pragma unroll
    for(int i=0;i<Rows;i++)for(int j=0;j<8;j++)up[i][j]=swinmh_decode(swinmh_encode(swinmh_silu(up[i][j])));
    __syncthreads();
    for(int word=l;word<256;word+=32) {
        unsigned v=__ldg(reinterpret_cast<const unsigned*>(w+h*4096+b*1024)+word);
        int o=4*word,j=o>>9,lane=(o>>4)&31,q=(o>>2)&3;
        int n=8*(2*j+(q>>1))+(lane>>2),k0=4*(lane&3)+16*(q&1);
        int nt=(n&7)>>1,nj=4*(n>>4)+2*((n>>3)&1)+(n&1);
        #pragma unroll
        for(int byte=0;byte<4;byte++) {
            int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
            bs[k*C+h*32+((nt*8+nj)^swinmh_bank(k))]=swinmh_decode(v>>(8*byte));
        }
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<32;k++) {
        constexpr unsigned mask=0xffffffff;
        const int src=(l&~3)+((k&7)>>1),j0=4*(k>>4)+2*((k>>3)&1)+(k&1);
        float4 b0=*reinterpret_cast<float4*>(bs+k*C+h*32+((t*8)^swinmh_bank(k))),b1=*reinterpret_cast<float4*>(bs+k*C+h*32+((t*8+4)^swinmh_bank(k)));
        float bv[8]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w};
        #pragma unroll
        for(int i=0;i<Rows;i++) {
            float av=__shfl_sync(mask,up[i][j0],src);
            #pragma unroll
            for(int j=0;j<8;j++)mid[i][j]=fmaf(av,bv[j],mid[i][j]);
        }
    }
    #pragma unroll
    for(int i=0;i<Rows;i++)for(int j=0;j<8;j++)mid[i][j]=swinmh_round(mid[i][j]);
}
#include "ffn_r11_acache.cuh"
__device__ __forceinline__ unsigned swinmh_expbits(float s) {
    float z=swinmh_half(fmaf(s,swinmh_half(__int_as_float(1027077105)),swinmh_half(__int_as_float(1067877303))));
    z=fminf(fmaxf(z,1.03125f),1.5693359375f);
    return (unsigned(__half_as_ushort(__float2half_rn(z)))-0x3c00)<<5;
}
__device__ __forceinline__ void swinmh_attention(unsigned char* qkv,unsigned char* out,unsigned short* kv,const unsigned char* w) {
    constexpr int C=swinmh_C;int l=threadIdx.x,h=threadIdx.y;
    const int tid=h*32+l;
    for(int pass=0;pass<2;pass++) {
        for(int z=tid;z<(swinmh_H/2)*1024;z+=32*swinmh_H) {
            int lh=z/1024,v=z%1024,part=v/512,at=v%512,key=at%64,cp=at/64,gh=pass*(swinmh_H/2)+lh;
            unsigned word=*reinterpret_cast<const unsigned*>(qkv+(part+1)*64*C+key*C+gh*32+cp*4);
            #pragma unroll
            for(int byte=0;byte<4;byte++) {
                unsigned bf=__float_as_uint(swinmh_decode(word>>(8*byte)))>>16;
                unsigned other=__shfl_xor_sync(0xffffffff,bf,1);
                if((l&1)==0)*reinterpret_cast<unsigned*>(kv+lh*4096+part*2048+(cp*4+byte)*64+key)=bf|(other<<16);
            }
        }
        __syncthreads();
        int lh=threadIdx.y/2;h=pass*(swinmh_H/2)+lh;
        int q=l+32*(threadIdx.y&1);
        float query[32],o[32];unsigned scores[32];
        #pragma unroll
        for(int cp=0;cp<8;cp++) {
            unsigned word=*reinterpret_cast<const unsigned*>(qkv+q*C+h*32+cp*4);
            #pragma unroll
            for(int j=0;j<4;j++){query[cp*4+j]=swinmh_decode(word>>(8*j));o[cp*4+j]=0;}
        }
        float sum=0;
        #pragma unroll 1
        for(int kp=0;kp<32;kp++) {
            float a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0;
            #pragma unroll
            for(int c=0;c<8;c++) {
                unsigned k0=*reinterpret_cast<const unsigned*>(kv+lh*4096+(4*c)*64+2*kp);
                unsigned k1=*reinterpret_cast<const unsigned*>(kv+lh*4096+(4*c+1)*64+2*kp);
                unsigned k2=*reinterpret_cast<const unsigned*>(kv+lh*4096+(4*c+2)*64+2*kp);
                unsigned k3=*reinterpret_cast<const unsigned*>(kv+lh*4096+(4*c+3)*64+2*kp);
                a0=fmaf(query[4*c],__uint_as_float(k0<<16),a0);b0=fmaf(query[4*c],__uint_as_float(k0&0xffff0000u),b0);
                a1=fmaf(query[4*c+1],__uint_as_float(k1<<16),a1);b1=fmaf(query[4*c+1],__uint_as_float(k1&0xffff0000u),b1);
                a2=fmaf(query[4*c+2],__uint_as_float(k2<<16),a2);b2=fmaf(query[4*c+2],__uint_as_float(k2&0xffff0000u),b2);
                a3=fmaf(query[4*c+3],__uint_as_float(k3<<16),a3);b3=fmaf(query[4*c+3],__uint_as_float(k3&0xffff0000u),b3);
            }
            int half=q>>5,mt=(q&31)>>4,row=q&15,nt=kp>>2;
            int lane=4*(row&7)+(kp&3),ww=2*(nt&1)+(row>>3);
            int off=h*4096+half*2048+((mt*4+(nt>>1))*32+lane)*8+ww*2;
            unsigned bias=__ldg(reinterpret_cast<const unsigned*>(w+off*2));
            unsigned e0=swinmh_expbits(swinmh_half((a0+a1)+(a2+a3)+__half2float(__ushort_as_half(bias))));
            unsigned e1=swinmh_expbits(swinmh_half((b0+b1)+(b2+b3)+__half2float(__ushort_as_half(bias>>16))));
            scores[kp]=e0|(e1<<16);
            sum+=__half2float(__ushort_as_half(e0));sum+=__half2float(__ushort_as_half(e1));
        }
        float inv=swinmh_half(1.f/swinmh_half(sum));
        #pragma unroll 1
        for(int kp=0;kp<32;kp++) {
            unsigned word=scores[kp];
            float p0=swinmh_decode(swinmh_encode(swinmh_half(__half2float(__ushort_as_half(word))*inv)));
            float p1=swinmh_decode(swinmh_encode(swinmh_half(__half2float(__ushort_as_half(word>>16))*inv)));
            #pragma unroll
            for(int c=0;c<32;c++) {
                unsigned v=*reinterpret_cast<const unsigned*>(kv+lh*4096+2048+c*64+2*kp);
                o[c]=fmaf(p0,__uint_as_float(v<<16),o[c]);o[c]=fmaf(p1,__uint_as_float(v&0xffff0000u),o[c]);
            }
            if(SWINMH_ROUND&&kp==15) {
                #pragma unroll
                for(int c=0;c<32;c++)o[c]=swinmh_half(o[c]);
            }
        }
        #pragma unroll
        for(int cp=0;cp<8;cp++) {
            unsigned word=0;
            #pragma unroll
            for(int j=0;j<4;j++)word|=swinmh_encode(swinmh_round(o[cp*4+j]))<<(8*j);
            *reinterpret_cast<unsigned*>(out+q*C+h*32+cp*4)=word;
        }
        __syncthreads();
    }
}
__device__ __forceinline__ void swinmh_layer(swinmh_params p) {
    constexpr int C=swinmh_C,H=swinmh_H,U=4*C*C,D=H*4096,R=C*C,S=SWINMH_VARIANT==3?U+D+R+2*C*C:U+D+R+16,QKV=SWINMH_VARIANT==3?S+4*C:S+2*C+16,BIAS=QKV+3*C*C,TAU=BIAS+H*8192,PROJ=TAU+(H>4?32:16),S2=PROJ+C*C;
    __shared__ float swinmh_shared[SWINMH_HEADS==2?3072:SWINMH_HEADS==4?6144:12288];
    float* swinmh_as=swinmh_shared;float* swinmh_bs=swinmh_shared+(SWINMH_HEADS==2?1024:2048);
    int l=threadIdx.x,h=threadIdx.y,g=l>>2,t=l&3,tid=h*32+l;
    int win=blockIdx.y*gridDim.x+blockIdx.x;
    
    if(tid==0) {
        unsigned smid;
        asm("mov.u32 %0, %%smid;" : "=r"(smid));
        constexpr unsigned perSM=swinmh_H==2?8:swinmh_H==4?4:2;
        int owner=-1;
        for(unsigned j=0;j<19;j++) {
            unsigned at=(smid+j)%19,old=atomicCAS(swinmh_pool_owners+at,0u,smid+1);
            if(old==0||old==smid+1) {owner=at;break;}
        }
        for(unsigned i=win;;i++) {
            unsigned slot=owner>=0?owner*perSM+(i&(perSM-1)):i%swinmh_windows;
            if(atomicCAS(swinmh_pool_used+slot,0u,1u)==0u) { swinmh_as[0]=__uint_as_float(slot);break; }
        }
    }
    __syncthreads();
    int slot=__float_as_uint(swinmh_as[0]);
    unsigned char* x=swinmh_scratch+slot*64*C*4;__half* y=reinterpret_cast<__half*>(x+2*64*C);
    unsigned char* midbytes=x;unsigned char* qkv=x+64*C;unsigned char* o=qkv;
    bool cached=swinmh_r7_acquire(p.weights+QKV,swinmh_as);
    if(SWINMH_VARIANT==1||SWINMH_VARIANT==2) {
        if(tid==0&&p.waitFlags) {
            int nx=(p.width-min(p.ox,0)+7)/8,ny=(p.height-min(p.oy,0)+7)/8;
            int gx=blockIdx.x*8+p.x0-p.ox,gy=blockIdx.y*8+p.y0-p.oy;
            for(int fy=max(gy,0)/8;fy<min((gy+15)/8,ny);fy++)for(int fx=max(gx,0)/8;fx<min((gx+15)/8,nx);fx++)while(*reinterpret_cast<const volatile int*>(p.waitFlags+fy*nx+fx)<0){}
            __threadfence();
        }
        __syncthreads();
    }
    if(SWINMH_VARIANT==3) {
        int ih=((p.height+1)/2+3)&~3,iw=((p.width+1)/2+3)&~3;
        for(int z=tid;z<32*2*C;z+=32*H) {
            int r=z/(2*C),c=z%(2*C),xx=(int(blockIdx.x)*8+p.x0)/2+(r&3),yy=(int(blockIdx.y)*8+p.y0)/2+(r>>2);
            midbytes[z]=r<16?swinmh_read(p.input,xx,yy,c,ih,iw,false):0;
        }
        __syncthreads();
        float a[2][8];swinmh_gemm<6,2>(midbytes,p.weights+U+D+R,swinmh_as,swinmh_bs,a);
        #pragma unroll
        for(int i=0;i<2;i++)for(int j=0;j<8;j++) {
            int r=g+8*i,c=h*32+swinmh_col(t,j);
            for(int py=0;py<2;py++)for(int px=0;px<2;px++) {
                int dx=2*(r&3)+px,dy=2*(r>>2)+py,xx=int(blockIdx.x)*8+p.x0+dx,yy=int(blockIdx.y)*8+p.y0+dy;
                int tok=16*((dx>>2)+2*(dy>>2))+(dx&3)+4*(dy&3);
                float res=swinmh_half(swinmh_decode(swinmh_read(p.residual,xx,yy,c,p.height,p.width,true))*__half2float(reinterpret_cast<const __half*>(p.weights+S+2*C)[c]));
                unsigned v=swinmh_encode(swinmh_half(a[i][j]+res));x[tok*C+c]=v;
                y[tok*C+c]=__float2half_rn(swinmh_decode(v)*__half2float(reinterpret_cast<const __half*>(p.weights+S)[c]));
            }
        }
    } else {
    for(int z=tid;z<32*C;z+=32*H) {
        int r=z/(C/2),c=2*(z%(C/2)),xx=blockIdx.x*8+p.x0+4*((r>>4)&1)+(r&3),yy=blockIdx.y*8+p.y0+4*(r>>5)+((r>>2)&3);
        unsigned word=0;
        if(xx>=0&&xx<p.width&&yy>=0&&yy<p.height) {
            int off=SWINMH_VARIANT!=0?swinmh_blockoff(xx,yy,c,p.width,C):((c/16*p.height+yy)*p.width+xx)*16+(swinmh_pos(c)&15);
            word=*reinterpret_cast<const unsigned short*>(p.input+off);
        }
        *reinterpret_cast<unsigned short*>(x+2*z)=word;
        float2 scales=__half22float2(*reinterpret_cast<const __half2*>(p.weights+S+2*c));
        *reinterpret_cast<__half2*>(y+2*z)=__floats2half2_rn(swinmh_decode(word)*scales.x,swinmh_decode(word>>8)*scales.y);
    }
    }
    __syncthreads();
    swinmh_ffn_r4(x,qkv,p.weights,swinmh_shared,swinmh_shared+1280*H);
    __syncthreads();
    for(int z=tid;z<64*C;z+=32*H)midbytes[z]=qkv[z];
    __syncthreads();
    {
        for(int batch=0;batch<1;batch++){float a[8][8];swinmh_gemm<3>(midbytes+batch*64*C,p.weights+U+D,swinmh_as,swinmh_bs,a,0,y+batch*64*C);
        #pragma unroll
        for(int i=0;i<8;i++)for(int j=0;j<8;j+=2) {int off=(batch*64+swinmh_tok(g,i))*C+h*32+swinmh_col(t,j);*reinterpret_cast<unsigned short*>(midbytes+off)=swinmh_encode(a[i][j])|(swinmh_encode(a[i][j+1])<<8);}}
    }
    __syncthreads();
    for(int part=0;part<3;part++) {for(int batch=0;batch<1;batch++){
        float a[8][8];swinmh_gemm<2>(midbytes+batch*64*C,p.weights+QKV,swinmh_as,swinmh_bs,a,part,nullptr,nullptr,cached);
        #pragma unroll
        for(int i=0;i<8;i++) {
            float inv=1;
            if(part<2) {
                float ss=0;
                #pragma unroll
                for(int j=0;j<8;j++)ss+=swinmh_half(a[i][j]*a[i][j]);
                ss+=__shfl_xor_sync(0xffffffff,ss,2);ss+=__shfl_xor_sync(0xffffffff,ss,1);
                inv=swinmh_half(rsqrtf(fmaxf(swinmh_half(ss),swinmh_half(__int_as_float(948045311)))));
            }
            float tau=part==0?swinmh_half(reinterpret_cast<const float*>(p.weights+TAU)[h]):1;
            #pragma unroll
            for(int j=0;j<8;j+=2) {
                float v0=part<2?swinmh_half(swinmh_half(a[i][j]*inv)*tau):a[i][j];
                float v1=part<2?swinmh_half(swinmh_half(a[i][j+1]*inv)*tau):a[i][j+1];
                *reinterpret_cast<unsigned short*>(qkv+part*64*C+(batch*64+swinmh_tok(g,i))*C+h*32+swinmh_col(t,j))=swinmh_encode(v0)|(swinmh_encode(v1)<<8);
            }
        }
    }}
    __syncthreads();
    swinmh_r7_release(cached);
    swinmh_attention(qkv,o,reinterpret_cast<unsigned short*>(swinmh_bs),p.weights+BIAS);
    __syncthreads();
    {
        for(int batch=0;batch<1;batch++){float a[8][8];swinmh_gemm<4>(o+batch*64*C,p.weights+PROJ,swinmh_as,swinmh_bs,a,0,reinterpret_cast<const __half*>(midbytes+batch*64*C),reinterpret_cast<const __half*>(p.weights+S2));
        #pragma unroll
        for(int i=0;i<8;i++)for(int j=0;j<8;j+=2) {
            int r=batch*64+swinmh_tok(g,i),c=h*32+swinmh_col(t,j),off=r*C+c;
            unsigned v=swinmh_encode(a[i][j])|(swinmh_encode(a[i][j+1])<<8);
            int xx=int(blockIdx.x)*8+p.x0+4*((r>>4)&1)+(r&3),yy=int(blockIdx.y)*8+p.y0+4*(r>>5)+((r>>2)&3);
            if(SWINMH_VARIANT==2)*reinterpret_cast<__half2*>(y+off)=__floats2half2_rn(a[i][j],a[i][j+1]);
            if(xx>=0&&xx<p.width&&yy>=0&&yy<p.height) {
                if(SWINMH_VARIANT==1)*reinterpret_cast<unsigned short*>(p.output+((c/16*p.height+yy)*p.width+xx)*16+(swinmh_pos(c)&15))=v;
                else *reinterpret_cast<unsigned short*>(p.output+swinmh_blockoff(xx,yy,c,p.width,C))=v;
            }
        }
    }}
    if(SWINMH_VARIANT==2) {
        __syncthreads();
        for(int z=tid;z<32*C;z+=32*H) {
            int r=z/C,c=z%C,px=r&3,py=r>>2;float avg=0;
            if(r<16) {
                float v[4];
                for(int i=0;i<4;i++){int dx=2*px+(i&1),dy=2*py+(i>>1),tok=16*((dx>>2)+2*(dy>>2))+(dx&3)+4*(dy&3);v[i]=__half2float(y[tok*C+c]);}
                avg=swinmh_half(swinmh_half(swinmh_half(v[0]+v[1])+swinmh_half(v[2]+v[3]))*.25f);
            }
            midbytes[z]=swinmh_encode(avg);
        }
        __syncthreads();
        for(int part=0;part<2;part++) {
            float a[2][8];swinmh_gemm<5,2>(midbytes,p.weights+S2+2*C,swinmh_as,swinmh_bs,a,part);
            #pragma unroll
            for(int i=0;i<2;i++)for(int j=0;j<8;j++) {
                int r=g+8*i,c=part*C+h*32+swinmh_col(t,j),xx=(int(blockIdx.x)*8+p.x0)/2+(r&3),yy=(int(blockIdx.y)*8+p.y0)/2+(r>>2);
                if(xx>=0&&xx<p.dw&&yy>=0&&yy<p.dh)p.ds[((c/16*p.dh+yy)*p.dw+xx)*16+(swinmh_pos(c)&15)]=(xx<p.width/2&&yy<p.height/2)?swinmh_encode(a[i][j]):0;
            }
        }
        if(blockIdx.x==0&&blockIdx.y==0) {
            const int vw=(p.width+1)/2,vh=(p.height+1)/2,padw=max(p.dw-vw,0),padh=max(p.dh-vh,0);
            for(int z=tid;z<padw*p.dh*2*C;z+=32*H) {
                int s=z&15,k=z>>4,xx=k%padw;k/=padw;int yy=k%p.dh,group=k/p.dh;
                p.ds[((group*p.dh+yy)*p.dw+vw+xx)*16+s]=0;
            }
            for(int z=tid;z<vw*padh*2*C;z+=32*H) {
                int s=z&15,k=z>>4,xx=k%vw;k/=vw;int yy=k%padh,group=k/padh;
                p.ds[((group*p.dh+vh+yy)*p.dw+xx)*16+s]=0;
            }
        }
    }
    __threadfence();__syncthreads();
    if((SWINMH_VARIANT==0||SWINMH_VARIANT==3)&&tid==0&&p.signalFlags)*reinterpret_cast<volatile int*>(p.signalFlags+win)=0;
    if(tid==0)atomicExch(swinmh_pool_used+slot,0u);
}
#if SWINMH_HEADS==2
extern "C" __global__ __launch_bounds__(64) void SWINMH_JOIN(cc_tinlayout_fused_swin_2h_64_2_,SWINMH_SUFFIX)(swinmh_params p){swinmh_layer(p);}
#elif SWINMH_HEADS==4
extern "C" __global__ __launch_bounds__(128) void SWINMH_JOIN(cc_tinlayout_fused_swin_4h_128_4_,SWINMH_SUFFIX)(swinmh_params p){swinmh_layer(p);}
#else
extern "C" __global__ __launch_bounds__(256) void SWINMH_JOIN(cc_tinlayout_fused_swin_8h_256_8_,SWINMH_SUFFIX)(swinmh_params p){swinmh_layer(p);}
#endif
