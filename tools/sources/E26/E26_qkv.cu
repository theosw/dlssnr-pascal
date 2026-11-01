#include "vitfuse_common.cuh"
struct __align__(8) vitfuse_QkvParams {const unsigned char* input; unsigned char* output; const unsigned char* weights; int H,W,x0,y0;const int* wait;int* signal;};
static_assert(sizeof(vitfuse_QkvParams)==56,"params");
struct vitfuse_Scratch {
 union {
  struct {float A[32][64];float B[4][32][32];} gemm;
  struct {unsigned char V[4][64][32];__half E[4][64][32];} post;
 };
 unsigned char K[4][32][64];
};
__device__ __forceinline__ void vitfuse_gemm(const vitfuse_QkvParams& p,vitfuse_Scratch& sm,int comp,float (&acc)[2][32]) {
 const int l=threadIdx.x,w=threadIdx.y,th=w*32+l,head=blockIdx.z*4+w;
#pragma unroll
 for(int i=0;i<2;i++) {
#pragma unroll
  for(int j=0;j<32;j++)acc[i][j]=0.f;
 }
 for(int stage=0;stage<16;++stage) {
  __syncthreads();
  int block=th>>5,g=l>>2,t=l&3;
  int by=(p.y0+8*(int)blockIdx.y)/4+(block>>1),bx=(p.x0+8*(int)blockIdx.x)/4+(block&1);
  uint4 av=make_uint4(0,0,0,0);
  if(by>=0&&by<p.H/4&&bx>=0&&bx<p.W/4)av=*reinterpret_cast<const uint4*>(p.input+(by*(p.W/4)+bx)*8192+stage*512+l*16);
  unsigned aw[4]={av.x,av.y,av.z,av.w};
#pragma unroll
  for(int half=0;half<2;++half) {
#pragma unroll
   for(int r=0;r<2;r++) {
    unsigned v=aw[half*2+r];
#pragma unroll
    for(int b=0;b<4;b++)sm.gemm.A[16*half+4*t+b][block*16+g+8*r]=vitfuse_decode(v>>(8*b));
   }
  }
#pragma unroll
  for(int c=0;c<2;c++) {
   uint4 wv=*reinterpret_cast<const uint4*>(p.weights+stage*1536*32+(head*96+comp*32)*32+c*512+l*16);
   unsigned ww[4]={wv.x,wv.y,wv.z,wv.w};
#pragma unroll
   for(int half=0;half<2;++half) {
#pragma unroll
    for(int r=0;r<2;r++) {
     unsigned v=ww[r*2+half];
#pragma unroll
     for(int b=0;b<4;b++)sm.gemm.B[w][16*half+4*t+b][c*16+r*8+g]=vitfuse_decode(v>>(8*b));
    }
   }
  }
  __syncthreads();
#pragma unroll 4
  for(int k=0;k<32;k++) {
   const int g=l>>1,t=l&1;
   float a[4];
#pragma unroll
   for(int i=0;i<4;++i)a[i]=sm.gemm.A[k][g+16*i];
   const float4* br=reinterpret_cast<const float4*>(&sm.gemm.B[w][k][t*16]);
   float4 b0=br[0],b1=br[1],b2=br[2],b3=br[3];
   float bv[16]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w,b2.x,b2.y,b2.z,b2.w,b3.x,b3.y,b3.z,b3.w};
#pragma unroll
   for(int i=0;i<4;++i) {
#pragma unroll
    for(int j=0;j<16;++j)acc[i>>1][(i&1)*16+j]=fmaf(a[i],bv[j],acc[i>>1][(i&1)*16+j]);
   }
  }

  {
#pragma unroll
   for(int i=0;i<2;i++) {
#pragma unroll
    for(int j=0;j<32;j++)acc[i][j]=vitfuse_half(acc[i][j]);
   }
  }
 }

#pragma unroll
 for(int i=0;i<2;++i) {
#pragma unroll
  for(int j=0;j<16;++j) {
   int t=l&1;
   float low=acc[i][j],high=acc[i][16+j];
   float peer=__shfl_xor_sync(0xffffffff,t?low:high,1);
   acc[i][j]=t?peer:low;
   acc[i][16+j]=t?high:peer;
  }
 }

}
__device__ __forceinline__ void vitfuse_scores(const vitfuse_QkvParams& p,const vitfuse_Scratch& sm,const unsigned (&q8)[8],int tok,int head,int part,float (&scores)[16]) {
 int w=threadIdx.y;
#pragma unroll
 for(int n=0;n<16;n++)scores[n]=0;
#pragma unroll
 for(int k=0;k<32;k++) {
  float q=vitfuse_decode(q8[k/4]>>(8*(k&3)));
#pragma unroll
  for(int n=0;n<16;n++)scores[n]=fmaf(q,vitfuse_decode(sm.K[w][k][part*16+n]),scores[n]);
 }
#pragma unroll
 for(int n=0;n<16;n++) {
  int key=part*16+n,block=tok/16,r=tok&15,lane=(r&7)*4+(key&7)/2;
  int at=(block*4+key/16)*256+lane*8+((key>>3)&1)*4+(r>>3)*2+(key&1);
  float bias=__half2float(reinterpret_cast<const __half*>(p.weights+786432+head*8192)[at]);
  scores[n]=vitfuse_exp(vitfuse_half(vitfuse_half(scores[n])+bias));
 }
}
__device__ __forceinline__ unsigned vitfuse_positive_prob8(float x) {
 unsigned b=__float_as_uint(x);
 if(b<0x3c800000u)return __float_as_uint(__fadd_rn(x,16384.f))-0x46800000u;
 return ((b+0x7ffffu+((b>>20)&1u))>>20)-960u;
}

extern "C" __global__ void __launch_bounds__(128,3) cc_split_swin_16h_qkv_512_chained_fp8(vitfuse_QkvParams p) {
 __shared__ vitfuse_Scratch vitfuse_sm;
 const int l=(threadIdx.x>>1)+16*(threadIdx.x&1),w=threadIdx.y,head=blockIdx.z*4+w;
 const int wx=p.x0+8*(int)blockIdx.x,wy=p.y0+8*(int)blockIdx.y;
 if(l==0&&w==0) {
  // Producers publish two channel groups for each 8x8 spatial tile.
  for(int y=max(wy,0)/8;y<(min(wy+8,p.H)+7)/8;y++) {
   for(int x=max(wx,0)/8;x<(min(wx+8,p.W)+7)/8;x++) {
    for(int c=0;c<2;c++)vitfuse_wait(p.wait+(y*2+c)*((p.W+7)/8)+x);
   }
  }  __threadfence();
 }
 __syncthreads();
 unsigned q8[2][8];
 float acc[2][32];
#pragma unroll 1
 for(int comp=0;comp<3;comp++) {
  vitfuse_gemm(p,vitfuse_sm,comp,acc);
  if(comp==2)__syncthreads(); // V reuses GEMM A after every warp finishes the final slice.
#pragma unroll
  for(int i=0;i<2;i++) {
   float norm=1.f;
#pragma unroll
   for(int j=0;j<32;j++)acc[i][j]=vitfuse_half(acc[i][j]);
   if(comp<2) {
    float parity[2];
#pragma unroll
    for(int b=0;b<2;b++) {
     float v[4];
#pragma unroll
     for(int t=0;t<4;t++) {
      float x0=acc[i][2*t+b],x1=acc[i][8+2*t+b],x2=acc[i][16+2*t+b],x3=acc[i][24+2*t+b];
      v[t]=vitfuse_half(vitfuse_half(vitfuse_half(x1*x1)+vitfuse_half(x3*x3))+vitfuse_half(vitfuse_half(x0*x0)+vitfuse_half(x2*x2)));
     }
     parity[b]=vitfuse_half(vitfuse_half(v[0]+v[2])+vitfuse_half(v[1]+v[3]));
    }
    float ss=vitfuse_half(parity[0]+parity[1]);
    norm=vitfuse_half(rsqrtf(fmaxf(vitfuse_half(ss),.00006198883f)));
   }
   float scale=comp==0?vitfuse_half(reinterpret_cast<const float*>(p.weights+917504)[head]):1.f;
#pragma unroll
   for(int z=0;z<8;z++) {
    unsigned bits=0;
#pragma unroll
    for(int b=0;b<4;b++) {
     float v=acc[i][4*z+b];
     if(comp<2)v=vitfuse_half(vitfuse_half(v*norm)*scale);
     bits|=vitfuse_encode(v)<<(8*b);
    }
    if(comp==0)q8[i][z]=bits;
    else if(comp==2)*reinterpret_cast<unsigned*>(&vitfuse_sm.post.V[w][l+32*i][4*z])=bits;
    else {
#pragma unroll
     for(int b=0;b<4;b++)vitfuse_sm.K[w][4*z+b][l+32*i]=(bits>>(8*b))&255;
    }
   }
  }
 }


 __syncthreads();
 const int pl=threadIdx.x,rg=pl>>2,ct=pl&3;
#pragma unroll
 for(int pass=0;pass<2;++pass) {
  __syncthreads();
  float sums[4][2]={};
#pragma unroll 1
  for(int part=0;part<4;++part) {
   float sc[4][4]={};
#pragma unroll
   for(int word=0;word<8;++word) {
    unsigned qw[4];
#pragma unroll
    for(int i=0;i<4;++i) {
     int query=rg+8*i,src=(query&15)*2+(query>>4);
     qw[i]=__shfl_sync(0xffffffff,q8[pass][word],src);
    }
#pragma unroll
    for(int z=0;z<4;++z) {
     int d=word*4+z;
     float av[4],bv[4];
#pragma unroll
     for(int i=0;i<4;++i)av[i]=vitfuse_decode(qw[i]>>(8*z));
#pragma unroll
     for(int j=0;j<4;++j)bv[j]=vitfuse_decode(vitfuse_sm.K[w][d][part*16+ct*2+(j&1)+8*(j>>1)]);
#pragma unroll
     for(int i=0;i<4;++i) {
#pragma unroll
      for(int j=0;j<4;++j)sc[i][j]=fmaf(av[i],bv[j],sc[i][j]);
     }
    }
   }
#pragma unroll
   for(int i=0;i<4;++i) {
    int tok=rg+8*i+32*pass;
#pragma unroll
    for(int j=0;j<4;++j) {
     int key=part*16+ct*2+(j&1)+8*(j>>1),block=tok/16,r=tok&15,lane=(r&7)*4+(key&7)/2;
     int at=(block*4+key/16)*256+lane*8+((key>>3)&1)*4+(r>>3)*2+(key&1);
     float bias=__half2float(reinterpret_cast<const __half*>(p.weights+786432+head*8192)[at]);
     sc[i][j]=vitfuse_exp(vitfuse_half(vitfuse_half(sc[i][j])+bias));
     vitfuse_sm.post.E[w][key][rg+8*i]=__float2half_rn(sc[i][j]);
    }
#pragma unroll
    for(int j=0;j<2;++j)sums[i][j]=vitfuse_half(sums[i][j]+vitfuse_half(sc[i][j]+sc[i][j+2]));
   }
  }
  float inv[4];
#pragma unroll
  for(int i=0;i<4;++i) {
   float lo=__shfl_sync(0xffffffff,sums[i][0],pl&~3),hi=__shfl_sync(0xffffffff,sums[i][1],pl&~3);
#pragma unroll
   for(int t2=1;t2<4;++t2) {
    lo=vitfuse_half(lo+__shfl_sync(0xffffffff,sums[i][0],(pl&~3)|t2));
    hi=vitfuse_half(hi+__shfl_sync(0xffffffff,sums[i][1],(pl&~3)|t2));
   }
   inv[i]=vitfuse_half(1.f/fmaxf(vitfuse_half(lo+hi),.00006198883f));
  }
  __syncwarp();
  unsigned char* pp=reinterpret_cast<unsigned char*>(&vitfuse_sm.post.E[w][0][0]);
#pragma unroll 1
  for(int part=0;part<4;++part) {
   unsigned codes[4][4];
#pragma unroll
   for(int i=0;i<4;++i) {
#pragma unroll
    for(int j=0;j<4;++j) {
     int key=part*16+ct*2+(j&1)+8*(j>>1);
     codes[i][j]=vitfuse_positive_prob8(vitfuse_half(__half2float(vitfuse_sm.post.E[w][key][rg+8*i])*inv[i]));
    }
   }
   __syncwarp();
#pragma unroll
   for(int i=0;i<4;++i) {
#pragma unroll
    for(int j=0;j<4;++j) {
     int key=part*16+ct*2+(j&1)+8*(j>>1);
     pp[key*32+rg+8*i]=(unsigned char)codes[i][j];
    }
   }
   __syncwarp();
  }
  float out[4][8]={};
#pragma unroll 2
  for(int n=0;n<64;++n) {
   float av[4],bv[8];
#pragma unroll
   for(int i=0;i<4;++i)av[i]=vitfuse_decode(pp[n*32+rg+8*i]);
   const unsigned* vp=reinterpret_cast<const unsigned*>(&vitfuse_sm.post.V[w][n][ct*8]);
   unsigned v0=vp[0],v1=vp[1];
#pragma unroll
   for(int j=0;j<8;++j)bv[j]=vitfuse_decode((j<4?v0:v1)>>(8*(j&3)));
#pragma unroll
   for(int i=0;i<4;++i) {
#pragma unroll
    for(int j=0;j<8;++j)out[i][j]=fmaf(av[i],bv[j],out[i][j]);
   }
   if((n&31)==31) {
#pragma unroll
    for(int i=0;i<4;++i) {
#pragma unroll
     for(int j=0;j<8;++j)out[i][j]=vitfuse_half(out[i][j]);
    }
   }
  }
#pragma unroll
  for(int i=0;i<4;++i) {
   int tok=rg+8*i+32*pass,m=tok>>4,r=tok&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
   if(bx>=0&&bx<p.W/4&&by>=0&&by<p.H/4) {
#pragma unroll
    for(int j=0;j<8;j+=2) {
     int st=vitfuse_stored(ct*8+j);
     unsigned v=vitfuse_encode(vitfuse_half(out[i][j]))|(vitfuse_encode(vitfuse_half(out[i][j+1]))<<8);
     *reinterpret_cast<unsigned short*>(p.output+(by*(p.W/4)+bx)*8192+head*512+vitfuse_aoff(r,st))=v;
    }
   }
  }
 }
 __threadfence();__syncthreads();
 if(l==0&&w==0)p.signal[((int)blockIdx.y*4+blockIdx.z)*((p.W-min(p.x0,0)+7)/8)+blockIdx.x]=0;
}


