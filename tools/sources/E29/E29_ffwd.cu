// NR 310.9.16: dense 512->512, then eight grouped 64->256->64 MLPs.
// Blob: dense 0, grouped up 262144, grouped down 393216. FP8 stage boundaries, f16 per 32-k accumulation.
#include "vitfuse_common.cuh"
#include "vitfuse_r5_silu_lut.cuh"
struct __align__(8) vitfuse_FfwdParams {const unsigned char* input;unsigned char* output;const unsigned char* weights;int H,W;const int* wait;int x0,y0;int* signal;};
static_assert(sizeof(vitfuse_FfwdParams)==56,"params");
union vitfuse_FfwdScratch {struct {float A[32][64];float B[32][128];} gemm;struct {unsigned short X[2][64][64];union {unsigned short up[2][64][32];unsigned short down[2][32][64];} group;} post;};
__device__ __forceinline__ float vitfuse_silu(float x) {float c=fminf(fmaxf(x,-4.f),4.f);float a=vitfuse_half(fmaf(-.055908203125f,fabsf(c),.447265625f));return vitfuse_half(x*vitfuse_half(fmaf(c,a,.89453125f)));}
template<bool Inp>
__device__ __forceinline__ void vitfuse_Ffwd(vitfuse_FfwdParams p) {
 __shared__ vitfuse_FfwdScratch vitfuse_fm;
 int lane=threadIdx.x,l=(lane>>2)+8*(lane&3),w=threadIdx.y,th=w*32+lane,group=(int)blockIdx.z*2+(w&1);
 int wx=blockIdx.x*8,wy=blockIdx.y*8;
 if(!Inp&&th==0) {
  int nx=(p.W-min(p.x0,0)+7)/8;
  for(int y=(wy-p.y0)/8;y<=(min(wy+7,p.H-1)-p.y0)/8;y++)for(int x=(wx-p.x0)/8;x<=(min(wx+7,p.W-1)-p.x0)/8;x++)for(int c=0;c<2;c++)vitfuse_wait(p.wait+(y*2+c)*nx+x);
  __threadfence();
 }
 __syncthreads();
 for(int pass=0;pass<1;pass++) {
 int tok=l+32*(w>>1);
 float acc[64];
#pragma unroll
 for(int j=0;j<(64);j++)acc[j]=0;
 for(int sl=0;sl<16;sl++) {
  __syncthreads();
  if(th<128) {
   int block=th>>5,g=lane>>2,t=lane&3,by=wy/4+(block>>1),bx=wx/4+(block&1);uint4 a=make_uint4(0,0,0,0);
   if(!Inp&&by<p.H/4&&bx<p.W/4)a=*reinterpret_cast<const uint4*>(p.input+(by*(p.W/4)+bx)*8192+sl*512+lane*16);
   unsigned aw[4]={a.x,a.y,a.z,a.w};
   if(Inp) {
#pragma unroll
    for(int r=0;r<4;r++) {
     int row=g+8*(r&1),X=bx*4+(row&3),Y=by*4+((row>>2)&1)+2*(row>>3),half=sl*2+(r>>1);
     aw[r]=(Y<p.H&&X<p.W)?*reinterpret_cast<const unsigned*>(p.input+((half*p.H+Y)*p.W+X)*16+t*4):0;
    }
   }
#pragma unroll
   for(int r=0;r<4;r++) {
#pragma unroll
    for(int b=0;b<4;b++)vitfuse_fm.gemm.A[4*t+16*(r>>1)+b][block*16+g+8*(r&1)]=vitfuse_decode(aw[r]>>(8*b));
   }
  }
  for(int load=0;load<(1);load++) {
   int loadth=th+load*128;
   const uint4* src=reinterpret_cast<const uint4*>(p.weights+sl*512*32+(int)blockIdx.z*128*32+loadth*32);
   uint4 a=src[0],b=src[1];unsigned words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
   for(int r=0;r<8;r++) {
    int off=loadth*32+r*4,chunk=off>>9,fl=(off>>4)&31,rr=(off>>2)&3,n=chunk*16+8*(rr>>1)+(fl>>2),k=4*(fl&3)+16*(rr&1);
#pragma unroll
    for(int b=0;b<4;b++)vitfuse_fm.gemm.B[k+b][(n/64)*64+((n%16)/4)*16+((n/16)%4)*4+(n%4)]=vitfuse_decode(words[r]>>(8*b));
   }
  }
  __syncthreads();
#pragma unroll 4
  for(int k=0;k<32;k++) {
   float a[4];
#pragma unroll
   for(int i=0;i<(4);++i)a[i]=vitfuse_fm.gemm.A[k][(lane>>2)+8*i+32*(w>>1)];
   const float4* bp=reinterpret_cast<const float4*>(&vitfuse_fm.gemm.B[k][(w&1)*64+(lane&3)*4]);
   float4 b0=bp[0],b1=bp[4],b2=bp[8],b3=bp[12];
   float bv[16]={b0.x,b0.y,b0.z,b0.w,b1.x,b1.y,b1.z,b1.w,b2.x,b2.y,b2.z,b2.w,b3.x,b3.y,b3.z,b3.w};
#pragma unroll
   for(int i=0;i<(4);++i) {
#pragma unroll
    for(int j=0;j<16;++j)acc[i*16+j]=fmaf(a[i],bv[j],acc[i*16+j]);
   }
  }
#pragma unroll
  for(int j=0;j<(64);j++)acc[j]=vitfuse_half(acc[j]);
 }
 __syncthreads();
#pragma unroll
 for(int j=0;j<16;++j) {
  unsigned v[4];
#pragma unroll
  for(int i=0;i<4;++i)v[i]=__float_as_uint(vitfuse_decode(vitfuse_encode(vitfuse_half(acc[i*16+j]))))>>16;
  *reinterpret_cast<uint2*>(&vitfuse_fm.post.X[w&1][(lane&3)*16+j][(lane>>2)*8+4*(w>>1)])=make_uint2(v[0]|(v[1]<<16),v[2]|(v[3]<<16));
 }
#pragma unroll
 for(int j=0;j<(64);j++)acc[j]=0;
 for(int part=0;part<8;part++) {
  __syncthreads();
  for(int load=0;load<(1);load++) {
   int loadth=th+load*128,g=loadth>>6,local=(loadth&63)*32;
   const uint4* src=reinterpret_cast<const uint4*>(p.weights+262144+((int)blockIdx.z*2+g)*16384+(local/1024)*8192+part*1024+(local%1024));
   uint4 a=src[0],b=src[1];unsigned words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
   for(int r=0;r<8;r++) {
    int off=local+r*4,ks=off/1024,chunk=(off%1024)/512,fl=(off>>4)&31,rr=(off>>2)&3,n=chunk*16+8*(rr>>1)+(fl>>2),k=ks*32+4*(fl&3)+16*(rr&1);
#pragma unroll
    for(int b=0;b<4;b++)vitfuse_fm.post.group.up[g][k+b][n]=__float_as_uint(vitfuse_decode(words[r]>>(8*b)))>>16;
   }
  }
  __syncthreads();float h[4][8]={};
#pragma unroll 2
  for(int k=0;k<64;k++) {
   int nk=16*(k>>4)+vitfuse_channel(k&15);
   uint2 ap=*reinterpret_cast<const uint2*>(&vitfuse_fm.post.X[w&1][nk][(lane>>2)*8+4*(w>>1)]);
   float av[4]={__uint_as_float(ap.x<<16),__uint_as_float(ap.x&0xffff0000u),__uint_as_float(ap.y<<16),__uint_as_float(ap.y&0xffff0000u)};
   const uint2* bp=reinterpret_cast<const uint2*>(&vitfuse_fm.post.group.up[w&1][k][(lane&3)*8]);
   uint2 b0=bp[0],b1=bp[1];
   float bv[8]={__uint_as_float(b0.x<<16),__uint_as_float(b0.x&0xffff0000u),__uint_as_float(b0.y<<16),__uint_as_float(b0.y&0xffff0000u),__uint_as_float(b1.x<<16),__uint_as_float(b1.x&0xffff0000u),__uint_as_float(b1.y<<16),__uint_as_float(b1.y&0xffff0000u)};
#pragma unroll
   for(int i=0;i<(4);++i) {
#pragma unroll
    for(int j=0;j<8;++j)h[i][j]=fmaf(av[i],bv[j],h[i][j]);
   }
   if((k&31)==31) {
#pragma unroll
    for(int i=0;i<(4);++i) {
#pragma unroll
     for(int j=0;j<8;++j)h[i][j]=vitfuse_half(h[i][j]);
    }
   }
  }
#pragma unroll
  for(int i=0;i<(4);++i) {
#pragma unroll
   for(int j=0;j<8;++j)h[i][j]=vitfuse_decode(__ldg(vitfuse_r9_silu_lut+__half_as_ushort(__float2half_rn(h[i][j]))));
  }
  __syncthreads();
  for(int load=0;load<(1);load++) {
   int loadth=th+load*128,g=loadth>>6,local=(loadth&63)*32;
   const uint4* src=reinterpret_cast<const uint4*>(p.weights+393216+((int)blockIdx.z*2+g)*16384+part*2048+local);
   uint4 a=src[0],b=src[1];unsigned words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
   for(int r=0;r<8;r++) {
    int off=local+r*4,chunk=off>>9,fl=(off>>4)&31,rr=(off>>2)&3,n=chunk*16+8*(rr>>1)+(fl>>2),k=4*(fl&3)+16*(rr&1);
#pragma unroll
    for(int b=0;b<4;b++)vitfuse_fm.post.group.down[g][k+b][((n%16)/4)*16+(n/16)*4+(n%4)]=__float_as_uint(vitfuse_decode(words[r]>>(8*b)))>>16;
   }
  }
  __syncthreads();
#pragma unroll 4
  for(int k=0;k<32;k++) {
   int nk=16*(k>>4)+vitfuse_channel(k&15);
   float av[4];
#pragma unroll
   for(int i=0;i<(4);++i)av[i]=__shfl_sync(0xffffffff,h[i][nk&7],(lane&~3)|(nk>>3));
   const uint2* bp=reinterpret_cast<const uint2*>(&vitfuse_fm.post.group.down[w&1][k][(lane&3)*4]);
   uint2 b0=bp[0],b1=bp[4],b2=bp[8],b3=bp[12];
   float bv[16]={__uint_as_float(b0.x<<16),__uint_as_float(b0.x&0xffff0000u),__uint_as_float(b0.y<<16),__uint_as_float(b0.y&0xffff0000u),__uint_as_float(b1.x<<16),__uint_as_float(b1.x&0xffff0000u),__uint_as_float(b1.y<<16),__uint_as_float(b1.y&0xffff0000u),__uint_as_float(b2.x<<16),__uint_as_float(b2.x&0xffff0000u),__uint_as_float(b2.y<<16),__uint_as_float(b2.y&0xffff0000u),__uint_as_float(b3.x<<16),__uint_as_float(b3.x&0xffff0000u),__uint_as_float(b3.y<<16),__uint_as_float(b3.y&0xffff0000u)};
#pragma unroll
   for(int i=0;i<(4);++i) {
#pragma unroll
    for(int j=0;j<16;++j)acc[i*16+j]=fmaf(av[i],bv[j],acc[i*16+j]);
   }
  }
#pragma unroll
  for(int j=0;j<(64);j++)acc[j]=vitfuse_half(acc[j]);
 }
#pragma unroll
 for(int i=0;i<(4);++i) {
  int query=(lane>>2)+8*i+32*(w>>1);
  int m=query>>4,r=query&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
  if(by<p.H/4&&bx<p.W/4) {
#pragma unroll
   for(int j=0;j<16;++j)p.output[(by*(p.W/4)+bx)*8192+group*1024+vitfuse_aoff(r,vitfuse_stored((lane&3)*16+j))]=vitfuse_encode(acc[i*16+j]);
  }
 }

 }
 __threadfence();__syncthreads();if(th==0) {
 int* flag=p.signal+((int)blockIdx.y*2+blockIdx.z/2)*((p.W+7)/8)+blockIdx.x;
 if(atomicSub(flag,1)==-2) {__threadfence();*reinterpret_cast<volatile int*>(flag)=0;}
 }
}

extern "C" __global__ void __launch_bounds__(128,4) cc_split_swin_16h_ffwd_inpview_512_tilesync_fp8(vitfuse_FfwdParams p) {vitfuse_Ffwd<true>(p);}

extern "C" __global__ void __launch_bounds__(128,4) cc_split_swin_16h_ffwd_512_chained_fp8(vitfuse_FfwdParams p) {vitfuse_Ffwd<false>(p);}
