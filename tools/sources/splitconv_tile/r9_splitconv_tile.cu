// NR 310.9.16 split16 projections: scaled residual initializes f16 MMA accumulation.
// Completion flags cover 256 output channels per 8x8 window; pooling publishes the corresponding 4x4 low-resolution tile.
#include "s16v_common.cuh"
struct __align__(8) s16v_SplitConvParams {const unsigned char* input;const unsigned char* residual;unsigned char* output;const unsigned char* weights;int H,W;void* unused;int* signal;const int* wait;int x0,y0;};
static_assert(sizeof(s16v_SplitConvParams)==72,"params");
template<int Warps,int Groups,bool ResidualView=false,bool OutputView=false,int N=512,bool Residual=true,int WaitTile=8,bool Pool=false,int Cols=32>
__device__ __forceinline__ void s16v_SplitConv(const s16v_SplitConvParams& p,unsigned char* low=nullptr,int lowH=0,int lowW=0) {
 __shared__ float s16v_scA[32][64];__shared__ float s16v_scB[32][4*Cols];
 const int l=threadIdx.x,w=threadIdx.y,th=w*32+l,nx=(p.W+7)/8,cTile=blockIdx.x/nx,wx=(blockIdx.x%nx)*8,wy=blockIdx.y*8;
 if(th==0) {
  int ix=(p.W-min(p.x0,0)+WaitTile-1)/WaitTile;
  for(int y=(wy-p.y0)/WaitTile;y<=(min(wy+7,p.H-1)-p.y0)/WaitTile;y++)for(int x=(wx-p.x0)/WaitTile;x<=(min(wx+7,p.W-1)-p.x0)/WaitTile;x++)for(int c=0;c<Groups;c++)s16v_wait(p.wait+(y*Groups+c)*ix+x);
  __threadfence();
 }
 __syncthreads();
 for(int cp=0;cp<64/Cols;cp++) {
 constexpr int Rows=32/Warps, J=Cols/4;
 float acc[Rows][J];
#pragma unroll
 for(int i=0;i<Rows;i++) {
#pragma unroll
  for(int j=0;j<J;j++) {
   int tok=(l>>2)+8*i+32*(Warps==8?(w>>2):0),m=tok>>4,r=tok&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
   int c=cTile*256+cp*(4*Cols)+(w&3)*Cols+(l&3)*J+j,at=(by*(p.W/4)+bx)*16*N+s16v_aoff(r,s16v_stored(c));
   int X=bx*4+(r&3),Y=by*4+((r>>2)&1)+2*(r>>3),planar=((c/16*p.H+Y)*p.W+X)*16+(s16v_stored(c)&15);
   float v=0;
   if(Residual&&bx<p.W/4&&by<p.H/4) {float g=__half2float(reinterpret_cast<const __half*>(p.weights+512*N)[c]);v=s16v_half(s16v_decode(p.residual[ResidualView?planar:at])*g);}
   acc[i][j]=v;
  }
 }
 for(int sl=0;sl<16;sl++) {
  __syncthreads();
  if(th<128) {
   int block=th>>5,g=l>>2,t=l&3,by=wy/4+(block>>1),bx=wx/4+(block&1);uint4 a=make_uint4(0,0,0,0);
   if(by<p.H/4&&bx<p.W/4)a=*reinterpret_cast<const uint4*>(p.input+(by*(p.W/4)+bx)*8192+sl*512+l*16);
   unsigned words[4]={a.x,a.y,a.z,a.w};
#pragma unroll
   for(int r=0;r<4;r++) {
#pragma unroll
    for(int b=0;b<4;b++)s16v_scA[4*t+16*(r>>1)+b][((block*16+g+8*(r&1))&7)*8+((block*16+g+8*(r&1))>>3)]=s16v_decode(words[r]>>(8*b));
   }
  }
  if(th<4*Cols) {
   int idx=th;const uint4* src=reinterpret_cast<const uint4*>(p.weights+sl*N*32+cTile*256*32+cp*(4*Cols)*32+idx*32);
   uint4 a=src[0],b=src[1];unsigned words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
   for(int r=0;r<8;r++) {
    int off=idx*32+r*4,chunk=off>>9,fl=(off>>4)&31,rr=(off>>2)&3,n=chunk*16+8*(rr>>1)+(fl>>2),k=4*(fl&3)+16*(rr&1);
#pragma unroll
    for(int b=0;b<4;b++)s16v_scB[k+b][n]=s16v_decode(words[r]>>(8*b));
   }
  }
  __syncthreads();
#pragma unroll 4
  for(int k=0;k<32;k++) {
   const float4* ap=reinterpret_cast<const float4*>(&s16v_scA[k][(l>>2)*8+4*(Warps==8?(w>>2):0)]);
   float4 a0=ap[0];
   float av[Rows];
   av[0]=a0.x;av[1]=a0.y;av[2]=a0.z;av[3]=a0.w;
   if(Warps==4) {float4 a1=ap[1];av[4]=a1.x;av[5]=a1.y;av[6]=a1.z;av[7]=a1.w;}
   float bv[J];
#pragma unroll
   for(int q=0;q<J/4;q++) {
    float4 v=*reinterpret_cast<const float4*>(&s16v_scB[k][(w&3)*Cols+(l&3)*J+q*4]);
    bv[q*4]=v.x;bv[q*4+1]=v.y;bv[q*4+2]=v.z;bv[q*4+3]=v.w;
   }
#pragma unroll
   for(int i=0;i<Rows;i++) {
#pragma unroll
    for(int j=0;j<J;j++)acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
   }
  }

#pragma unroll
  for(int i=0;i<Rows;i++) {
#pragma unroll
   for(int j=0;j<J;j++)acc[i][j]=s16v_half(acc[i][j]);
  }
 }
#pragma unroll
 for(int i=0;i<Rows;i++) {
  int tok=(l>>2)+8*i+32*(Warps==8?(w>>2):0),m=tok>>4,r=tok&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
  if(bx<p.W/4&&by<p.H/4) {
#pragma unroll
   for(int j=0;j<J;j++) {
    int c=cTile*256+cp*(4*Cols)+(w&3)*Cols+(l&3)*J+j,at=(by*(p.W/4)+bx)*16*N+s16v_aoff(r,s16v_stored(c));
    int X=bx*4+(r&3),Y=by*4+((r>>2)&1)+2*(r>>3);
    int planar=((c/16*p.H+Y)*p.W+X)*16+(s16v_stored(c)&15);
    float v=acc[i][j];
    p.output[OutputView?planar:at]=s16v_encode(v);
   }
  }
 }
 if(Pool) {
#pragma unroll
  for(int i=0;i<Rows;i++) {
   int tok=(l>>2)+8*i,m=tok>>4,r=tok&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
   int X=(bx*4+(r&3))/2,Y=(by*4+((r>>2)&1)+2*(r>>3))/2;
#pragma unroll
   for(int j=0;j<J;j++) {
    float a=acc[i][j],b=__shfl_xor_sync(0xffffffff,a,4),c=__shfl_xor_sync(0xffffffff,a,16),d=__shfl_xor_sync(0xffffffff,a,20);
    float v=s16v_half(s16v_half(s16v_half(a+b)+s16v_half(c+d))*.25f);
    if((tok&5)==0&&X<lowW&&Y<lowH) {
     int chan=cTile*256+cp*(4*Cols)+(w&3)*Cols+(l&3)*J+j,lr=(X&3)+4*(Y&1)+8*((Y>>1)&1);
     int at=((Y/4)*(lowW/4)+X/4)*8192+s16v_aoff(lr,s16v_stored(chan));
     low[at]=s16v_encode(v);
    }
   }
  }
 }
 }
 __threadfence();__syncthreads();
 if(th==0&&p.signal)p.signal[(int)blockIdx.y*2*nx+blockIdx.x]=0;
}
extern "C" __global__ void __launch_bounds__(128) cc_split_swin_16h_ffwd_proj_512_chained_fp8(s16v_SplitConvParams p) {s16v_SplitConv<4,2>(p);}
extern "C" __global__ void __launch_bounds__(256) cc_split_swin_16h_proj_512_chained_fp8(s16v_SplitConvParams p) {s16v_SplitConv<8,4>(p);}

extern "C" __global__ void __launch_bounds__(128) cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8(s16v_SplitConvParams p) {s16v_SplitConv<4,2,true>(p);}
extern "C" __global__ void __launch_bounds__(128) cc_split_swin_16h_proj_512_outview_wait_fp8(s16v_SplitConvParams p) {s16v_SplitConv<4,4,false,true>(p);}

struct __align__(8) s16v_HeadParams {const unsigned char* input;unsigned char* output;const unsigned char* weights;const int* wait;int H,W;};
static_assert(sizeof(s16v_HeadParams)==40,"params");
extern "C" __global__ void __launch_bounds__(256) cc_split_swin_16h_final_head_512_wait_fp8(s16v_HeadParams p) {
 s16v_SplitConvParams q={p.input,nullptr,p.output,p.weights,p.H,p.W,nullptr,nullptr,p.wait,0,0};s16v_SplitConv<8,2,false,false,1024,false,4,false,64>(q);
}
struct __align__(8) s16v_PoolParams {const unsigned char* input;const unsigned char* residual;unsigned char* high;unsigned char* low;const unsigned char* weights;const int* wait;int* signal;int x0,y0,H,W,lowH,lowW;};
static_assert(sizeof(s16v_PoolParams)==80,"params");
extern "C" __global__ void __launch_bounds__(128) cc_split_swin_16h_proj_pool_512_chained_fp8(s16v_PoolParams p) {
 s16v_SplitConvParams q={p.input,p.residual,p.high,p.weights,p.H,p.W,nullptr,p.signal,p.wait,p.x0,p.y0};s16v_SplitConv<4,4,false,false,512,true,8,true,32>(q,p.low,p.lowH,p.lowW);
}

