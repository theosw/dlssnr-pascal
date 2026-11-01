// NR 310.9.16 split16 projections: scaled residual initializes f16 MMA accumulation.
// Completion flags cover 256 output channels per 8x8 window; pooling publishes the corresponding 4x4 low-resolution tile.
#include "vitfuse_common.cuh"
struct __align__(8) vitfuse_SplitConvParams {const unsigned char* input;const unsigned char* residual;unsigned char* output;const unsigned char* weights;int H,W;void* unused;int* signal;const int* wait;int x0,y0;};
static_assert(sizeof(vitfuse_SplitConvParams)==72,"params");
template<int Warps,int Groups,bool ResidualView=false,bool OutputView=false,int N=512,bool Residual=true,int WaitTile=8,bool Pool=false,int Cols=32>
__device__ __forceinline__ void vitfuse_SplitConv(const vitfuse_SplitConvParams& p,unsigned char* low=nullptr,int lowH=0,int lowW=0) {
 __shared__ float vitfuse_scA[32][64];__shared__ float vitfuse_scB[32][4*Cols];
 const int l=threadIdx.x,w=threadIdx.y,th=w*32+l,nx=(p.W+7)/8,cTile=(blockIdx.x/nx)/2,part=(blockIdx.x/nx)&1,wx=(blockIdx.x%nx)*8,wy=blockIdx.y*8;
 if(th==0) {
  int ix=(p.W-min(p.x0,0)+WaitTile-1)/WaitTile;
  for(int y=(wy-p.y0)/WaitTile;y<=(min(wy+7,p.H-1)-p.y0)/WaitTile;y++)for(int x=(wx-p.x0)/WaitTile;x<=(min(wx+7,p.W-1)-p.x0)/WaitTile;x++)for(int c=0;c<Groups;c++)vitfuse_wait(p.wait+(y*Groups+c)*ix+x);
  __threadfence();
 }
 __syncthreads();
 for(int cp=part;cp<part+1;cp++) {
 constexpr int Rows=8/Warps;
 float acc[Rows][Cols];
#pragma unroll
 for(int i=0;i<Rows;i++) {
#pragma unroll
  for(int j=0;j<Cols;j++) {
   int tok=l+32*(Warps==8?(w>>2):i),m=tok>>4,r=tok&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
   int c=cTile*256+cp*(4*Cols)+(w&3)*Cols+j,at=(by*(p.W/4)+bx)*16*N+vitfuse_aoff(r,vitfuse_stored(c));
   int X=bx*4+(r&3),Y=by*4+((r>>2)&1)+2*(r>>3),planar=((c/16*p.H+Y)*p.W+X)*16+(vitfuse_stored(c)&15);
   float v=0;
   if(Residual&&bx<p.W/4&&by<p.H/4) {float g=__half2float(reinterpret_cast<const __half*>(p.weights+512*N)[c]);v=vitfuse_half(vitfuse_decode(p.residual[ResidualView?planar:at])*g);}
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
    for(int b=0;b<4;b++)vitfuse_scA[4*t+16*(r>>1)+b][block*16+g+8*(r&1)]=vitfuse_decode(words[r]>>(8*b));
   }
  }
  if(th<4*Cols) {
   int idx=th;const uint4* src=reinterpret_cast<const uint4*>(p.weights+sl*N*32+cTile*256*32+cp*(4*Cols)*32+idx*32);
   uint4 a=src[0],b=src[1];unsigned words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
   for(int r=0;r<8;r++) {
    int off=idx*32+r*4,chunk=off>>9,fl=(off>>4)&31,rr=(off>>2)&3,n=chunk*16+8*(rr>>1)+(fl>>2),k=4*(fl&3)+16*(rr&1);
#pragma unroll
    for(int b=0;b<4;b++)vitfuse_scB[k+b][n]=vitfuse_decode(words[r]>>(8*b));
   }
  }
  __syncthreads();
#pragma unroll 4
  for(int k=0;k<32;k++) {
#pragma unroll
   for(int i=0;i<Rows;i++) {
    float a=vitfuse_scA[k][l+32*(Warps==8?(w>>2):i)];
#pragma unroll
    for(int j=0;j<Cols;j++)acc[i][j]=fmaf(a,vitfuse_scB[k][(w&3)*Cols+j],acc[i][j]);
   }
  }
#pragma unroll
  for(int i=0;i<Rows;i++) {
#pragma unroll
   for(int j=0;j<Cols;j++)acc[i][j]=vitfuse_half(acc[i][j]);
  }
 }
#pragma unroll
 for(int i=0;i<Rows;i++) {
  int tok=l+32*(Warps==8?(w>>2):i),m=tok>>4,r=tok&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
  if(bx<p.W/4&&by<p.H/4) {
#pragma unroll
   for(int j=0;j<Cols;j++) {
    int c=cTile*256+cp*(4*Cols)+(w&3)*Cols+j,at=(by*(p.W/4)+bx)*16*N+vitfuse_aoff(r,vitfuse_stored(c));
    int X=bx*4+(r&3),Y=by*4+((r>>2)&1)+2*(r>>3);
    int planar=((c/16*p.H+Y)*p.W+X)*16+(vitfuse_stored(c)&15);
    float v=acc[i][j];
    p.output[OutputView?planar:at]=vitfuse_encode(v);
   }
  }
 }
 if(Pool) {
#pragma unroll
  for(int i=0;i<Rows;i++) {
   int tok=l+32*i,m=tok>>4,r=tok&15,bx=wx/4+(m&1),by=wy/4+(m>>1);
   int X=(bx*4+(r&3))/2,Y=(by*4+((r>>2)&1)+2*(r>>3))/2;
#pragma unroll
   for(int j=0;j<Cols;j++) {
    float a=acc[i][j],b=__shfl_xor_sync(0xffffffff,a,1),c=__shfl_xor_sync(0xffffffff,a,4),d=__shfl_xor_sync(0xffffffff,a,5);
    float v=vitfuse_half(vitfuse_half(vitfuse_half(a+b)+vitfuse_half(c+d))*.25f);
    if((l&5)==0&&X<lowW&&Y<lowH) {
     int chan=cTile*256+cp*(4*Cols)+(w&3)*Cols+j,lr=(X&3)+4*(Y&1)+8*((Y>>1)&1);
     int at=((Y/4)*(lowW/4)+X/4)*8192+vitfuse_aoff(lr,vitfuse_stored(chan));
     low[at]=vitfuse_encode(v);
    }
   }
  }
 }
 }
 __threadfence();__syncthreads();
 if(th==0&&p.signal) {
  int* flag=p.signal+((int)blockIdx.y*2+cTile)*nx+blockIdx.x%nx;
  // Two finite producers decrement -1 -> -2 -> -3; only the last publishes zero.
  if(atomicSub(flag,1)==-2) {__threadfence();*reinterpret_cast<volatile int*>(flag)=0;}
 }
}
extern "C" __global__ void __launch_bounds__(128) cc_split_swin_16h_ffwd_proj_512_chained_fp8(vitfuse_SplitConvParams p) {vitfuse_SplitConv<4,2>(p);}
