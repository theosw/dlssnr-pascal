#pragma once
namespace prepost_fourwarp {
__device__ __forceinline__ void prepost_r7_input(__half* state,const unsigned char* blob,const unsigned char* input,const unsigned char* skip,int H,int W,int x0,int y0,Scratch& s) {
    if(threadIdx.x<64)reinterpret_cast<float*>(s.weights)[(threadIdx.x&127)]=__half2float(reinterpret_cast<const __half*>(blob+8272)[(threadIdx.x&127)]);
    __syncthreads();
    int tok=(threadIdx.x&127)>>1,hh=(threadIdx.x&127)&1,x,y;Pixel(tok,x,y);
    int wx=8*(2*blockIdx.x+(threadIdx.x>>7))+x0,wy=8*blockIdx.y+y0;
    int gx=wx+x,gy=wy+y,hx=wx/2+x/2,hy=wy/2+y/2;
    int row=(gx&3)+4*(gy&1)+8*((gy>>1)&1),g=row&7,rb=row>>3;
    bool iv=hx>=0&&hx<W/2&&hy>=0&&hy<H/2;
    bool sv=gx>=0&&gx<W&&gy>=0&&gy<H;
    #pragma unroll
    for(int st=0;st<4;st++) {
        unsigned a=iv?__ldg(reinterpret_cast<const unsigned*>(input+((size_t(hh)*(H/2)+hy)*(W/2)+hx)*16)+st):0;
        unsigned b=sv?__ldg(reinterpret_cast<const unsigned*>(skip+(size_t(gy/4)*((W+3)/4)+gx/4)*512)+(4*g+st)*4+2*hh+rb):0;
        #pragma unroll
        for(int e=0;e<4;e++) {
            int c=16*hh+Channel16(4*st+e);
            float av=Half(E4M3(a>>(8*e))*reinterpret_cast<const float*>(s.weights)[c]);
            float bv=Half(E4M3(b>>(8*e))*reinterpret_cast<const float*>(s.weights)[32+c]);
            prepost_state_write(state,tok*32+c,Half(av+bv));
        }
    }
    __syncthreads();
}
}
