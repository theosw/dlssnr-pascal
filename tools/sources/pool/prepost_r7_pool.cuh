#pragma once
extern "C" __device__ __noinline__ void prepost_publish_pre(const __half* state,unsigned char* output,int H,int W,unsigned char* pool,int halfH,int halfW) {
    using namespace prepost_fourwarp;
    int tok=threadIdx.x>>1,hh=threadIdx.x&1,m=tok>>4,row=tok&15,g=row&7,rb=row>>3;
    int bx=2*blockIdx.x+(m&1),by=2*blockIdx.y+(m>>1);
    if(bx<((W+3)/4)&&by<((H+3)/4)) {
        unsigned* chunk=reinterpret_cast<unsigned*>(output+(by*((W+3)/4)+bx)*512);
        #pragma unroll
        for(int st=0;st<4;st++) {
            unsigned w=0;
            #pragma unroll
            for(int e=0;e<4;e++){int c=16*hh+Channel16(4*st+e);w|=EncodeE4M3(prepost_state_read(state,tok*32+c))<<(8*e);}
            chunk[(4*g+st)*4+2*hh+rb]=w;
        }
    }
    if(pool)for(int i=threadIdx.x;i<512;i+=128) {
        int pix=i/32,c=i&31,x=2*(pix&3),y=2*(pix>>2);
        int t=16*((x>>2)+2*(y>>2))+(x&3)+4*(y&1)+8*((y>>1)&1);
        float a=prepost_state_read(state,t*32+c),b=prepost_state_read(state,(t+1)*32+c);
        float d=prepost_state_read(state,(t+4)*32+c),e=prepost_state_read(state,(t+5)*32+c);
        float v=Half(Half(Half(a+b)+Half(d+e))*.25f);
        int ox=4*blockIdx.x+(pix&3),oy=4*blockIdx.y+(pix>>2);
        int stored=4*((c&7)>>1)+(c&1)+2*((c>>3)&1);
        if(ox<halfW&&oy<halfH)pool[((size_t(c>>4)*halfH+oy)*halfW+ox)*16+stored]=EncodeE4M3(v);
    }
}
