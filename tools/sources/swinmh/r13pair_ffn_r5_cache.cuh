#ifndef SWINMH_FFN_WARPS
#define SWINMH_FFN_WARPS 1
#endif
#ifndef SWINMH_FFN_UNROLL
#define SWINMH_FFN_UNROLL 32
#endif

// WH warps own a head; head waves retain the stock block and occupancy.
__device__ __forceinline__ void swinmh_ffn_r4(const unsigned char* x,unsigned char* out,const unsigned char* w,float* as,float* bs,bool cached) {
    constexpr int H=swinmh_H,C=swinmh_C,WH=SWINMH_FFN_WARPS,T=16*WH,NH=H/WH,U=4*C*C;
    int l=threadIdx.x,warp=(threadIdx.y%SWINMH_HEADS),tid=warp*32+l;
    int t0=((warp%WH)*4+(l>>3))*4,c0=(l&7)*4,localh=warp/WH;
    for(int batch=0;batch<64/T;batch++)for(int pass=0;pass<WH;pass++) {
        int h=pass*NH+localh;
        float mid[4][4]={};
        for(int b=0;b<4;b++) {
            float up[4][4]={};
            for(int sl=0;sl<H;sl++) {
                __syncthreads();
                for(int z=tid;z<8*(T/4);z+=32*H) {
                    int r0=(z%(T/4))*4,kp=z/(T/4);
                    unsigned p[4];
                    #pragma unroll
                    for(int i=0;i<4;i++)p[i]=*reinterpret_cast<const unsigned*>(x+((batch*T)+r0+i)*C+sl*32+kp*4);
                    #pragma unroll
                    for(int j=0;j<4;j++) {
                        int k=kp*4+j;
                        *reinterpret_cast<float4*>(as+k*T+(r0^(((k>>2)*4)&(T-1))))=make_float4(swinmh_decode(p[0]>>(8*j)),swinmh_decode(p[1]>>(8*j)),swinmh_decode(p[2]>>(8*j)),swinmh_decode(p[3]>>(8*j)));
                    }
                }
                if(threadIdx.y<SWINMH_HEADS) {
                if(threadIdx.y<SWINMH_HEADS) {
            if(cached) {
                    for(int z=tid;z<NH*256;z+=32*H) {
                        int gh=pass*NH+z/256,word=z&255,base=gh*C*128+sl*4096+b*1024;
                        float4 v=swinmh_r5_cache_load(swinmh_r5_cache_data+base+4*word);
                        *reinterpret_cast<float4*>(bs+(word/8)*(NH*32)+(z/256)*32+(word&7)*4)=v;
                    }
                } else {
                for(int z=tid;z<NH*256;z+=32*H) {
                    int gh=pass*NH+z/256,word=z&255;
                    unsigned v=__ldg(reinterpret_cast<const unsigned*>(w+gh*C*128+sl*4096+b*1024)+word);
                    int o=4*word,j=o>>9,lane=(o>>4)&31,q=(o>>2)&3;
                    int n=8*(2*j+(q>>1))+(lane>>2),k0=4*(lane&3)+16*(q&1);
                    #pragma unroll
                    for(int byte=0;byte<4;byte++) {
                        int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                        bs[k*(NH*32)+(z/256)*32+(n^swinmh_bank(k))]=swinmh_decode(v>>(8*byte));
                    }
                }
                }
                }
                }
            __syncthreads();
                #pragma unroll SWINMH_FFN_UNROLL
                for(int k=0;k<32;k++) {
                    float4 a=*reinterpret_cast<const float4*>(as+k*T+(t0^(((k>>2)*4)&(T-1))));
                    float4 v=*reinterpret_cast<const float4*>(bs+k*(NH*32)+localh*32+(c0^swinmh_bank(k)));
                    float av[4]={a.x,a.y,a.z,a.w},bv[4]={v.x,v.y,v.z,v.w};
                    #pragma unroll
                    for(int i=0;i<4;i++)for(int j=0;j<4;j++)up[i][j]=fmaf(av[i],bv[j],up[i][j]);
                }
                #pragma unroll
                for(int i=0;i<4;i++)for(int j=0;j<4;j++)up[i][j]=swinmh_half(up[i][j]);
            }
            __syncthreads();
            #pragma unroll
            for(int j=0;j<4;j++) {
                int k=c0+j;
                *reinterpret_cast<float4*>(as+(localh*32+k)*T+(t0^(((k>>2)*4)&(T-1))))=make_float4(swinmh_r5_silu(up[0][j]),swinmh_r5_silu(up[1][j]),swinmh_r5_silu(up[2][j]),swinmh_r5_silu(up[3][j]));
            }
            if(cached) {
                for(int z=tid;z<NH*256;z+=32*H) {
                    int gh=pass*NH+z/256,word=z&255,base=U+gh*4096+b*1024;
                    float4 v=swinmh_r5_cache_load(swinmh_r5_cache_data+base+4*word);
                    *reinterpret_cast<float4*>(bs+(word/8)*(NH*32)+(z/256)*32+(word&7)*4)=v;
                }
            } else {
            for(int z=tid;z<NH*256;z+=32*H) {
                int gh=pass*NH+z/256,word=z&255;
                unsigned v=__ldg(reinterpret_cast<const unsigned*>(w+U+gh*4096+b*1024)+word);
                int o=4*word,j=o>>9,lane=(o>>4)&31,q=(o>>2)&3;
                int n=8*(2*j+(q>>1))+(lane>>2),k0=4*(lane&3)+16*(q&1);
                #pragma unroll
                for(int byte=0;byte<4;byte++) {
                    int p=k0+byte,k=16*(p>>4)+2*((p&15)>>2)+(p&1)+8*((p>>1)&1);
                    bs[k*(NH*32)+(z/256)*32+(n^swinmh_bank(k))]=swinmh_decode(v>>(8*byte));
                }
            }
            }
            __syncthreads();
            #pragma unroll SWINMH_FFN_UNROLL
            for(int k=0;k<32;k++) {
                float4 a=*reinterpret_cast<const float4*>(as+(localh*32+k)*T+(t0^(((k>>2)*4)&(T-1))));
                float4 v=*reinterpret_cast<const float4*>(bs+k*(NH*32)+localh*32+(c0^swinmh_bank(k)));
                float av[4]={a.x,a.y,a.z,a.w},bv[4]={v.x,v.y,v.z,v.w};
                #pragma unroll
                for(int i=0;i<4;i++)for(int j=0;j<4;j++)mid[i][j]=fmaf(av[i],bv[j],mid[i][j]);
            }
            #pragma unroll
            for(int i=0;i<4;i++)for(int j=0;j<4;j++)mid[i][j]=swinmh_half(mid[i][j]);
        }
        #pragma unroll
        for(int i=0;i<4;i++)for(int j=0;j<4;j++)out[(batch*T+t0+i)*C+h*32+c0+j]=swinmh_encode(mid[i][j]);
    }
}

