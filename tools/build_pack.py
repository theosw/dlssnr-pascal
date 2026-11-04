"""Rebuild the byte-identical Pascal pack from sources and the original DLL.
No download, installation, runtime loading, or GPU dispatch is performed.
"""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import regenerate as regen
from prepare_native import transform
from inventory import PACK_SHA256,ORIGINAL_SHA256
ROOT=Path(__file__).resolve().parent

def merge_ptx(paths):
    globals_seen={};bodies=[]
    for path in paths:
        body=path.read_text().split('.address_size 64',1)[1]
        def decl(m):
            name=re.search(r'\b(\w+)\s*(?:\[\d+\])?\s*(?:=|;)',m[0])[1]
            if name in globals_seen:
                if globals_seen[name]!=m[0]:raise ValueError('conflicting global '+name)
                return ''
            globals_seen[name]=m[0];return m[0]
        bodies.append(re.sub(r'^\.global\b[^;]+;',decl,body,flags=re.M))
    return '.version 8.8\n.target sm_61\n.address_size 64\n'+'\n'.join(bodies)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('official','cuda12','cuda13','vcvars','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--generate-only',action='store_true',help='regenerate all nine locked inputs without building a pack')
    a=p.parse_args()
    if regen.sha(a.official).upper()!=ORIGINAL_SHA256:p.error('original DLL hash differs')
    for root,name,version in [(a.cuda12,'nvcc','V12.9.86'),(a.cuda12,'ptxas','V12.9.86'),(a.cuda12,'fatbinary','V12.9.86'),(a.cuda13,'cuobjdump','V13.4.92')]:
        text=subprocess.run([str(root/'bin'/(name+'.exe')),'--version'],capture_output=True,text=True,check=True).stdout
        if version not in text:p.error(name+' version differs from '+version)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=regen.regenerate(a.official,a.cuda12,a.cuda13,a.vcvars,out/'regenerated')
    if a.generate_only:return
    native=out/'native';shutil.copytree(ROOT/'sources/swinmh',native)
    shutil.copyfile(ROOT/'sm_general_slots.cuh',native/'sm_general_slots.cuh')
    for h in (2,4,8):
        for v in range(4):
            stem=f'r13linearh2v{v}' if h==2 else f'r11vectorcomboh{h}v{v}'
            cu=native/f'h{h}v{v}.cu';cu.write_text(transform((native/(stem+'.cu')).read_text(),h),encoding='ascii',newline='\n')
            flags=['-std=c++17',f'-DSWINMH_HEADS={h}',f'-DSWINMH_VARIANT={v}']
            if h!=2 or v not in (1,3):flags.append(f'-DSWINMH_FFN_WARPS={2 if h==2 else 4}')
            ptx=cu.with_suffix('.ptx');regen.compile_source(a.cuda12,a.vcvars,cu,ptx,flags)
            text=re.sub(r'(\.maxntid[^\n]*\n)',r'\1.maxnreg 128\n',ptx.read_text())
            text=re.sub(r'ld\.global\.u64(\s+[^\n]+\[swinmh_r(?:5_cache|7)_key_\d+\])',r'ld.global.cg.u64\1',text)
            ptx.write_text(text,encoding='ascii',newline='\n')
            print(f'compiled H{h} phase {v}',flush=True)
        (native/f'family{h}.ptx').write_text(merge_ptx([native/f'h{h}v{v}.ptx' for v in range(4)]),encoding='ascii',newline='\n')
    selections=[]
    for row in json.loads((ROOT/'selection.json').read_text()):
        bundle=row['bundle'];path=native/f'family{2**bundle}.ptx' if bundle in (1,2,3) else inputs[row['path']]
        selections+=['--native',f'{bundle}={path}']
    launches=[]
    for line in (ROOT/'launch-rules.txt').read_text().splitlines():
        if line.strip():launches+=['--launch',line.strip()]
    pack=out/'sm-general-pascal.trpk'
    command=[sys.executable,'-I','-B',str(ROOT/'build_pascal_kernels.py'),'--official',str(a.official),'--cuda12',str(a.cuda12),'--cuda13',str(a.cuda13),'--vcvars',str(a.vcvars),'--emulation-ptx',str(inputs['lower/pascal_emulation.ptx']),*selections,*launches,'--architectures','61','--work-dir',str(out/'pack-work'),'--output',str(pack)]
    with (out/'pack-build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    actual=regen.sha(pack).upper()
    (out/'acceptance.json').write_text(json.dumps(dict(complete=True,externalInputs=[],produced=actual,shipped=PACK_SHA256,matched=actual==PACK_SHA256),indent=2)+'\n')
    if actual!=PACK_SHA256:raise ValueError('whole-pack identity differs: '+actual)
    print('byte-identical pack '+actual+' (all inputs regenerated)',flush=True)
if __name__=='__main__':main()
