"""CPU-only source regeneration. Every historical PTX output is hash checked."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess
import build_pascal_kernels as builder
import historical_ptx
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def compile_source(cuda,vcvars,source,output,flags=()):
    output.parent.mkdir(parents=True,exist_ok=True)
    command=[str(cuda/'bin/nvcc.exe'),'-ptx','-arch=sm_61','-O3','-Wno-deprecated-gpu-targets',*flags,str(source),'-o',str(output)]
    # Batch metacharacters are not safe in a toolchain/build path.
    if any(any(c in str(x) for c in '&|<>^%!\r\n"') for x in [vcvars,*command]):
        raise ValueError('unsupported batch metacharacter in build paths')
    batch=output.with_suffix('.cmd')
    batch.write_text('@echo off\ncall "'+str(vcvars)+'" >nul 2>nul\n'+' '.join('"'+x+'"' for x in command)+'\nexit /b %errorlevel%\n')
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(['cmd','/d','/c',str(batch)],stdout=log,stderr=subprocess.STDOUT,check=True)
def entries(s):
    result={}
    for m in re.finditer(r'\.visible\s+\.entry\s+(\w+)\(',s):
        a=s.index('{',m.end());z=a+1;depth=1
        while depth:depth+=(s[z]=='{')-(s[z]=='}');z+=1
        result[m[1]]=(m.start(),z)
    return result
def replace(s,new,names):
    found=entries(new)
    for name,(a,z) in reversed(list(entries(s).items())):
        if name in names:
            c,d=found[name];s=s[:a]+new[c:d]+s[z:]
    return s
def apply_recipe(sources,recipe):
    lines={name:text.splitlines(keepends=True) for name,text in sources.items()}
    for name,digest in recipe['sources'].items():
        if hashlib.sha256(sources[name].encode('ascii')).hexdigest()!=digest:
            raise ValueError('prepost generated base differs: '+name)
    out=[]
    for op in recipe['ops']:
        if 'add' in op:out.append(op['add'])
        elif 'copyText' in op:
            text=sources[op['copyText']];a=op['offset'];z=a+op['count']
            if a<0 or z>len(text) or z<a:raise ValueError('recipe text span outside generated base')
            out.append(text[a:z])
        else:
            text=lines[op['copy']];a=op['line'];z=a+op['count']
            if a<0 or z>len(text) or z<a:raise ValueError('recipe line span outside generated base')
            out.extend(text[a:z])
    return ''.join(out)
def regenerate(official,cuda12,cuda13,vcvars,out):
    out.mkdir(parents=True,exist_ok=False)
    compiled={}
    for job in json.loads((ROOT/'compiles.json').read_text()):
        label=job['label'];src=ROOT/job['source'];folder=out/'compile'/label
        shutil.copytree(src.parent,folder)
        ptx=folder/(src.stem+'.ptx')
        compile_source(cuda12,vcvars,folder/src.name,ptx,job['flags'])
        compiled[label]=ptx
        print('compiled '+label,flush=True)
    paths={
      'lower/pascal_emulation.ptx':compiled['emu'],
      'hillclimb/lanes/vitfuse/E26_qkv.sm_61.ptx':compiled['E26'],
      'hillclimb/lanes/vitfuse/E27_projection.sm_61.ptx':compiled['E27'],
      'hillclimb/lanes/vitfuse/E29_ffwd.sm_61.ptx':compiled['E29'],
      'hillclimb/lanes/vitfuse/baseline/r9_attention_ready.sm_61.ptx':compiled['r9']}
    s=historical_ptx.combo_half(compiled['half_control'].read_text(),
      compiled['half_pair'].read_text(),compiled['half_bank'].read_text())
    path=out/'combo_half.ptx';path.write_text(s,encoding='ascii',newline='\r\n')
    paths['hillclimb/lanes/swin1hfam/round8/combo_half.sm_61.ptx']=path
    stages=historical_ptx.vit_stages({label:path.read_text() for label,path in compiled.items()})
    history=out/'historical';history.mkdir()
    for label,s in stages.items():
        path=history/(label+'.ptx');path.write_text(s,encoding='ascii',newline='\r\n')
    paths['hillclimb/lanes/vitfuse/baseline/r5_selected.sm_61.ptx']=history/'r5_selected.ptx'
    s=replace(compiled['splitconv'].read_text(),compiled['splitconv_tile'].read_text(),
      ['cc_split_swin_16h_proj_512_chained_fp8','cc_split_swin_16h_proj_pool_512_chained_fp8'])
    s=replace(s,compiled['E36'].read_text(),['cc_split_swin_16h_ffwd_proj_512_chained_fp8'])
    path=out/'E36_selected.ptx';path.write_text(s,encoding='ascii',newline='\r\n')
    paths['hillclimb/lanes/vitfuse/E36_selected.sm_61.ptx']=path
    original=official.read_bytes()
    if builder.ada.sha256(original)!=builder.ada.OFFICIAL_SHA256:raise ValueError('original DLL hash differs')
    image=builder.ada.Image(original);builder.ada.check_sites(image)
    lowered=builder.module_ptx(cuda13,0,builder.ada.fatbin_at(image,builder.ada.BUNDLES[0][3]),out)
    base=builder.pascal_lowering.lower(lowered,compiled['emu'].read_text())[0]
    pool=compiled['pool'].read_text();bank=compiled['bankpairs'].read_text()
    # Pin the historical internal-linkage name of TRP's own SiLU table. NVCC's
    # generated identifier varies with the source directory, not table content.
    lut='__nv_static_37__b769b0dc_15_core_r7_pool_cu_932e54db_prepost_r6_silu_lut'
    pool=re.sub(r'\b\w*prepost_r6_silu_lut\b',lut,pool)
    bank=re.sub(r'\b\w*prepost_r6_silu_lut\b',lut,bank)
    sources=dict(dll=base,dll_renamed=base.replace('trp_','prepost_trp_'),
      emu=compiled['emu'].read_text().replace('trp_','prepost_trp_'),pool=pool,bank=bank,dual=re.sub(r'\b\w*prepost_r6_silu_lut\b','__nv_static_36__d0c3fd3e_14_dual_device_cu_932e54db_prepost_r6_silu_lut',compiled['dual'].read_text()))
    s=apply_recipe(sources,json.loads((ROOT/'prepost-transform.json').read_text()))
    path=out/'combo_prepost.ptx';path.write_text(s,encoding='ascii',newline='\r\n')
    paths['hillclimb/lanes/swin1hfam/round8/combo_prepost.sm_61.ptx']=path
    lock=json.loads((ROOT/'ptx-lock.json').read_text());results=[]
    if paths.keys()!=lock.keys():raise ValueError('regeneration does not cover exactly the locked inputs')
    for rel,path in paths.items():
        actual=sha(path);expected=lock[rel]
        results.append(dict(input=rel,expected=expected,actual=actual,matched=actual==expected))
        if actual!=expected:raise ValueError('regeneration hash differs: '+rel)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    return paths
