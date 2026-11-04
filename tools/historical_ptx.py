"""TRP-authored historical selection/dispatch transforms, with no kernel bodies.

All instruction bodies come from freshly compiled TRP CUDA sources. These are
the CPU-only parts of the half-family round4/round8 and ViT round5/round7 /
integer round4/round5 research generators. Formatting is part of the lock.
"""
import re


def parse(text):
    text = re.sub(
        r'\.shared\s+\.align\s+(\d+)\s+\.[ubsf](\d+)\s+(\w+)\s*;',
        lambda m: f'.shared .align {m[1]} .b8 {m[3]}[{int(m[2]) // 8}];', text)
    entries = {}
    for m in re.finditer(r'\.visible\s+\.entry\s+(\w+)\(', text):
        a = text.index('{', m.end())
        b, depth = a + 1, 1
        while depth:
            depth += (text[b] == '{') - (text[b] == '}')
            b += 1
        entries[m[1]] = (text[m.start():a], text[a + 1:b - 1])
    shared = {n: int(sz) for n, sz in re.findall(
        r'\.shared\s+\.align\s+\d+\s+\.b8\s+(\w+)\[(\d+)\];', text)}
    return entries, shared, text


def replace_entries(text, replacements):
    entries, _, text = parse(text)
    for name, (header, body) in entries.items():
        if name in replacements:
            text = text.replace(header + '{' + body + '}', replacements[name])
    if replacements.keys() - entries.keys():
        raise ValueError('replacement entry absent from generated base')
    return text


def branch(body, symbols, suffix):
    symbols = dict(symbols)
    symbols.update((n, int(sz)) for n, sz in re.findall(
        r'\.shared\s+\.align\s+\d+\s+\.b8\s+(\w+)\[(\d+)\];', body))
    body = re.sub(r'\s*\.shared\s+\.align\s+\d+\s+\.b8\s+\w+\[\d+\];', '', body)
    used = [n for n in symbols if re.search(r'\b' + n + r'\b', body)]
    offset = 0
    for name in used:
        body = re.sub(r'\b' + name + r'\b',
                      'int8_r3_scratch' + ('+' + str(offset) if offset else ''), body)
        offset = (offset + symbols[name] + 15) & ~15
    return re.sub(r'\$L\w+', lambda m: m[0] + suffix, body), offset


def _s16_branch(body, symbols, suffix):
    body = re.sub(r'\s*\.shared \.align 16 \.b8 \w+\[\d+\];', '', body)
    used = [name for name in symbols if re.search(r'\b' + name + r'\b', body)]
    if len(used) != 2 or any(not n.endswith(('2As', '2Bs')) for n in used):
        raise ValueError('expected two generated shared arrays')
    for name in used:
        body = body.replace(name, 's16v_r5_' + ('A' if name.endswith('2As') else 'B'))
    return re.sub(r'\$L\w+', lambda m: m[0] + suffix, body)


def _s16_dispatch(fast, generic):
    fe, fs, _ = parse(fast)
    ge, gs, _ = parse(generic)
    if fe.keys() != ge.keys():
        raise ValueError('historical ViT entry sets differ')
    out = ['// Round 5: exact frozen r4 at T=288; frozen r3 at every other T.\n'
           '// Shared A/B storage aliases across mutually exclusive paths.\n'
           '.version 8.8\n.target sm_61\n.address_size 64\n']
    for name, (header, fb) in fe.items():
        off = 72 if 'qkv' in name else 64
        bs = 12288 if 'qkv' in name else 16384
        out.append(header + '{\n'
            '.shared .align 16 .b8 s16v_r5_A[16384];\n'
            f'.shared .align 16 .b8 s16v_r5_B[{bs}];\n'
            '.reg .u32 s16v_r5_h, s16v_r5_w, s16v_r5_T;\n.reg .pred s16v_r5_generic;\n'
            f'ld.param.u32 s16v_r5_h, [{name}_param_0+{off}];\n'
            f'ld.param.u32 s16v_r5_w, [{name}_param_0+{off + 4}];\n'
            'mul.lo.u32 s16v_r5_T, s16v_r5_h, s16v_r5_w;\n'
            'setp.ne.u32 s16v_r5_generic, s16v_r5_T, 288;\n'
            '@s16v_r5_generic bra.uni s16v_r5_fallback;\n{\n'
            + _s16_branch(fb, fs, '_r4') + '\n}\ns16v_r5_fallback:\n{\n'
            + _s16_branch(ge[name][1], gs, '_r3') + '\n}\n}\n')
    return '\n'.join(out)


def _s16_fixed(dispatch, expand):
    entries, _, text = parse(dispatch)
    for name, (header, body) in reversed(list(entries.items())):
        if 'ffn_expand' in name:
            text = text.replace(header + '{' + body + '}', '')
    extra = re.sub(r'^\.(version|target|address_size)[^\n]*\n', '', expand, flags=re.M)
    symbols = re.findall(r'\.shared \.align 16 \.b8 (\w+)\[\d+\];', extra)
    for i, symbol in enumerate(dict.fromkeys(symbols)):
        extra = extra.replace(symbol, 's16v_r5_expand_shared_' + str(i))
    return text + '\n' + extra


def _s16_96(text):
    replacements = {}
    for name, (header, body) in parse(text)[0].items():
        if 'ffn_expand' in name:
            continue
        old = 'setp.ne.u32 s16v_r5_generic, s16v_r5_T, 288;'
        if body.count(old) != 1:
            raise ValueError('historical ViT dispatch guard differs')
        body = body.replace(old, '.reg .pred s16v_r7_other;\n' + old +
            '\nsetp.ne.u32 s16v_r7_other, s16v_r5_T, 96;\n'
            'and.pred s16v_r5_generic, s16v_r5_generic, s16v_r7_other;')
        replacements[name] = header + '{' + body + '}'
    return replace_entries(text, replacements)


def _s16_q128(text, qkv):
    qe = parse(qkv)[0]
    replacements = {}
    for name, (header, body) in parse(text)[0].items():
        if 'qkv' not in name:
            continue
        extra = qe[name][1]
        symbols = re.findall(r'\.shared \.align 16 \.b8 (\w+)\[\d+\];', extra)
        if len(symbols) != 2:
            raise ValueError('expected two generated QKV shared arrays')
        extra = re.sub(r'\s*\.shared \.align 16 \.b8 \w+\[\d+\];', '', extra)
        for symbol in symbols:
            extra = extra.replace(symbol, 's16v_r5_' + ('A' if symbol.endswith('2As') else 'B'))
        extra = re.sub(r'\$L\w+', lambda m: m[0] + '_r7q', extra)
        part = header + '{' + body + '}'
        part = part[:part.index('s16v_r5_fallback:')] + 's16v_r5_fallback:\n{\n' + extra + '\n}\n}\n'
        replacements[name] = part
    return replace_entries(text, replacements)


def _r4_contract(generic, fast, fallback):
    ge, gs, gt = parse(generic)
    fe, fs, ft = parse(fast)
    we, ws, _ = parse(fallback)
    out = ['// int8 round4, c26 FP32 fallback.\n.version 8.8\n.target sm_61\n.address_size 64\n']
    used = []
    for name, (gh, gb) in ge.items():
        if 'contract' not in name:
            out.append(gh + '{' + gb + '}')
            continue
        header, fb = fe[name]
        fb, fsize = branch(fb, fs, '_r4int')
        gb, gsize = branch(we[name][1], ws, '_r4fp')
        guard = ('.reg .u32 int8_r4_h,int8_r4_w,int8_r4_T;\n'
            '.reg .pred int8_r4_match,int8_r4_one;\n'
            f'ld.param.u32 int8_r4_h,[{name}_param_0+64];\n'
            f'ld.param.u32 int8_r4_w,[{name}_param_0+68];\n'
            'mul.lo.u32 int8_r4_T,int8_r4_h,int8_r4_w;\n'
            'setp.eq.u32 int8_r4_match,int8_r4_T,96;\n')
        for tokens in (240, 288):
            guard += (f'setp.eq.u32 int8_r4_one,int8_r4_T,{tokens};\n'
                      'or.pred int8_r4_match,int8_r4_match,int8_r4_one;\n')
        guard += '@!int8_r4_match bra.uni int8_r4_fallback;\n'
        out.append(header + '{\n' + f'.shared .align 16 .b8 int8_r3_scratch[{max(fsize, gsize)}];\n'
            + guard + '{\n' + fb + '\n}\nint8_r4_fallback:\n{\n' + gb + '\n}\n}')
        used.append(fb)
    body = '\n'.join(out)
    declarations = [d for d in re.findall(r'(?m)^(?:\.visible\s+)?\.global[^;]+;', ft)
        if re.search(r'\b' + re.search(r'\.[busf]\d+\s+(\w+)', d)[1] + r'\b', '\n'.join(used))]
    out.insert(1, '\n'.join(declarations))
    outside = gt
    for gh, gb in ge.values():
        outside = outside.replace(gh + '{' + gb + '}', '')
    out.insert(1, '\n'.join(d for d in re.findall(r'(?m)^\.shared[^;]+;', outside)
        if re.search(r'\b' + re.search(r'\.[busf]\d+\s+(\w+)', d)[1] + r'\b', body)))
    return '\n'.join(out)


def compose(generic, fast, kind, active):
    ge, gs, gt = parse(generic)
    fe, fs, ft = parse(fast)
    for decl in re.findall(r'(?m)^(?:\.visible\s+)?\.global[^;]+;', ft):
        symbol = re.search(r'\.[busf]\d+\s+(\w+)', decl)[1]
        if not re.search(r'\b' + symbol + r'\b', gt):
            gt = gt.replace('.address_size 64', '.address_size 64\n' + decl, 1)
    for name, (gh, gb) in reversed(list(ge.items())):
        if kind not in name:
            continue
        fh, fb = fe[name]
        fb, fsize = branch(fb, fs, '_r5new')
        gb, gsize = branch(gb, gs, '_r5old')
        off = 72 if 'qkv' in name else 64
        pred = '\n'.join(f'setp.eq.u32 int8_r5_one,int8_r5_T,{tokens};\n'
            'or.pred int8_r5_match,int8_r5_match,int8_r5_one;' for tokens in active)
        guard = ('.reg .u32 int8_r5_h,int8_r5_w,int8_r5_T;\n.reg .pred int8_r5_match,int8_r5_one;\n'
            f'ld.param.u32 int8_r5_h,[{name}_param_0+{off}];\n'
            f'ld.param.u32 int8_r5_w,[{name}_param_0+{off + 4}];\n'
            'mul.lo.u32 int8_r5_T,int8_r5_h,int8_r5_w;\nsetp.eq.u32 int8_r5_match,0,1;\n'
            + pred + '\n@!int8_r5_match bra.uni int8_r5_fallback;\n')
        fallbackguard = ''
        if '.maxntid 256' in fh and '.maxntid 128' in gh:
            fallbackguard = ('.reg .u32 int8_r5_y; .reg .pred int8_r5_upper; mov.u32 int8_r5_y,%tid.y; '
                'setp.ge.u32 int8_r5_upper,int8_r5_y,4; @int8_r5_upper ret;\n')
        body = fh + '{\n' + f'.shared .align 16 .b8 int8_r3_scratch[{max(fsize, gsize)}];\n'
        body += guard + '{\n' + fb + '\n}\nint8_r5_fallback:\n' + fallbackguard + '{\n' + gb + '\n}\n}'
        gt = gt.replace(gh + '{' + ge[name][1] + '}', body)
    return gt


def replace_kind(generic, fast, kind):
    ge, _, gt = parse(generic)
    fe, fs, _ = parse(fast)
    for name, (gh, gb) in ge.items():
        if kind not in name:
            continue
        fh, fb = fe[name]
        fb, size = branch(fb, fs, '_r5replacement')
        gt = gt.replace(gh + '{' + gb + '}', fh + '{\n' +
            f'.shared .align 16 .b8 int8_r3_scratch[{size}];\n' + fb + '\n}')
    return gt


def vit_stages(compiled):
    stages = {}
    stages['r5_vit_dispatch'] = _s16_dispatch(compiled['vit_r4'], compiled['vit_r3'])
    stages['r5_vit_fixed'] = _s16_fixed(stages['r5_vit_dispatch'], compiled['vit_expand'])
    stages['r7_vit_96'] = _s16_96(stages['r5_vit_fixed'])
    stages['r7_vit_q128'] = _s16_q128(stages['r7_vit_96'], compiled['vit_qkv'])
    stages['r4_conly128'] = _r4_contract(stages['r7_vit_q128'], compiled['r4_c128'], compiled['r3_fallback_w8'])
    stages['r5_c96'] = compose(stages['r4_conly128'], compiled['r4_ccompact'], 'contract', (96,))
    stages['r5_ebase'] = replace_kind(stages['r5_c96'], compiled['r5_efallback'], 'expand')
    stages['r5_pre_e96'] = compose(stages['r5_ebase'], compiled['r5_efp96'], 'expand', (96,))
    stages['r5_pre_ec'] = compose(stages['r5_pre_e96'], compiled['r3_fallback_w8'], 'expand', (240,))
    stages['r5_selected'] = compose(stages['r5_pre_ec'], compiled['r5_qwhole'], 'qkv', (240,))
    return stages


def combo_half(control, paired, bank):
    def bounds(text):
        return re.sub(r'(\.visible \.entry ([^(]+)\([^)]*\)\s*\.maxntid[^\n]+)\n',
            lambda m: m[1] + '\n.maxnreg ' + str(96 if 'upsample' in m[2] else 64) + '\n', text)
    control, paired, bank = map(bounds, (control, paired, bank))
    old = '_ZZ53cc_tinlayout_fused_swin_1h_32_1_upsample_tilesync_fp8E8swin1h_s'
    bank = bank.replace('.maxnreg 96', '.maxnreg 64').replace(old, 'swin1hfam_r5_up_shared')
    replacements = {}
    for text, kinds in [(paired, ('inpview_tilesync',)), (bank, ('upsample_tilesync', 'outview_wait'))]:
        for name, (header, body) in parse(text)[0].items():
            if any(kind in name for kind in kinds):
                replacements[name] = header + '{' + body + '}'
    return replace_entries(control, replacements).replace(old, 'swin1hfam_r5_up_shared')
