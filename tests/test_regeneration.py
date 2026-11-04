"""CPU checks for shared-storage aliasing and hash-bound transform refusal."""
from pathlib import Path
import hashlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import historical_ptx as h
from regenerate import apply_recipe


def module(body, threads=128, declarations=''):
    return ('.version 8.8\n.target sm_61\n.address_size 64\n' + declarations +
        '.visible .entry fixture_expand(\n.param .align 8 .b8 fixture_expand_param_0[72]\n)\n'
        f'.maxntid {threads}, 1, 1\n' + '{\n' + body + '\n}')


class RegenerationTests(unittest.TestCase):
    def test_nested_shared_regions_stay_disjoint_and_aligned(self):
        body = '.shared .align 16 .b8 local[17];\nmov.u32 %r1, outer;\nmov.u32 %r2, local;\n$L_done:\nret;'
        aliased, extent = h.branch(body, {'outer': 16}, '_selected')
        self.assertEqual(extent, 48)
        self.assertIn('mov.u32 %r1, int8_r3_scratch;', aliased)
        self.assertIn('mov.u32 %r2, int8_r3_scratch+16;', aliased)
        self.assertNotIn('.shared', aliased)
        self.assertIn('$L_done_selected:', aliased)

    def test_scalar_shared_storage_keeps_its_extent(self):
        _, shared, text = h.parse(module('.shared .align 4 .u32 flag;\nmov.u32 %r1, flag;'))
        self.assertEqual(shared, {'flag': 4})
        aliased, extent = h.branch(h.parse(text)[0]['fixture_expand'][1], shared, '_s')
        self.assertEqual(extent, 16)
        self.assertIn('mov.u32 %r1, int8_r3_scratch;', aliased)

    def test_nested_selection_retains_fallback_and_thread_guard(self):
        generic = module('// generic body\n$L_end:\nret;')
        fast = module('// T96 body\n$L_end:\nret;', threads=256)
        selected = h.compose(generic, fast, 'expand', (96,))
        nested = h.compose(selected, module('// T240 body\n$L_end:\nret;', threads=256), 'expand', (240,))
        self.assertIn('setp.ge.u32 int8_r5_upper,int8_r5_y,4; @int8_r5_upper ret;', nested)
        self.assertIn('$L_end_r5old_r5old:', nested)
        self.assertIn('$L_end_r5new_r5old:', nested)
        self.assertIn('$L_end_r5new:', nested)
        self.assertLess(nested.index('// T240 body'), nested.index('// T96 body'))
        self.assertLess(nested.index('// T96 body'), nested.index('// generic body'))

    def test_missing_replacement_entry_refuses(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            h.replace_entries(module('ret;'), {'missing': ''})

    def test_changed_generated_base_refuses_before_copying(self):
        source = 'generated input\n'
        recipe = {'sources': {'base': hashlib.sha256(source.encode('ascii')).hexdigest()},
                  'ops': [{'copy': 'base', 'line': 0, 'count': 1}]}
        self.assertEqual(apply_recipe({'base': source}, recipe), source)
        with self.assertRaisesRegex(ValueError, 'generated base differs'):
            apply_recipe({'base': source + 'changed\n'}, recipe)

    def test_out_of_bounds_transform_refuses(self):
        source = 'generated input\n'
        for op in ({'copy': 'base', 'line': 0, 'count': 2},
                   {'copyText': 'base', 'offset': -1, 'count': 2}):
            with self.subTest(op=op), self.assertRaisesRegex(ValueError, 'outside generated base'):
                apply_recipe({'base': source}, {'sources': {}, 'ops': [op]})


if __name__ == '__main__':
    unittest.main()
