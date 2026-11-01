"""CPU-only package pin/refusal checks, independent of any installed assets."""
from pathlib import Path
import hashlib,json,tempfile,unittest,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import inventory
class InventoryTests(unittest.TestCase):
 def test_pins_agree(self):
  root=Path(__file__).resolve().parents[1]
  self.assertIn(inventory.PACK_SHA256,(root/'src/NeuralAdmission.h').read_text())
  self.assertIn(inventory.ORIGINAL_SHA256,(root/'src/NeuralRuntimeKernels.h').read_text())
  self.assertEqual(inventory.PACK_BYTES,11574444)
 def test_tamper_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'fake.trpk';p.write_bytes(b'not the pinned pack')
   self.assertFalse(inventory.verify_pack(p))
   self.assertFalse(inventory.verify_pack(Path(d)/'missing'))
if __name__=='__main__':unittest.main()
