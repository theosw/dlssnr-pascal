"""Pinned Pascal component identities. No install, download, or GPU operations."""
from pathlib import Path
import hashlib
PACK_SHA256='5AC8546C9E08AD8D0722DE54DA5E2BFDB1976FCCC2D227B3079E739BDA2C4B43'
ORIGINAL_SHA256='2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F'
PACK_BYTES=11574444
PACK_NAME='nvngx_dlssnr_310.9.16_pascal_v1.trpk'
def verify_pack(path):
 try:
  return path.stat().st_size==PACK_BYTES and hashlib.sha256(path.read_bytes()).hexdigest().upper()==PACK_SHA256
 except OSError:return False
