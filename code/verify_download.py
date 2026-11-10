"""Check size and SHA-256 of junyi.rar before any processing."""
import hashlib, os, sys
PATH = sys.argv[1] if len(sys.argv) > 1 else 'junyi.rar'
SIZE = 1_295_551_222
SHA = '7335386710db473a475ac47f6049ab8f1031bb703e6937f2d5529b46293ff35e'
if not os.path.exists(PATH): sys.exit(f'ERROR: {PATH} not found. See README, section "Data".')
if os.path.getsize(PATH) != SIZE: sys.exit(f'ERROR: size {os.path.getsize(PATH):,} != {SIZE:,}; download incomplete. Resume with: curl -C - -o junyi.rar <URL>')
h = hashlib.sha256()
with open(PATH, 'rb') as f:
    for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
if h.hexdigest() != SHA: sys.exit('ERROR: SHA-256 mismatch; file altered or corrupt.')
print('OK: junyi.rar verified')
