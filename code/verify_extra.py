"""Check size and SHA-256 of the replication files (ASSISTments 2009-2010, EdNet-KT1 shard 0, EdNet questions, XES3G5M archive)."""
import hashlib, os, sys
EXP = [('skill_builder_data_corrected_collapsed.csv', 64_412_812, '162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da'),
       ('kt1 shard 0 (train-00000-of-00011.parquet)', 170_673_611, '7908e8d37925de5321877a7a62bfe082aa7cda38b415923e2b28e350fe61fdd9'),
       ('questions.parquet', None, '90a63786bac294e1e9e20b6f78ab2ee787934159d95c3669c79768c69645f84d'),
       ('XES3G5M.tar.gz', 373_235_093, '62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73')]
for path, (name, size, sha) in zip(sys.argv[1:5], EXP):
    if not os.path.exists(path): sys.exit(f'ERROR: {path} not found ({name}). See README, section "Data".')
    if size and os.path.getsize(path) != size: sys.exit(f'ERROR: {path}: size {os.path.getsize(path):,} != {size:,}')
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    if h.hexdigest() != sha: sys.exit(f'ERROR: {path}: SHA-256 mismatch ({name})')
    print(f'OK: {name}')
