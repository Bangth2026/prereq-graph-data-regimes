import sys,os
os.chdir(sys.argv[1]); sys.path.insert(0,'.')
import numpy as np
from evalcore import *
share=np.load('data/dir_share.npy'); expdir=W['BIN_EXP']==1; exprev=np.array([(d,s) in Ex for s,d in zip(SRC,DST)])
ok=~np.isnan(share)
noexp=~expdir&~exprev
rows={
 'log agrees (>0.6), expert same':int((ok&(share>0.6)&expdir).sum()),
 'log agrees (>0.6), no expert edge':int((ok&(share>0.6)&noexp).sum()),
 'log agrees (>0.6), expert reverse -> disputed':int((ok&(share>0.6)&exprev).sum()),
 'log disagrees (<0.4), expert same -> disputed':int((ok&(share<0.4)&expdir).sum()),
 'log disagrees (<0.4), no expert -> disputed':int((ok&(share<0.4)&noexp).sum()),
 'log disagrees (<0.4), expert reverse -> disputed':int((ok&(share<0.4)&exprev).sum()),
 'weak (0.4-0.6), expert same -> ambiguous':int((ok&(share>=0.4)&(share<=0.6)&expdir).sum()),
 'weak, no expert -> ambiguous':int((ok&(share>=0.4)&(share<=0.6)&noexp).sum()),
 'weak, expert reverse -> ambiguous':int((ok&(share>=0.4)&(share<=0.6)&exprev).sum()),
 'no log, expert same -> supported(expert only)':int((~ok&expdir).sum()),
 'no log, expert reverse -> disputed':int((~ok&exprev).sum()),
 'no log, no expert -> unresolved':int((~ok&noexp).sum()),
 'both expdir and exprev':int((expdir&exprev).sum()),
 'n ok':int(ok.sum()),'n expdir':int(expdir.sum()),'n exprev':int(exprev.sum()),
}
for k,v in rows.items(): print(v,k)
