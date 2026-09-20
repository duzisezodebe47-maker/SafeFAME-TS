"""Prepare an isolated expanded dataset; never replace original observations."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from data_utils import load_time_ordered_frame, numerical_intervals
from train_dlinear import DATE_COLUMNS
from build_text_index import load_corpus, build_domain_index
from build_semantic_features import run as aggregate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data_processed/v4'

def main():
    config=json.loads((ROOT/'configs/safefame_v4.json').read_text(encoding='utf-8'))
    raw_root=ROOT/'references/external/Time-MMD'
    OUT.mkdir(parents=True,exist_ok=True)
    audits=[]
    source_hashes={}
    for domain in config['domains']:
        path=raw_root/'numerical'/domain/f'{domain}.csv'
        f,order=load_time_ordered_frame(path)
        starts,ends=numerical_intervals(f,path)
        numeric=f.drop(columns=[c for c in DATE_COLUMNS if c in f],errors='ignore').apply(pd.to_numeric,errors='coerce')
        nonfinite_counts={c:int(np.isinf(numeric[c]).sum()) for c in numeric if np.isinf(numeric[c]).any()}
        numeric=numeric.replace([np.inf,-np.inf],np.nan)
        numeric=numeric.loc[:,numeric.notna().any()]
        duplicate=starts.duplicated(keep=False)
        if duplicate.any():
            f.loc[duplicate].to_csv(OUT/f'{domain}_ambiguous_raw_rows.csv',index=False)
            # Conflicting observations have no adjudicated truth: mark unknown,
            # preserve the timestamp, exclude any target window touching it.
            numeric.loc[duplicate,:]=np.nan
        clean=numeric.copy()
        clean.insert(0,'end_date',ends.dt.strftime('%Y-%m-%d'))
        clean.insert(0,'start_date',starts.dt.strftime('%Y-%m-%d'))
        clean=clean.loc[~starts.duplicated(keep='first')].reset_index(drop=True)
        first,last=clean.OT.first_valid_index(),clean.OT.last_valid_index()
        clean=clean.loc[first:last].reset_index(drop=True)
        assert clean.OT.notna().sum()>100 and not clean.start_date.duplicated().any()
        target=OUT/'numerical'/domain/f'{domain}.csv'
        target.parent.mkdir(parents=True,exist_ok=True)
        clean.to_csv(target,index=False)
        audits.append(dict(domain=domain,raw_rows=len(f),clean_rows=len(clean),
                           duplicate_rows=int(duplicate.sum()),unknown_target_rows=int(clean.OT.isna().sum()),
                           trimmed_edge_rows=int(first+len(numeric.loc[~starts.duplicated(keep='first')])-1-last),
                           first=clean.start_date.iloc[0],last=clean.end_date.iloc[-1],nonfinite_replaced=json.dumps(nonfinite_counts),**order))
        source_hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    corpus,corpus_audit=load_corpus(raw_root,list(config['domains']))
    for p in (raw_root/'textual').glob('*/*.csv'):
        source_hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    textroot=OUT/'text'; textroot.mkdir(exist_ok=True)
    corpus.to_csv(textroot/'fact_corpus.csv',index=False)
    indices=[]
    for domain,settings in config['domains'].items():
        f=pd.read_csv(OUT/'numerical'/domain/f'{domain}.csv')
        indices.append(build_domain_index(domain,pd.to_datetime(f.start_date),pd.to_datetime(f.end_date),
                       corpus[corpus.domain==domain],settings['input_len'],32))
        print('indexed',domain,len(indices[-1]),flush=True)
    pd.concat(indices,ignore_index=True).to_csv(textroot/'sample_text_index.csv',index=False)
    # Reuse already-audited vectors by stable text identity; only encode additions.
    oldroot=ROOT/'data_processed/embeddings/all_minilm_l6_v2'
    oldindex=pd.read_csv(oldroot/'fact_embedding_index.csv')
    oldcorpus=pd.read_csv(ROOT/'data_processed/text/fact_corpus.csv').set_index('text_id')
    oldvectors=np.load(oldroot/'fact_embeddings.npy',mmap_mode='r')
    mapping=dict(zip(oldindex.text_id,oldindex.row))
    vectors=np.empty((len(corpus),384),dtype=np.float32)
    missing=[]
    for i,row in enumerate(corpus.itertuples(index=False)):
        if row.text_id in mapping:
            assert oldcorpus.loc[row.text_id,'fact']==row.fact
            vectors[i]=oldvectors[mapping[row.text_id]]
        else:missing.append(i)
    from encode_text import MODEL_NAME,MODEL_REVISION,corpus_digest
    if missing:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(MODEL_NAME,revision=MODEL_REVISION,local_files_only=True,device='cuda')
        vectors[missing]=model.encode(corpus.iloc[missing].fact.tolist(),batch_size=128,
            convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
    eroot=OUT/'embeddings';eroot.mkdir(exist_ok=True)
    np.save(eroot/'fact_embeddings.npy',vectors)
    pd.DataFrame({'text_id':corpus.text_id,'row':range(len(corpus))}).to_csv(eroot/'fact_embedding_index.csv',index=False)
    (eroot/'fact_embedding_metadata.json').write_text(json.dumps(dict(model=MODEL_NAME,revision=MODEL_REVISION,
        corpus_sha256=corpus_digest(corpus),shape=list(vectors.shape),reused=len(corpus)-len(missing),encoded=len(missing)),indent=2),encoding='utf-8')
    aggregate(textroot,eroot,OUT/'semantic_features')
    pd.DataFrame(audits).to_csv(OUT/'numerical_audit.csv',index=False)
    (OUT/'preparation_audit.json').write_text(json.dumps(dict(
        sources_sha256=source_hashes,corpus_audit=corpus_audit,numerical=audits,
        unknown_target_policy='exclude complete forecast windows containing unknown truth; causal filling is for inputs only',
        text_time='end_date proxy strictly before forecast start; not verified publication time',
        upstream_checked_revision=config['data_revision'],upstream_last_commit='2025-01-14; no newer snapshot found'),
        ensure_ascii=False,indent=2),encoding='utf-8')
    print('PREPARATION_PASS',len(corpus),'facts',flush=True)

if __name__=='__main__': main()
