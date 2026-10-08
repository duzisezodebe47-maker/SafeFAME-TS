"""独立复算第15轮选择规则、成熟反馈、修正模型与指标。"""
from common15 import *
from analysis15 import state_features,age_mask

def verify():
    if not LOCAL.exists():
        print('PARTIAL: 第15轮本地大型证据未随Git分发');return
    registration=read(LOCAL/'input_registration.json')
    for rel,digest in registration['files'].items():require(sha(PROJECT/rel)==digest,'registered input/code changed: '+rel)
    for rel,digest in read(LOCAL/'MANIFEST.json')['files'].items():require(sha(LOCAL/rel)==digest,'output bytes changed: '+rel)
    metric_count=0;online_count=0;selection_count=0;decomposition_count=0
    for h in config()['horizons']:
        folder=PREVIOUS/f'prepared/h{h}';sm=pd.read_csv(folder/'selection/metadata.csv');tm=pd.read_csv(folder/'test_features/metadata.csv');sy=np.load(folder/'selection/y.npy');ty=np.load(folder/'test_truth.npy')
        sp=np.load(PREVIOUS/f'models/h{h}/selection_predictions.npz');tp=np.load(PREVIOUS/f'evaluation/h{h}/fixed_predictions.npz');dec=sm.segment.eq('decision').to_numpy();cal=sm.segment.eq('calibration').to_numpy();names=config()['numeric_candidates']
        require(sm.loc[dec,'target_end'].max()<tm.target_date.min(),'decision crosses test')
        numeric=np.load(LOCAL/f'h{h}/numeric_predictions.npz');attribution=np.load(LOCAL/f'h{h}/attribution_predictions.npz');gated=np.load(LOCAL/f'h{h}/gated_predictions.npz')
        choices=read(LOCAL/f'h{h}/policy_choices.json');loss=np.stack([((sp[n][dec]-sy[dec])**2).mean(axis=1) for n in names],axis=1)
        for rule,name in choices['fixed_numeric'].items():
            if rule=='decision_mean':score=loss.mean(axis=0)
            elif rule=='recent_26_origins':score=loss[-26:].mean(axis=0)
            else:
                blocks=np.array([v.mean(axis=0) for v in np.array_split(loss,3)]);score=(blocks/np.maximum(blocks.min(axis=1)[:,None],1e-12)-1).max(axis=0)
            winner=min(range(len(names)),key=lambda j:(score[j],names[j]));require(name==names[winner],'fixed choice differs');np.testing.assert_array_equal(numeric[rule],tp[name]);selection_count+=1
        # Independent per-origin reconstruction, not the production online loop.
        origins=np.r_[sm.loc[dec,'origin_index'],tm.origin_index];targets=np.r_[sy[dec],ty];p=np.stack([np.r_[sp[n][dec],tp[n]] for n in names],axis=1);warm=int(dec.sum())
        for lookback in config()['online_lookbacks']:
            key=f'OnlineMatured{lookback}';trace=pd.read_csv(LOCAL/f'h{h}/{key}_trace.csv')
            for row,now in enumerate(tm.origin_index):
                ids=np.flatnonzero(origins+h<=now)[-lookback:]
                values=((p[ids]-targets[ids,None,:])**2).mean(axis=(0,2));winner=min(range(len(names)),key=lambda j:(values[j],names[j]))
                require(trace.iloc[row].candidate==names[winner],'online selection mismatch');require(int(trace.iloc[row].last_feedback_origin)==origins[ids[-1]],'feedback identity differs')
                np.testing.assert_array_equal(numeric[key][row],p[warm+row,winner]);online_count+=1
        base=tp['SelectedNumeric'];available=tm.text_available.to_numpy(bool);correction=tp['DirectFusion']-base
        oof=np.load(PREVIOUS/f'models/h{h}/oof_residuals.npz');mean=oof['residual'].mean(axis=0)
        np.testing.assert_allclose(attribution['ConstantResidual'],base+mean*available[:,None],atol=1e-12)
        np.testing.assert_allclose(attribution['OriginalText'],tp['DirectFusion'],atol=1e-12)
        np.testing.assert_allclose(attribution['CenteredText']-base+attribution['ConstantResidual']-base,correction,atol=1e-12);decomposition_count+=1
        for weeks in config()['age_weeks']:np.testing.assert_allclose(attribution[f'TextAge{weeks}w'],base+correction*age_mask(tm,weeks)[:,None],atol=1e-12)
        state=read(LOCAL/f'h{h}/state_model.json');sx=np.load(folder/'selection/x.npy');tx=np.load(folder/'test_features/x.npy');sf=state_features(sx,pd.to_datetime(sm.target_date));tf=state_features(tx,pd.to_datetime(tm.target_date));ids=sm.set_index('origin_index').index.get_indexer(oof['origins'])
        mu=sf[ids].mean(axis=0);std=sf[ids].std(axis=0);std[std<1e-12]=1;np.testing.assert_allclose(mu,state['mean']);np.testing.assert_allclose(std,state['std'])
        x=(sf[ids]-mu)/std;y=oof['residual'];xm=x.mean(axis=0);ym=y.mean(axis=0);xc=x-xm;w=np.linalg.solve(xc.T@xc+state['alpha']*np.eye(x.shape[1]),xc.T@(y-ym))
        np.testing.assert_allclose(w,state['ridge']['w'],rtol=1e-10,atol=1e-12)
        pred=((tf-mu)/std-xm)@w+ym;pred[~available]=0;np.testing.assert_allclose(attribution['NumericStateResidual'],base+pred,atol=1e-12)
        gate_table=pd.read_csv(HERE/f'results/h{h}/gate_selection.csv')
        for name,weight in choices['gates'].items():
            table=gate_table[gate_table.candidate==name];valid=table[(table.weight==0)|((table.half1_benefit>0)&(table.half2_benefit>0))];best=valid.sort_values(['mse','weight']).iloc[0]
            require(float(best.weight)==weight,'gate choice differs');np.testing.assert_allclose(gated[name],base+weight*(attribution[name]-base),atol=1e-12)
        frames={'numeric_selection':numeric,'attribution':attribution,'decision_gated_controls':gated};table=pd.read_csv(HERE/f'results/h{h}/metrics.csv',float_precision='round_trip')
        expected={(g,n) for g,data in frames.items() for n in data.files if n!='origins'};require(set(zip(table.group,table.candidate))==expected,'metric identities differ')
        for row in table.itertuples():
            pred=frames[row.group][row.candidate];actual=mse(pred,ty);require(np.isclose(actual,row.mse,rtol=1e-12,atol=1e-14),'MSE differs');require(np.isclose(100*(1-actual/mse(base,ty)),row.gain_vs_round14_selected_pct,atol=1e-10),'gain differs');metric_count+=1
    result={'status':'PASS_EXPLORATORY_WITH_SOURCE_LIMITATIONS','input_files_checked':len(registration['files']),'metrics_recomputed':metric_count,'online_predictions_recomputed':online_count,'fixed_selections_checked':selection_count,'decompositions_checked':decomposition_count,'strict_source_eligible':0,'role':config()['role'],'verifier_sha256':sha(Path(__file__))}
    dump(HERE/'results/verification.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':verify()
