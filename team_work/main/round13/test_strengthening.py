"""关键行为回归测试：因果性、序列边界、模型序列化和真实推理/API。"""
import contextlib
import io
import json
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from common import *
from adaptive import DelayedRiskMixer,causal_replay
from predictor import forecast,example,validate_request,MODELS
from robustness import delayed_features
from experiments import refit_ar,ar_predict

class StrengtheningTests(unittest.TestCase):
    def setUp(self):self.config={**protocol()['adaptive'],'minimum_mature_origins':6}
    def test_reject_future_feedback(self):
        m=DelayedRiskMixer(3,self.config)
        with self.assertRaises(ValueError):m.observe(10,[0]*3,[1]*3,[1]*3,12)
        m.observe(10,[0]*3,[1]*3,[1]*3,13)
        with self.assertRaises(ValueError):m.observe(10,[0]*3,[1]*3,[1]*3,14)
    def test_nonfinite_feedback_and_prediction(self):
        for bad in ([1,2],[1,2,float('nan')]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):DelayedRiskMixer(3,self.config).observe(0,[0]*3,[1]*3,bad,3)
        with self.assertRaises(ValueError):DelayedRiskMixer(3,self.config).choose(1,[0,float('inf'),0],[0]*3)
    def test_future_target_perturbation_cannot_change_earlier_weights(self):
        h=3;o=np.arange(60);base=np.zeros((60,h));text=np.ones((60,h));truth1=np.ones((60,h));truth2=truth1.copy();truth2[30:]=-20
        p1,t1=causal_replay(o,base,text,np.ones(60,bool),h,self.config,lambda i:truth1[i])
        p2,t2=causal_replay(o,base,text,np.ones(60,bool),h,self.config,lambda i:truth2[i])
        np.testing.assert_array_equal(p1[:33],p2[:33]);self.assertTrue(any(t['weight']>0 for t in t1));self.assertTrue(any(t1[i]['weight']!=t2[i]['weight'] for i in range(33,60)))
    def test_no_label_request_before_maturity(self):
        def forbidden(i):raise AssertionError('future target accessed')
        pred,_=causal_replay([10,11,12],np.zeros((3,4)),np.ones((3,4)),[True]*3,4,self.config,forbidden)
        np.testing.assert_array_equal(pred,0)
    def test_reject_duplicate_prediction_origins(self):
        with self.assertRaises(ValueError):causal_replay([1,1],np.zeros((2,1)),np.ones((2,1)),[True]*2,1,self.config,lambda i:[0])
    def test_source_missing_and_nan_text_force_fallback_after_warmup(self):
        for source,available,text in ((False,True,[1]*3),(True,False,[1]*3),(True,True,[float('nan')]*3)):
            m=DelayedRiskMixer(3,self.config)
            for i in range(12):m.observe(i,[0]*3,[1]*3,[1]*3,20)
            p,t=m.choose(21,[0]*3,text,available,source);np.testing.assert_array_equal(p,0);self.assertEqual(t['weight'],0)
    def test_delayed_text_never_uses_future_or_wraps(self):
        o=np.array([10,12,15,21]);f=FeatureBundle(np.zeros((4,3)),np.arange(4)[:,None],np.ones((4,2)),np.ones(4,bool),o)
        changed,indices=delayed_features(f,4);self.assertEqual(indices.tolist(),[-1,-1,0,2]);self.assertEqual(changed.text_available.tolist(),[False,False,True,True]);self.assertEqual(changed.semantic[:2].sum(),0)
        with self.assertRaises(ValueError):delayed_features(f,-1)
    def test_saved_ridge_float32_matches_reference_float64(self):
        rng=np.random.default_rng(2026);x=rng.normal(size=(40,7)).astype('float32');y=rng.normal(size=(40,3)).astype('float32');tr=np.arange(40)<20;cal=(np.arange(40)>=20)&(np.arange(40)<30);segments=np.where(tr,'train',np.where(cal,'calibration','decision'));f=FeatureBundle(x,np.zeros((40,1)),np.zeros((40,1)),np.zeros(40,bool),np.arange(40));fy=y.astype(float);fy[~(tr|cal)]=np.nan
        b=numeric_baselines(f,segments,fy,seasonal_period=4);state=refit_ar(x,y,tr,np.zeros(40,bool),b['ridge_alpha']);np.testing.assert_allclose(ar_predict(state,x),b['predictions']['AR-Ridge'],rtol=1e-12,atol=1e-12)
    def test_request_validation(self):
        v=example(protocol()['tasks'][0]);good={'task':v['task'],'history':v['history']}
        bads=[[],{**good,'task':'../../x'},{**good,'model':'unknown'},{**good,'targets':[1]},{**good,'history':v['history'][:-1]},{**good,'history':[True]*len(v['history'])},{**good,'history':['1']*len(v['history'])},{**good,'history':[float('nan')]*len(v['history'])}]
        for bad in bads:
            with self.subTest(bad=str(bad)[:60]):
                with self.assertRaises(ValueError):validate_request(bad)
    def test_all_tasks_all_inference_models(self):
        for task in protocol()['tasks']:
            e=example(task)
            for model in MODELS:
                with self.subTest(task=task,model=model):
                    result=forecast({'task':task,'model':model,'history':e['history']});self.assertEqual(len(result['forecast']),e['horizon']);self.assertTrue(np.isfinite(result['forecast']).all());self.assertEqual(result['text_weight'],0)
            strict=forecast({'task':task,'history':e['history']});base=forecast({'task':task,'model':'N' if not task.startswith('Agriculture') else 'StrictSourceFallback','history':e['history']});np.testing.assert_array_equal(strict['forecast'],base['forecast'])
    def test_real_inference_matches_frozen_predictions(self):
        from verify_strengthening import grid
        for task in protocol()['tasks']:
            e=example(task);meta,_=features(task,'proxy');t=task_spec(task);fit=read(BUNDLE/task/'numeric_fit.json');pred=grid(LOCAL/'comparison'/task/'proxy/static_test_predictions.csv',meta,t['horizon']);pred.update(grid(LOCAL/'deep'/task/'test_predictions.csv',meta,t['horizon']))
            for model in ('N','AR-Ridge-refit','DLinear','PatchTST'):
                with self.subTest(task=task,model=model):
                    result=forecast({'task':task,'model':model,'history':e['history']});np.testing.assert_allclose(result['forecast'],pred[model][-1]*fit['std']+fit['mean'],rtol=2e-5,atol=2e-5)
    def test_grid_rejects_missing_or_duplicate_rows(self):
        from verify_strengthening import grid
        task=protocol()['tasks'][0];meta,_=features(task,'proxy');h=task_spec(task)['horizon'];df=pd.read_csv(LOCAL/'deep'/task/'test_predictions.csv')
        with tempfile.TemporaryDirectory(dir=LOCAL) as directory:
            p=Path(directory)/'bad.csv'
            for bad in (df.iloc[1:],pd.concat([df,df.iloc[:1]])):
                bad.to_csv(p,index=False)
                with self.assertRaises(ValueError):grid(p,meta,h)
    def test_oversized_bootstrap_blocks_rejected(self):
        from robustness import block_ci
        for block in (0,10,20):
            with self.assertRaises(ValueError):block_ci(np.arange(10),block)
        ci=block_ci(np.arange(10),2);self.assertLess(ci[0],ci[1]);self.assertLess(ci[0],4.5);self.assertGreater(ci[1],4.5)
    def test_http_contract(self):
        from serve_demo import Handler,ThreadingHTTPServer
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}'
        def request(path,body=None,ctype='application/json'):
            return urllib.request.urlopen(urllib.request.Request(url+path,data=body,headers={'Content-Type':ctype}),timeout=30)
        try:
            for path in ('/','/app.js','/style.css','/api/summary'):
                with request(path) as r:self.assertEqual(r.status,200)
            e=example(protocol()['tasks'][0]);payload=json.dumps({'task':e['task'],'history':e['history']}).encode()
            with request('/api/predict',payload) as r:self.assertEqual(len(json.load(r)['forecast']),12)
            for path,body,code,ctype in [('/api/predict',b'{',400,'application/json'),('/api/predict',payload,400,'text/plain'),('/api/example?task=bad',None,400,'application/json'),('/../protocol.json',None,404,'application/json')]:
                with self.assertRaises(urllib.error.HTTPError) as caught:request(path,body,ctype)
                self.assertEqual(caught.exception.code,code)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(StrengtheningTests);result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);text=stream.getvalue();(HERE/'results/final/tests.log').write_text(text,encoding='utf-8');dump(HERE/'results/final/tests.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'status':'PASS' if result.wasSuccessful() else 'FAIL','subcase_scope':'24 task/model inference combinations plus 16 frozen prediction comparisons and negative inputs','test_source_sha256':sha(Path(__file__))});print(text);raise SystemExit(0 if result.wasSuccessful() else 1)
