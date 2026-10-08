"""三州输入、窗口隔离、反馈延迟和真实权重推理的验收。"""
import unittest,io
from common16 import *
from prepare16 import validate_series,windows
from predict16 import validate,forecast

class Tests(unittest.TestCase):
    def rows(self):return [{'name':'California','mapDate':f'2024-01-{d:02d}T00:00:00','dsci':float(d)} for d in (2,9,16)]
    def test_identity_rejected(self):
        with self.assertRaises(ValueError):validate_series(self.rows(),'Texas')
    def test_duplicate_and_missing_week(self):
        for rows in (self.rows()+self.rows()[:1],[self.rows()[0],self.rows()[2]]):
            with self.assertRaises(ValueError):validate_series(rows,'California')
    def test_bad_raw_values(self):
        for value in (np.nan,-1,501):
            rows=self.rows();rows[0]['dsci']=value
            with self.assertRaises(ValueError):validate_series(rows,'California')
    def test_target_boundary_purge(self):
        dates=pd.Series(pd.date_range('2020-01-07',periods=30,freq='7D'));p={'input_length':4,'segments':{'train':str(dates[15].date()),'test':str(dates[29].date())}};m,x,y=windows(np.arange(30),dates,4,p)
        self.assertTrue((m[m.segment=='train'].origin_index<=12).all());self.assertFalse(m.origin_index.isin([13,14,15]).any());np.testing.assert_array_equal(x[0],[0,1,2,3]);np.testing.assert_array_equal(y[0],[4,5,6,7])
    def test_future_target_changes_not_history(self):
        dates=pd.Series(pd.date_range('2020-01-07',periods=30,freq='7D'));p={'input_length':4,'segments':{'test':str(dates[29].date())}};a=np.arange(30,dtype=float);b=a.copy();b[20:]=999;m,x,y=windows(a,dates,4,p);_,z,_=windows(b,dates,4,p);np.testing.assert_array_equal(x[m.origin_index<=20],z[m.origin_index<=20])
    def test_prediction_invalid_inputs(self):
        normal={'state':'CA','horizon':4,'rule':'CalChoice','history':[1]*52}
        for patch in ({'state':'ZZ'},{'horizon':True},{'rule':'best_test_score'},{'history':[1]*51},{'history':[True]*52},{'history':[float('inf')]*52},{'history':[-1]*52}):
            with self.assertRaises(ValueError):validate(normal|patch)
        with self.assertRaises(ValueError):validate(normal|{'text':'unverified'})
    def test_delayed_feedback_with_gaps(self):
        a=np.zeros((4,2,4));a[:,1]=1;read_ids=[];p,t=online_select([0,2,5,6],a,4,['Last','Ridge'],26,lambda i:(read_ids.append(i) or np.ones(4)));self.assertEqual(read_ids,[0,1]);self.assertEqual([v['last_feedback_origin'] for v in t],[None,None,0,2])
    def test_future_labels_do_not_change_prior_choices(self):
        rng=np.random.default_rng(2026);p=rng.normal(size=(80,3,12));a=np.ones((80,12));b=a.copy();b[40:]=50
        x,t=online_select(np.arange(80),p,12,['Last','Ridge','DLinear'],26,lambda i:a[i]);z,u=online_select(np.arange(80),p,12,['Last','Ridge','DLinear'],26,lambda i:b[i]);np.testing.assert_array_equal(x[:52],z[:52]);self.assertEqual(t[:52],u[:52])
    def test_all_six_tasks_saved_inference(self):
        for state in cfg()['states']:
            for h in cfg()['horizons']:
                code=state['code'];scale=read(LOCAL/f'prepared/{code}/scaler.json');x=np.load(LOCAL/f'prepared/{code}_h{h}/test_features.npz')['x'][-1];rule='three_block_minimax_relative_regret';r=forecast({'state':code,'horizon':h,'rule':rule,'history':(x*scale['std']+scale['mean']).tolist()});saved=np.load(LOCAL/f'evaluation/{code}_h{h}/fixed_predictions.npz')[rule][-1];np.testing.assert_allclose(r['forecast_dsci'],saved*scale['std']+scale['mean'],atol=1e-4);self.assertEqual(r['text_weight'],0)
    def test_scaler_tamper_rejected_before_inference(self):
        from unittest.mock import patch
        with patch('predict16.sha',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'scaler bytes changed'):forecast({'state':'CA','horizon':4,'rule':'CalChoice','history':[1]*52})

if __name__=='__main__':
    import torch
    torch.set_num_threads(2);stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));print(stream.getvalue());(HERE/'results/tests.log').write_text(stream.getvalue(),encoding='utf8');dump(HERE/'results/tests.json',{'passed':result.wasSuccessful(),'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'source_sha256':sha(Path(__file__))});sys.exit(0 if result.wasSuccessful() else 1)
