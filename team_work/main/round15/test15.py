"""选择规则及因果反馈的边界测试。"""
import unittest,io
from common15 import *
from analysis15 import *

class Contracts(unittest.TestCase):
    def test_relative_regret_not_same_as_mean(self):
        loss=np.array([[.1,1],[1,2],[6,3]])
        self.assertEqual(choose_numeric(loss,['A','B'],'decision_mean')[0],1)
        self.assertEqual(choose_numeric(loss,['A','B'],'three_block_minimax_relative_regret')[0],0)
    def test_tie_is_lexical(self):
        self.assertEqual(choose_numeric(np.ones((3,2)),['Z','A'],'decision_mean')[0],1)
    def test_selection_rejects_nan_and_empty(self):
        for x in (np.empty((0,2)),np.array([[1,np.nan]])):
            with self.assertRaises(ValueError):choose_numeric(x,['A','B'],'decision_mean')
    def test_gapped_origin_maturity(self):
        pred=np.zeros((4,2,4));pred[:,1]=1;reads=[]
        p,t=online_select([0,2,5,6],pred,4,['Last','B'],26,lambda i:(reads.append(i) or np.ones(4)))
        self.assertEqual(reads,[0,1]);self.assertEqual([r['last_feedback_origin'] for r in t],[None,None,0,2]);np.testing.assert_array_equal(p[2:],1)
    def test_future_targets_cannot_change_earlier_choices(self):
        rng=np.random.default_rng(2026);pred=rng.normal(size=(100,2,4));a=np.zeros((100,4));b=a.copy();b[50:]=100
        p,t=online_select(np.arange(100),pred,4,['Last','B'],26,lambda i:a[i]);q,u=online_select(np.arange(100),pred,4,['Last','B'],26,lambda i:b[i])
        np.testing.assert_array_equal(p[:54],q[:54]);self.assertEqual(t[:54],u[:54])
    def test_invalid_origin_order(self):
        with self.assertRaises(ValueError):online_select([2,1],np.zeros((2,2,4)),4,['Last','B'],26,lambda i:np.ones(4))
    def test_gate_requires_both_halves(self):
        y=np.array([[1],[1],[-1],[-1]]);w,_=gate(np.zeros((4,1)),np.ones((4,1)),y,[0,.25,.5,1]);self.assertEqual(w,0)
    def test_report_age_boundary(self):
        m=pd.DataFrame({'target_date':['2024-05-21']*5,'source_map_date':['2024-05-14','2024-05-07','2024-04-16','2024-05-21',None],'text_available':[True,True,True,True,False]})
        self.assertEqual(age_mask(m,0).tolist(),[True,False,False,False,False]);self.assertEqual(age_mask(m,4).tolist(),[True,True,True,False,False])
    def test_state_features_only_use_given_history(self):
        a=np.arange(104,dtype=float).reshape(2,52);b=a.copy();b[1]+=100
        dates=pd.to_datetime(['2024-01-02','2024-01-09']);x=state_features(a,dates);z=state_features(b,dates)
        np.testing.assert_array_equal(x[0],z[0]);self.assertEqual(x[0,1],4);self.assertEqual(x[0,2],13)
    def test_constant_bootstrap_exact(self):
        lo,hi=block_interval(np.ones(50),4,200,2026);self.assertEqual(lo,1);self.assertEqual(hi,1)
    def test_saved_numeric_policy_inference(self):
        from predict15 import forecast
        from predictor14 import example
        for h in (4,12):
            result=forecast({'horizon':h,'history':example(h)['history']})
            s=read(PREVIOUS/f'prepared/h{h}/scaler.json');saved=np.load(LOCAL/f'h{h}/numeric_predictions.npz')['three_block_minimax_relative_regret'][-1]
            np.testing.assert_allclose(result['forecast_dsci'],saved*s['std']+s['mean'],atol=1e-5)
            self.assertEqual(result['text_weight'],0);self.assertEqual(result['model'],{4:'Ridge',12:'Last'}[h])

if __name__=='__main__':
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts));print(stream.getvalue());(HERE/'results/tests.log').write_text(stream.getvalue(),encoding='utf8');dump(HERE/'results/tests.json',{'passed':r.wasSuccessful(),'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'test_source_sha256':sha(Path(__file__))});sys.exit(0 if r.wasSuccessful() else 1)
