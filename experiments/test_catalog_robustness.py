import unittest
import numpy as np
from catalog_robustness import inject_catalog,farthest_catalog
from frbo import finite_rank_biased_overlap
class CatalogTests(unittest.TestCase):
 def test_whole_history_excluded_and_ties(self):
  x=np.array([[0.,0.],[1.,0.],[10.,0.],[-10.,0.],[9.,0.]])
  ids,d=farthest_catalog(x,[[0,1,1,2]],2)
  self.assertEqual(ids,[[3,4]]) # item 2 excluded although outside a hypothetical input window
  ids,_=farthest_catalog(x,[[0]],2);self.assertEqual(ids,[[2,3]])
 def test_relocation_matches_old_removal(self):
  s=list(range(20));rank=list(reversed(range(20)));new=list(range(100,106))
  for a,n in [(.05,1),(.1,2),(.2,4),(.3,6)]:
   for dest in ['old','middle','recent']:
    out,removed,inserted,items=inject_catalog(s,rank,new,a,dest)
    self.assertEqual(removed,list(range(n)));self.assertEqual(len(out),20)
    self.assertEqual([v for v in out if v not in items],s[n:])
    self.assertEqual([out[i] for i in inserted],new[:n])
    self.assertEqual(len(set(out)-set(s)),n)
 def test_reject_seen(self):
  with self.assertRaises(ValueError):inject_catalog([1,2],[0,1],[2],.1,'recent')
 def test_distance_reference(self):
  rng=np.random.default_rng(42);x=rng.normal(size=(71,13));hist=[[0,1,1,5],[3,7,12]]
  ids,ds=farthest_catalog(x,hist,6)
  for h,got,dist in zip(hist,ids,ds):
   d=np.linalg.norm(x-x[sorted(set(h))].mean(0),axis=1)
   expected=sorted(set(range(len(x)))-set(h),key=lambda i:(-d[i],i))[:6]
   self.assertEqual(got,expected);np.testing.assert_allclose(dist,d[got])
 def test_frbo(self):
  self.assertEqual(finite_rank_biased_overlap(list(range(10)),list(range(10))),1)
  self.assertEqual(finite_rank_biased_overlap(list(range(10)),list(range(10,20))),0)
if __name__=='__main__':unittest.main()
