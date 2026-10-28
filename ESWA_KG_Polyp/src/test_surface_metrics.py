"""Independent geometric regression checks; no training data or torch needed."""
import unittest
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from surface_metrics import surface_distance


def brute_force(pred, gt):
    a=np.argwhere(pred & ~binary_erosion(pred))
    b=np.argwhere(gt & ~binary_erosion(gt))
    distances=cdist(a,b)
    both=np.r_[distances.min(axis=1),distances.min(axis=0)]
    return np.percentile(both,95),both.mean()


class SurfaceDistanceTests(unittest.TestCase):
    def test_nested_boundaries_and_hole(self):
        a=np.zeros((20,20),bool);a[2:18,2:18]=True
        b=np.zeros_like(a);b[5:15,5:15]=True
        for pred,gt in [(a,b),(b,a),(a,a)]:
            np.testing.assert_allclose(surface_distance(pred,gt),brute_force(pred,gt),rtol=0,atol=1e-12)
        a[7:11,7:11]=False
        np.testing.assert_allclose(surface_distance(a,b),brute_force(a,b),rtol=0,atol=1e-12)

    def test_disjoint_single_pixels(self):
        a=np.zeros((12,12),bool);b=np.zeros_like(a)
        a[1,1]=True;b[4,5]=True
        self.assertEqual(surface_distance(a,b),(5.0,5.0))

    def test_partial_overlap_and_symmetry(self):
        a=np.zeros((20,20),bool);b=np.zeros_like(a)
        a[1:9,2:15]=True;b[4:17,9:18]=True
        np.testing.assert_allclose(surface_distance(a,b),brute_force(a,b),rtol=0,atol=1e-12)
        np.testing.assert_allclose(surface_distance(a,b),surface_distance(b,a),rtol=0,atol=1e-12)

    def test_empty_exclusion(self):
        a=np.zeros((5,5),bool);b=a.copy();b[2,2]=True
        for x,y in [(a,a),(a,b),(b,a)]:
            self.assertTrue(np.isnan(surface_distance(x,y)).all())


if __name__=='__main__':
    unittest.main()
