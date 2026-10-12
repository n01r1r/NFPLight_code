import unittest
from unittest.mock import patch
import numpy as np
from capture_processing.geometry import rectify_stages, area_resize, detect_marker_geometry
from capture_processing.burst import _to_final_pixels, _warp_stages

class CropOnlyTests(unittest.TestCase):
    def test_single_crop_removes_border_without_rescaling_marker_quad(self):
        image=np.full((512,512,3), 7, dtype=np.float32)
        image[46:466,46:466]=np.linspace(0,1,420,dtype=np.float32)[None,:,None]
        result=rectify_stages(image,np.eye(3,dtype=np.float32))
        np.testing.assert_array_equal(result['output256'],area_resize(image[46:466,46:466],(256,256)))
        np.testing.assert_array_equal(result['crop420'],image[46:466,46:466])
        self.assertEqual(result['output256'].dtype,np.float32)
        self.assertLessEqual(float(result['output256'].max()),1)
        self.assertEqual(result['geometry']['inset'],0)
        self.assertEqual(result['geometry']['sample_side_cm'],5)
        burst=_warp_stages(image,np.eye(3,dtype=np.float32))
        np.testing.assert_array_equal(burst['output256'],result['output256'])
        self.assertEqual(burst['geometry']['policy'],'sample_crop_only_v1')
    def test_rms_coordinates_use_post_crop_pixel_scale(self):
        points=np.array([[46,46],[466,466],[256,256]],dtype=np.float32)
        expected=np.array([[0,0],[256,256],[128,128]],dtype=np.float32)
        np.testing.assert_allclose(_to_final_pixels(points,512,46,256),expected,atol=2e-5)
    def test_marker_quad_is_not_inset(self):
        boxes=[(5,45),(108,45),(45,5),(45,108)]
        corners=[np.array([[[x,y],[x+15,y],[x+15,y+15],[x,y+15]]],dtype=np.float32) for x,y in boxes]
        with patch('cv2.aruco.ArucoDetector') as detector:
            detector.return_value.detectMarkers.return_value=(corners,np.arange(4,dtype=np.int32)[:,None],[])
            _, record=detect_marker_geometry(np.ones((128,128,3),dtype=np.float32))
        np.testing.assert_allclose(record['quad'],[[20,20],[108,20],[108,108],[20,108]],atol=1e-4)
        self.assertEqual(record['inset'],0)
    def test_nonzero_inset_is_rejected(self):
        for inset in (0.035,-0.1,float('nan')):
            with self.assertRaisesRegex(ValueError,'inset=0'):
                detect_marker_geometry(np.ones((128,128,3),dtype=np.float32),inset=inset)
    def test_invalid_crop_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside the DNG'):
            rectify_stages(np.ones((32,32,3),dtype=np.float32),np.eye(3,dtype=np.float32))

if __name__=='__main__': unittest.main()
