import os
from trimesh import Trimesh
from skimage import measure
from data import *




def visMC(VDtarget, VDpred, DataId, Threshold=0.5, path='MCs', supervised=None):
    # Padding required to remove artifacts in the isosurfaces..
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector
    
    # necessary for distributed training 
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass

    VDtarget = np.pad(VDtarget,2,pad_with)
    try:
        # verts, faces, normals, _ = measure.marching_cubes_lewiner(VDtarget, Threshold, allow_degenerate=False)
        verts, faces, normals, _ = measure.marching_cubes(VDtarget, 
                                                          Threshold, 
                                                          allow_degenerate=False, 
                                                          method='lewiner')
    except ValueError:
        Threshold = (VDtarget.max()-VDtarget.min())*0.5 + VDtarget.min()
        # verts, faces, normals, _ = measure.marching_cubes_lewiner(VDtarget, Threshold, allow_degenerate=False)
        verts, faces, normals, _ = measure.marching_cubes(VDtarget, 
                                                          Threshold, 
                                                          allow_degenerate=False, 
                                                          method='lewiner')
    
    mesh = Trimesh(vertices=verts,
                    faces=faces,
                    vertex_normals=normals)
    mesh.export(file_obj=os.path.join(path,'target_%s.obj'%(DataId)))

    VDpred = np.pad(VDpred,2,pad_with)
    try:
        # verts, faces, normals, _ = measure.marching_cubes_lewiner(VDpred, Threshold, allow_degenerate=False)
        verts, faces, normals, _ = measure.marching_cubes(VDpred, 
                                                          Threshold, 
                                                          allow_degenerate=False, 
                                                          method='lewiner')
    except ValueError:
        Threshold = (VDpred.max()-VDpred.min())*0.5 + VDpred.min()
        # verts, faces, normals, _ = measure.marching_cubes_lewiner(VDpred, Threshold, allow_degenerate=False)
        verts, faces, normals, _ = measure.marching_cubes(VDpred, 
                                                          Threshold, 
                                                          allow_degenerate=False, 
                                                          method='lewiner')
    
    mesh = Trimesh(vertices=verts,
                    faces=faces,
                    vertex_normals=normals)
    mesh.export(file_obj=os.path.join(path,'pred_%s.obj'%(DataId)))