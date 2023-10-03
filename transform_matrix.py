import numpy as np
import os
import random

class transform_matrix():
    def __init__(self):
        
        self.representation_type = 'surface_trimesh_voxels'
        #self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/scripts_bigData'
        #self.train_samples_filename = os.path.join(self.metadata_path, 'train_samples_128.txt')

    def matrix_element_normalize(self, element: float):
        shifted = element + 1
        scaled = shifted / 2
        return scaled
        

    def generate_matrix(self, x, y, theta, z = 0):
        #a b c 
        #d e f
        #f h i
        #
        # print("x: ", x, " y: ", y, " theta: ", theta, " z: ", z)
        theta = np.radians(theta)
        a = x*x*(1-np.cos(theta)) + np.cos(theta)
        b = y*x*(1-np.cos(theta)) - z*np.sin(theta)
        c = x*z*(1-np.cos(theta)) + y*np.sin(theta)
        d = y*x*(1-np.cos(theta)) + z*np.sin(theta)
        e = y*y*(1-np.cos(theta)) + np.cos(theta)
        f = y*z*(1-np.cos(theta)) - x*np.sin(theta)
        g = x*z*(1-np.cos(theta)) - y*np.sin(theta)
        h = y*z*(1-np.cos(theta)) + x*np.sin(theta)
        i = z*z*(1-np.cos(theta)) + np.cos(theta)

        output = np.array([[a,b,c],[d,e,f],[g,h,i]])

        return output
    
    def is_float(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def get_transforms(self, views_input, filepath):
        #views is a list of files to load
        transform_list = []
        views = [int(v) for v in views_input]

        #with open(self.train_samples_filename, 'r') as f:
        #    dir_list = f.readlines()
        #    dirs = [d.strip() for d in dir_list]

        #for dir in dirs:
        filepath = os.path.join(filepath, 'metadata.txt')

        with open(filepath, 'r') as f:
            metadata = f.readlines()
            metadata = [m.strip() for m in metadata]
            metadata = [float(m) if self.is_float(m) else m for m in metadata]

        for idx in range(0, len(views) - 1, 2) :
            start_meta_line = views[idx] * 4
            end_meta_line = views[idx + 1] * 4
            start_xytheta = metadata[start_meta_line+1], metadata[start_meta_line+2], metadata[start_meta_line+3]
            end_xytheta = metadata[end_meta_line+1], metadata[end_meta_line+2], metadata[end_meta_line+3]

            #get transform between two views
            start_matrix = self.generate_matrix(start_xytheta[0], start_xytheta[1], start_xytheta[2])
            end_matrix = self.generate_matrix(end_xytheta[0], end_xytheta[1], end_xytheta[2])

            end_inverted = np.linalg.inv(end_matrix)

            product = np.dot(start_matrix, end_inverted)

            normalize = np.vectorize(self.matrix_element_normalize)
            normalized_matrix = normalize(product)
         
            transform_list.append(normalized_matrix)





        return np.asarray(transform_list[0]).astype(np.float32)
