import numpy as np
import os
import random

class transform_matrix():
    def __init__(self):
        
        self.representation_type = 'surface_trimesh_voxels'
        #self.metadata_path = '/work/mech-ai-scratch/jrrade/Protein/scripts_bigData'
        #self.train_samples_filename = os.path.join(self.metadata_path, 'train_samples_128.txt')

    def generate_matrix(self, x, y, theta, z = 0):
        #a b c 
        #d e f
        #f h i

        a = x*x*(1-np.cos(theta)) + np.cos(theta)
        b = y*x*(1-np.cos(theta)) - z*np.sin(theta)
        c = x*z*(1-np.cos(theta)) + y*np.sin(theta)
        d = y*x*(1-np.cos(theta)) + z*np.sin(theta)
        e = y*y*(1-np.cos(theta)) + np.cos(theta)
        f = y*z*(1-np.cos(theta)) - x*np.sin(theta)
        g = x*z*(1-np.cos(theta)) - y*np.sin(theta)
        h = y*z*(1-np.cos(theta)) + x*np.sin(theta)
        i = z*z*(1-np.cos(theta)) + np.cos(theta)

        return np.array([[a,b,c],[d,e,f],[g,h,i]])
    
    def is_float(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def get_transforms(self, views, filepath):
        #views is a list of files to load
        transform_list = []
        views = [int(v) for v in views]

        #with open(self.train_samples_filename, 'r') as f:
        #    dir_list = f.readlines()
        #    dirs = [d.strip() for d in dir_list]

        #for dir in dirs:
        filepath = os.path.join(filepath, 'metadata.txt')

        with open(filepath, 'r') as f:
            metadata = f.readlines()
            metadata = [m.strip() for m in metadata]
            metadata = [float(m) if self.is_float(m) else m for m in metadata]

        for idx in range(0, len(views) - 1, 2):

            start_xytheta = metadata[idx+1], metadata[idx+2], metadata[idx+3]
            end_xytheta = metadata[idx+5], metadata[idx+6], metadata[idx+7]

            #get transform between two views
            start_matrix = self.generate_matrix(start_xytheta[0], start_xytheta[1], start_xytheta[2])
            end_matrix = self.generate_matrix(end_xytheta[0], end_xytheta[1], end_xytheta[2])
            end_inverted = np.linalg.inv(end_matrix)

            product = np.matmul(start_matrix, end_inverted)

            transform_list.append(product)

        return transform_list

def main():
    transform = transform_matrix()
    metadata_path = '/work/mech-ai-scratch/jrrade/Protein/scripts_bigData'
    train_samples_filename = os.path.join(metadata_path, 'train_samples_128.txt')
    representation_type='surface_trimesh_voxels'
    n_views_rendering = 10

    with open(train_samples_filename, 'r') as f:
        dir_list = f.readlines()
        dirs = [d.strip() for d in dir_list]

    filepath = os.path.join(str(dirs[1]), str(representation_type))
    views = random.sample(range(25), n_views_rendering * 2)

    print(transform.get_transforms(views, filepath))

if __name__ == '__main__':
    main()