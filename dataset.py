from skimage import io
import numpy as np
import argparse
import json

class Dataset:

   def __init__(self, name_of_set, type_of_set):
       self.name_of_set = name_of_set
       self.type_of_set = type_of_set

   def get_number_of_images(self):
       file_path = 'datasets/nerf/' + str(self.name_of_set)+ '/'
       file_name = 'transforms_'+ str(self.type_of_set)+'.json'
       file = open(file_path + file_name)
       data = json.load(file)
       global transformations
       transformations = []
       for i in data['frames']:
           transformations.append(i)
       file.close()

       return len(transformations)

   def get_image(self, image1, image2):
       '''
       Return image1, image2, T
       '''
       images_path = 'datasets/nerf/' + str(self.name_of_set) + '/' + str(self.type_of_set)+ '/'
       image1_file = 'r_' + str(image1) + '.png'
       image2_file = 'r_' + str(image2) + '.png'
       image1_file = io.imread(images_path + image1_file)
       image2_file = io.imread(images_path + image2_file)

       T1 = transformations[image1]['transform_matrix']
       T2 = transformations[image2]['transform_matrix']
       transformation_matrix = np.linalg.inv(T1) * T2

       return image1_file ,image2_file,transformation_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset initialization.')
    parser.add_argument("dataset_name", type=str,
                        help='name of the interested dataset')
    parser.add_argument("dataset_type", type=str,
                        help='type of the interested dataset[train,test,val]')
    parser.add_argument("image1_index", type=int,
                        help='index of the first image')
    parser.add_argument("image2_index", type=int,
                        help='index of the second image')

    # Overwrite input arguments:
    argv = ["chair","train", '50', '63']
    return parser.parse_args(argv[:])


if __name__ == '__main__':
    args = parse_args()
    dataset = Dataset(args.dataset_name, args.dataset_type)

    num_images = dataset.get_number_of_images()
    image1,image2,T = dataset.get_image(args.image1_index, args.image2_index)

    
