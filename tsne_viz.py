import json
import numpy
import sklearn
import argparse


openmic_vggish_path = './data/openmic-vggish'
openmic_openl3_path = './data/openmic-openl3'
irmas_vggish_path = './data/irmas-vggish'
irmas_openl3_path = './data/irmas-openl3'






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='t-SNE visualization script for openmic and irmas datasets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    args = parser.parse_args()