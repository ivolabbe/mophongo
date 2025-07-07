#%%
import sys
import os

# Add the source directory to Python path
project_root = '/Users/ivo/Desktop/mophongo/mophongo'
src_path = os.path.join(project_root, 'src')

# Make sure the path is added
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    
# test_example.py
def test_addition():
    assert 1 + 1 == 2

# %%
if __name__ == "__main__":

    print(1+1)

# %%

#from mophongo.psf import PSF
#PSF.gaussian(5,0.1,0.1).array.round()