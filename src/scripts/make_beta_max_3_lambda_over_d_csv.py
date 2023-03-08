import pandas as pd
import numpy as np
import paths
from hwo import *

# telescope d and wlen are imported from hwo file to make consistent for all the codes

# givea a nan for the first element because it tries to calculate 
# arccos of 1.032, presumably this is alpha or proxima cen

df = pd.read_csv(paths.data / '2646_NASA_ExEP_Target_List_HWO_Table.csv')
hab_zones = np.array(df['EEIDmas'].values[1:],dtype=float) #in mas
beta_max_3ld =  np.degrees(np.arccos(3*wavelength/d*206265/(hab_zones/1000)))
np.savetxt(paths.data / "beta_max_3_lambda_over_d.csv",beta_max_3ld,delimiter=",")