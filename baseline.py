# prompt: from recbole library, write my a python script that run models NMF, PMF, LightGCN on my dataset: Frappe.csv and MusicIncar.csv

!pip install recbole

import pandas as pd
from recbole.quick_start import run_recbole

# Specify dataset paths
frappe_path = "/content/Frappe.csv"  # Replace with your actual path
musicincar_path = "/content/MusicIncar.csv" # Replace with your actual path

# Create config files for each dataset
# For Frappe.csv
frappe_config = """
config_file = 'Frappe.yaml'
dataset_file = '/content/Frappe.csv'
model = NMF
epochs=1
"""
with open('Frappe.yaml', 'w') as f:
  f.write(frappe_config)

#Run NMF on Frappe dataset
run_recbole(model='NMF', dataset='Frappe', config_file='Frappe.yaml')

#Run PMF on Frappe dataset
run_recbole(model='PMF', dataset='Frappe', config_file='Frappe.yaml')


#Run LightGCN on Frappe dataset
run_recbole(model='LightGCN', dataset='Frappe', config_file='Frappe.yaml')



# Create config files for each dataset
# For MusicIncar.csv
musicincar_config = """
config_file = 'MusicIncar.yaml'
dataset_file = '/content/MusicIncar.csv'
model = NMF
epochs=1
"""
with open('MusicIncar.yaml', 'w') as f:
  f.write(musicincar_config)


#Run NMF on MusicIncar dataset
run_recbole(model='NMF', dataset='MusicIncar', config_file='MusicIncar.yaml')

#Run PMF on MusicIncar dataset
run_recbole(model='PMF', dataset='MusicIncar', config_file='MusicIncar.yaml')

#Run LightGCN on MusicIncar dataset
run_recbole(model='LightGCN', dataset='MusicIncar', config_file='MusicIncar.yaml')