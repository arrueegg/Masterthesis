# Masterthesis

This repository includes all code used for the generated results used in the corresponding report.

Each method presented in the report was generated with the code in the respective folder.

Single Sensor Solutions:      DDA_single
Fuison-DA (baseline):         DDA_UrbanExtraction
Discriminator approach:       Disc_reset
Input Augmentation approach:  KTH_SD_reset

all folders contain the train_dualnetwork.py file which is used to launch the runs, all corresponding files, with network architecture, dataloader, etc. are stored in the subfolders like utils. All these files vary slightly among the methods but are structured the same and also include mostly the identical code.

Files like example.py, extract_SD.py, report_plots.py, etc. are helper files to generate the dataset or visualizations for the report.
