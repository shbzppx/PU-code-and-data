Data Availability metadata preview

Neural-network random seeds used for OOF-5 training were 1, 3, 5, 7, 9, 11, 13, 15, 17 and 43.

Files
01_grid_and_channels.json              Grid size, spacing, extent, 14 channel names
02_training_seeds.json                 Neural-network training seeds
03_fold_manifest.csv                   OOF-5 outer-fold assignment
04_camp_assignment_no_xy.csv           Deposit-to-ore-field table without XY
05_km2_settings.json                   KM2 estimator settings (not used as the training loss prior)
06_word2vec_unit_map.csv               Geological-legend codes to English unit names
07_word2vec_summary_public.json        Word2Vec hyperparameters
08_word2vec_references_English.md      Word2Vec corpus bibliography (English; preferred for sharing)
09_word2vec_references_sorted.md       Same corpus list in Chinese (local/sorted mirror)
10_protocol_snapshot.json              Sanitized protocol aligned to the revised manuscript
                                       (3 equal-weight positive windows; inner leave-one among
                                        4 training ore fields with 500 m catchment buffer)

Not included here (restricted)
- A2K_main_dim5_seed42_faultfull.h5
- deposit.txt XY, catchment .grd, fault vertices
- stream-sediment point files and Surfer/SCB interpolated rasters
