import splitfolders
input_folder = 'data1/PlantVillage/plant_disease'
splitfolders.ratio(input_folder, output="dataset",
                   seed=42, ratio=(.8, .2),
                   group_prefix=None
)
