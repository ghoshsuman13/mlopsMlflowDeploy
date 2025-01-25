###############dataset generation###########
python .\createdataset.py

##############model training code with different noise level parameter###############
python .\trainmodel.py

###########dvc enablement#########################
dvc init -f
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -m "Track dataset with DVC"

##########run mlflow ui to see the runs with parametr value###########
mlflow ui

##########change dataset and run the model run again#########

echo "Adding more noise to dataset..."
sed -i 's/0.1/0.2/' generate_dataset.py
python generate_dataset.py

python .\trainmodel.py

mlflow ui

########### push this changes to git master branch##########
git add .
git commit -m "adding changed dataset and model runs"
git push

###########checking out master in detached mode to one commit before and revert to earlier version of commit and checkout that dataset version using dvc checkout###########

git checkout HEAD~1
dvc checkout


