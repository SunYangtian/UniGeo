import os

# for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
for ds in ['pumpkin', 'redkitchen', 'stairs']:

	print("=== Downloading 7scenes Data:", ds, "===============================")

	os.system('wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
	os.system('unzip ' + ds + '.zip')
	os.system('rm ' + ds + '.zip')

	sequences = os.listdir(ds)

	for file in sequences:
		if file.endswith('.zip'):

			print("Unpacking", file)
			os.system('unzip ' + ds + '/' + file + ' -d ' + ds)
			os.system('rm ' + ds + '/' + file)

print("Processing frames...")