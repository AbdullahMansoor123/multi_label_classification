import glob
import xml.etree.cElementTree as ET
from collections import Counter


def label_count(labels_dir):
    label_files = glob.glob(labels_dir + '\*')  # '\*' to get all label
    all_labels = []
    for label_file in label_files:
        # print(label_file)
        root = ET.parse(label_file).getroot()
        # all_labels = []
        for labels in root.iter('name'):  # label file name
            label = labels.text
            # print total labels
            all_labels.append(label)

    # Prints total number of unique labels
    unique_labels = Counter(all_labels).keys()
    print(f'total unique labels: {len(unique_labels)}\n')

    # print total count for unique label in dataset
    results = Counter(all_labels)
    for label_name in results:
        print(label_name + ': ', results[label_name])

    # if jason_data ==True:
        #save label file name and labels in the file


labels_dir = 'household\labels'
label_count(labels_dir)

###things to add in code###
# 1- spell checker for unique classes
# 2- export to jason format
