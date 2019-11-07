import os

if __name__ == '__main__':
    dataset = 'datas/jester/jester-v1'
    with open('%s-labels.csv' % dataset) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open('datas/jester/category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    # train and validate dataset
    files_input = ['%s-validation.csv' % dataset, '%s-train.csv' % dataset]
    files_output = ['datas/jester/val_videofolder.txt', 'datas/jester/train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            dir_files = os.listdir(os.path.join('/home/hjm/Data/20bn-jester-v1/', curFolder))
            output.append('%s %d %d' % ('/home/hjm/Data/20bn-jester-v1/' + curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))

    # test dataset
    with open('datas/jester/jester-v1-test.csv') as f:
        lines = f.readlines()
    folders = []
    for line in lines:
        folders.append(line.strip())
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        dir_files = os.listdir(os.path.join('/home/hjm/Data/20bn-jester-v1/', curFolder))
        output.append('%s %d' % ('/home/hjm/Data/20bn-jester-v1/' + curFolder, len(dir_files)))
        print('%d/%d' % (i, len(folders)))
    with open('datas/jester/test_videofolder.txt', 'w') as f:
        f.write('\n'.join(output))
