def dataset():
    prefix = '{:05d}.jpg'
    file_categories = 'datas/jester/category.txt'
    file_imglist_train = 'datas/jester/train_videofolder.txt'
    file_imglist_val = 'datas/jester/val_videofolder.txt'
    file_imglist_test = 'datas/jester/test_videofolder.txt'

    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    n_class = len(categories)
    print('jester: {} classes'.format(n_class))

    return n_class, file_imglist_train, file_imglist_val, file_imglist_test, prefix
