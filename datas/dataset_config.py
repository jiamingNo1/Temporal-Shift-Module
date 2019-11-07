import os

ROOT_DATA = '/home/hjm/PycharmProjects/temporal-shift-module/datas'


def dataset():
    file_categories = 'jester/category.txt'
    prefix = '{:05d}.jpg'
    file_imglist_train = 'jester/train_videofolder.txt'
    file_imglist_val = 'jester/val_videofolder.txt'
    file_imglist_test = 'jester/test_videofolder.txt'
    file_imglist_train = os.path.join(ROOT_DATA, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATA, file_imglist_val)
    file_imglist_test = os.path.join(ROOT_DATA, file_imglist_test)

    file_categories = os.path.join(ROOT_DATA, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]

    n_class = len(categories)
    print('jester: {} classes'.format(n_class))
    return n_class, file_imglist_train, file_imglist_val, file_imglist_test, prefix
