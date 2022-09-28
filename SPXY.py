def spxy(x, y, test_size=0.2):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size
    :return: spec_train :(n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    x_backup = x
    y_backup = y
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    y = (y - np.mean(y)) / np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = x[i, :]
        ya = y[i,:]
        for j in range((i + 1), M):
            xb = x[j, :]
            yb = y[j,:]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]

    print(spec_train.shape)
    print(spec_test.shape)
    print(target_train.shape)
    print(target_test.shape)

    #print(target_test)
    #spec_train = spec_train.data.numpy()
    #spec_train = np.array(spec_train)
    filename = xlwt.Workbook()  # 创建工作簿
    sheet1 = filename.add_sheet(u'sheet1', cell_overwrite_ok=True)
    [h, l] = spec_train.shape
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, str(spec_train[i, j]))
    filename.save('E:/小论文资料/程序/CARS-master/SPXYdata/mtl_523_spxy.xlsx')

    filename = xlwt.Workbook()  # 创建工作簿
    sheet1 = filename.add_sheet(u'sheet1', cell_overwrite_ok=True)
    [h1, l1] = spec_test.shape
    for i1 in range(h1):
        for j1 in range(l1):
            sheet1.write(i1, j1, str(spec_test[i1, j1]))
    filename.save('E:/小论文资料/程序/CARS-master/SPXYdata/mtl_131_1_spxy.xlsx')

    filename = xlwt.Workbook()  # 创建工作簿
    sheet1 = filename.add_sheet(u'sheet1', cell_overwrite_ok=True)
    [h2, l2] = target_train.shape
    for i2 in range(h2):
        for j2 in range(l2):
            sheet1.write(i2, j2, str(target_train[i2, j2]))
    filename.save('E:/小论文资料/程序/CARS-master/SPXYdata/mtl_523_label_spxy.xlsx')

    filename = xlwt.Workbook()  # 创建工作簿
    sheet1 = filename.add_sheet(u'sheet1', cell_overwrite_ok=True)
    [h3, l3] = target_test.shape
    for i3 in range(h3):
        for j3 in range(l3):
            sheet1.write(i3, j3, str(target_test[i3, j3]))
    filename.save('E:/小论文资料/程序/CARS-master/SPXYdata/mtl_131_1_label_spxy.xlsx')


    return spec_train, spec_test, target_train, target_test

import numpy as np
import pandas as pd
import xlwt
import scipy.io as scio
X=pd.read_csv(r'E:\小论文资料\数据\data\MTL_654.csv')
Y=pd.read_csv(r'E:\小论文资料\数据\data\MTL_654_label.csv')
print(X.shape)
print(Y.shape)
x=np.array(X)
y=np.array(Y)
SPXY = spxy(x,y,test_size=0.2)
