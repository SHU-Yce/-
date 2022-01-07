import numpy as np
import xlrd

def read_data(filepath):
    work_book = xlrd.open_workbook(filepath)
    table = work_book.sheet_by_index(0)
    print(">>>数据集长度：", table.nrows-1)
    data = []
    labels = []
    t = [0, 2, 4, 6, 8, 10, 12]
    for i in range(1, table.nrows):
        rowValue = table.row_values(i)
        data_row = []
        for j in range(table.ncols):
            if rowValue[j] != '' and j != table.ncols-1:
                data_row.append((t[j], rowValue[j]))
            elif j == table.ncols-1:
                labels.append(int(rowValue[j]))
        data.append(data_row)
    return data, labels