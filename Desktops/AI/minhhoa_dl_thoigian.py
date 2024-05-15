from datetime import datetime
import pandas as pd

dataset = pd.DataFrame({'created': ['2021-08-13 00:00:00', '2021-08-12 00:00:00', '2021-08-11 00:00:00', 
                                    '2021-08-10 00:00:00', '2021-08-09 00:00:00', '2021-08-08 00:00:00', '2021-08-07 00:00:00']})

def parser(x):
    # Để biết được định dạng strftime của một chuỗi kí tự ta phải tra trong bàng string format time
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

dataset['created'] = dataset['created'].map(lambda x: parser(x))
print(dataset['created'].dtypes)
