import pandas as pd
import pandas_datareader.data as web  # pip install pandas_datareader
import datetime
import sqlite3
import numpy as np

# DB연동
conn = sqlite3.connect("kospi.db")

# DataReader를 통해 GS 종목(종목 코드: '078930.KS')의 일별 데이터를 DataFrame 형태로 다운로드합니다.
start = datetime.datetime(2021, 4, 1)
end = datetime.datetime(2021, 4, 30)
df = web.DataReader("078930.KS", "yahoo", start, end)

# DB에 종목 결과 INSERT하기
df.to_sql('078930', conn, if_exists='replace')  # 테이블이 존재할 경우, 값을 교체

# DB로부터 종목 결과 불러들이기
# readed_df = pd.read_sql("SELECT * FROM '078930'", conn, index_col = 'Date')
readed_df = pd.read_sql("SELECT * FROM '078930'", conn)

# 종가 담기
df_use = readed_df.Close

# 종가를 배열에 담기
array_close = np.array(df_use)

# 종가 배열데이터 출력
print('종가 배열데이터:', array_close)

# 날짜 배열데이터 출력
array_date_max = len(array_close)
array_date = np.array(range(array_date_max))
print('날짜 배열데이터:', array_date)

# 각 배열데이터들을 스트링의 형태로 바꾸고 ,를 붙인다.
array_date_x = []
array_close_y = []
for i in array_date:
    array_date_x.append(str(i+1)+',')  # 인덱스는 0부터 시작하므로 1부터 시작하도록 변경
for i in array_close:
    array_close_y.append(str(i)+',')

# 배열의 마지막 인덱스의 값은 ,를 포함치 않도록 한다.
array_date_x[-1] = array_date_x[-1].replace(',', '')
array_close_y[-1] = array_close_y[-1].replace(',', '')

# 날짜 데이터를 data_x.txt 에 삽입한다.
f = open("data_x.txt", 'w')
array_date_x.insert(0, '[Data x value : Date]\n')
for i in array_date_x:
    if(int(i) < 16):
        f.write(i)
    else:
        break
f.close()

# 종가 데이터를 data_y.txt 에 삽입한다.
f = open("data_y.txt", 'w')
array_close_y.insert(0, '[Data y value : Close]\n')
for i in range(len(array_close_y)):  # for(let i=0; i < list.length; i++) {} 과 동일
    if(i < 15):
        f.write(array_close_y[i])
    else:
        break

array_date_x.insert(0, '[Data x value : Date]\n')
for i in array_date_x:
    f.write(i)
f.close()

# 실제 날짜 데이터를 data_x_real.txt 에 삽입한다.
f = open("data_x_real.txt", 'w')
array_close_y.insert(0, '[Original Data x value : Date]\n')

# 실제 종가 데이터를 data_x_real.txt 에 삽입한다.
f = open("data_y_real.txt", 'w')
array_close_y.insert(0, '[Original Data y value : Close]\n')
for i in array_close_y:
    f.write(i)
f.close()
