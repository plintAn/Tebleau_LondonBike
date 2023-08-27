# 런던 자전거 데이터 - 공유 시각화
Python을 통해 데이터에 대한 이해와 간단한 데이터 처리 후,이를 기반으로 태블로를 통한 시각화를 전개합니다

# 데이터 세트 설명

본 라이센스 이용 약관은 TfL의 무료 교통 데이터 서비스에 적용되며 Transport for London("라이센스")에 대한 특정 개정 사항이 포함된 Open Government 라이센스 버전 2.0을 기반으로 합니다. TfL은 언제든지 통지 없이 본 라이센스를 개정할 수 있습니다. 

데이터 세트 출처(https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

데이터 목록
* london_merged.csv 



<!-- 목차 -->

# 차 례

| 번호 | 내용                                             |
|------|--------------------------------------------------|
| 1  | [데이터 로드](#1)                                  |
| 2  | [데이터 확인 및 처리](#2)                         |
| 3  | [태블로 연결결](#3)                         |
| 4  | [데이터 전처리](#4)                            |
| 5  | [회귀 분석](#5)                                |
| 6  | [모델 평가](#6)                                |
| 7  | [예측](#7)                                     |
| 8  | [구글 스튜디오](#8)                            |




<!-- intro -->
<div id="1">

## 1.데이터 로드

임포트

```python
import pandas as pd
import zipfile
import kaggle
```
캐글에서 가져옵니다

```python
# zipfile_name 변수에 압축 파일 이름을 할당합니다.
zipfile_name = 'london-bike-sharing-dataset.zip'

# with 문을 사용하여 zipfile_name을 읽기 모드('r')로 엽니다.
with zipfile.ZipFile(zipfile_name, 'r') as file:
    # 압축 해제 메서드를 사용하여 모든 파일과 폴더를 현재 작업 디렉터리에 압축 해제합니다.
    file.extractall()
```

```python
df = pd.read_csv("../london_merged.csv")
```

</div>

<div id="2">

## 2.데이터 확인 및 처리

```python
df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 17414 entries, 0 to 17413
Data columns (total 10 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   timestamp     17414 non-null  object 
 1   cnt           17414 non-null  int64  
 2   t1            17414 non-null  float64
 3   t2            17414 non-null  float64
 4   hum           17414 non-null  float64
 5   wind_speed    17414 non-null  float64
 6   weather_code  17414 non-null  float64
 7   is_holiday    17414 non-null  float64
 8   is_weekend    17414 non-null  float64
 9   season        17414 non-null  float64
dtypes: float64(8), int64(1), object(1)
memory usage: 1.3+ MB
```

```python
df.shape

(17414, 10)
```

```python
df.describe()
```

```python

cnt	t1	t2	hum	wind_speed	weather_code	is_holiday	is_weekend	season
count	17414.000000	17414.000000	17414.000000	17414.000000	17414.000000	17414.000000	17414.000000	17414.000000	17414.000000
mean	1143.101642	12.468091	11.520836	72.324954	15.913063	2.722752	0.022051	0.285403	1.492075
std	1085.108068	5.571818	6.615145	14.313186	7.894570	2.341163	0.146854	0.451619	1.118911
min	0.000000	-1.500000	-6.000000	20.500000	0.000000	1.000000	0.000000	0.000000	0.000000
25%	257.000000	8.000000	6.000000	63.000000	10.000000	1.000000	0.000000	0.000000	0.000000
50%	844.000000	12.500000	12.500000	74.500000	15.000000	2.000000	0.000000	0.000000	1.000000
75%	1671.750000	16.000000	16.000000	83.000000	20.500000	3.000000	0.000000	1.000000	2.000000
max	7860.000000	34.000000	34.000000	100.000000	56.500000	26.000000	1.000000	1.000000	3.000000
```



df 데이터를 확인 결과 'df.describe()', 'season' 경우 날씨와 같은 환경과 관련된 변수로 확인됨.

```python
df['weather_code'].value_counts()
```
```python
weather_code
1.0     6150
2.0     4034
3.0     3551
7.0     2141
4.0     1464
26.0      60
10.0      14
Name: count, dtype: int64
```

```python
df['season'].value_counts()
```

```python
df.columns
```
```python
Index(['timestamp', 'cnt', 't1', 't2', 'hum', 'wind_speed', 'weather_code',
       'is_holiday', 'is_weekend', 'season'],
      dtype='object')
```

컬럼 확인 후 그대로 복사하여 열을 rename 하여 임의로 편의에 따라 변경.

```python
df = df.rename(columns = {
    'timestamp':'time', 
    'cnt':'count', 
    't1':'temp_real_C', 
    't2':'temp_feels_like_C', 
    'hum':'humidity_percent', 
    'wind_speed':'wind_speed_kph', 
    'weather_code':'weather',
    'is_holiday':'is_holiday', 
    'is_weekend':'is_weekend', 
    'season':'season'})
```

소수점 단위 퍼센테이지로 변경.

```python
df['humidity_percent'] = df['humidity_percent'] / 100
```

자전거 이용에 따른 환경 날씨 변수들을 편의에 따라 'float' to  'str'로 변환.

```python
# 계절에 대한 숫자와 문자열을 매핑하는 딕셔너리를 생성합니다.
season_dict = {
    '0.0':'spring',
    '1.0':'summer',
    '2.0':'autumn',
    '3.0':'winter'
}

# 날씨 상태에 대한 숫자와 문자열을 매핑하는 딕셔너리를 생성합니다.
weather_dict = {
    '1.0':'Clear',
    '2.0':'Scattered clouds',
    '3.0':'Broken clouds',
    '4.0':'Cloudy',
    '7.0':'Rain',
    '10.0':'Rain with thunderstorm',
    '26.0':'Snowfall'
}

# df의 season 컬럼의 데이터 타입을 문자열(str)로 변환합니다.
df.season = df.season.astype('str')
# season 컬럼의 값들을 season_dict를 사용하여 문자열로 매핑합니다.
df.season = df.season.map(season_dict)

# df의 weather 컬럼의 데이터 타입을 문자열(str)로 변환합니다.
df.weather = df.weather.astype('str')
# weather 컬럼의 값들을 weather_dict를 사용하여 문자열로 매핑합니다.
df.weather = df.weather.map(weather_dict)
```
OutPut

```python
time	count	temp_real_C	temp_feels_like_C	humidity_percent	wind_speed_kph	weather	is_holiday	is_weekend	season
0	2015-01-04 00:00:00	182	3.0	2.0	0.930	6.0	Broken clouds	0.0	1.0	winter
1	2015-01-04 01:00:00	138	3.0	2.5	0.930	5.0	Clear	0.0	1.0	winter
2	2015-01-04 02:00:00	134	2.5	2.5	0.965	0.0	Clear	0.0	1.0	winter
3	2015-01-04 03:00:00	72	2.0	2.0	1.000	0.0	Clear	0.0	1.0	winter
4	2015-01-04 04:00:00	47	2.0	0.0	0.930	6.5	Clear	0.0	1.0	winter
```

이를 최종적으로 저장

```python
df.to_excel('london_bikes_final.xlsx', sheet_name='Data')
```

</div>

3.태블로 연결

태블로 퍼블릭에 london_bikes_final.csv 파일을 연결합니다

![Tableau](https://github.com/plintAn/Tebleau_LondonBike/assets/124107186/cd94718a-dea6-4af1-992d-80f1dc3f618f)

3.1 매개변수 생성

그리고 시트1을 선택 후 매개 변수(Moving Everage Period, Moving Everage Duration) 생성합니다.

![Tableau](https://github.com/plintAn/Tebleau_LondonBike/assets/124107186/b064ce99-a4a3-44b3-baf8-860522847b86)

3.2 계산 필드 작성

![Tableau_2](https://github.com/plintAn/Tebleau_LondonBike/assets/124107186/cd720394-be1f-4aea-a4ac-65ca9824a383)

3.3 행,열 드롭

열 필드 : Moving Average Period
행 : Count

![Tableau_3](https://github.com/plintAn/Tebleau_LondonBike/assets/124107186/abc90395-7aa5-41df-8d13-c0b28a5f9c9c)

3.4 Min, Max Month 최대 최소 월 계산 필드

![image](https://github.com/plintAn/Tebleau_LondonBike/assets/124107186/8cc686c4-4a59-4c26-8085-fab6c70a9b54)

* Min 계산 필드{ MIN(IF [Moving Average Period 집합] THEN [Moving Average Period] END) }
* Max 계산 필드{ MAX(IF [Moving Average Period 집합] THEN [Moving Average Period] END) }

생상 후 Min, Max 계산 필드를 마크 > 세부정보 > 드롭, Moving Average Period 집합(집합 표시 체크)

Update Moving Average Period Set

3.5 Action(액션) 데이터 셋 추가

상단 워크시트 > 동작 > 동작 추가 > 집합 값 변경 > 대상 집합(DaTa(Lodon_bike_final) > 선택을 해제할 경우의 결과 > 집합 값 유지 > 생

Update Moving Average Period Set

3.6 참조 구간 설정(Min, Max)

왼쪽 상단 > 분석 > 참조 구간을 시트에 드래그 > 설정 

드래그 후 특정 그래프 기간의 Min, Max 값을 볼 수 있습니다.

![Tableau_4](https://github.com/plintAn/Tebleau_LondonBike/assets/124107186/22df8460-0ddf-486a-831f-fc3d7a3de8fe)



<div id="3">

</div>

<div id="4">

</div>

<div id="5">

</div>

<div id="6">

</div>

<div id="7">

</div>

<div id="8">

