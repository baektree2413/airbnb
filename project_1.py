#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정 (윈도우 기준: 맑은 고딕)
plt.rc('font', family='Malgun Gothic')

# 음수 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# In[2]:


import pandas as pd


# In[3]:


#파일 가져오기
df = pd.read_csv('2025_Airbnb_NYC_listings.csv')  # 업로드한 파일명 그대로 입력
df.head(5)


# In[4]:


df


# In[5]:


df.columns


# In[6]:


#삭제전 컬럼 개수
df.shape


# In[7]:


host_df = df.copy()


# In[8]:


#호스트 관련 host_df
host_related_columns = [
    'host_id',
    'host_name',
    'host_since',
    'host_location',
    'host_response_time',
    'host_response_rate',
    'host_acceptance_rate',
    'host_is_superhost',
    'host_listings_count',
    'host_identity_verified'
    ]

host_df = df[host_related_columns]
host_df


# In[9]:


#위치 관련 place_df
place_df = df.copy()
place_related_columns = [
    'neighbourhood',
    'neighbourhood_group_cleansed',
    'latitude',
    'longitude'
    ]
place_df = df[place_related_columns]
place_df


# In[10]:


#숙소 정보 관련 info_df
information_related_columns = [
    'property_type',
    'room_type',
    'accommodates',
    'bedrooms',
    'beds',
    'bathrooms',
    'amenities']
info_df = df[information_related_columns]
info_df.head(2)


# In[11]:


#가격 및 예약 관련 컬럼 price_df
price_related_columns = [
    'price',
    'minimum_nights',
    'maximum_nights',
    'availability_365',
    'instant_bookable'

]
price_df = df[price_related_columns]
price_df.head(2)


# In[12]:


#리뷰 및 평점 관련 review_df
review_related_columns = [
    'number_of_reviews',
    'reviews_per_month',
    'review_scores_rating',
    'review_scores_cleanliness',
    'review_scores_communication',
    'first_review',
    'last_review'
]

review_df = df[review_related_columns]
review_df.head(2)


# In[13]:


#df 정리
'''
#호스트 관련 host_df
#위치 관련 place_df
#숙소 정보 관련 property_df 
#가격 및 예약 관련 컬럼 price_df
#리뷰 및 평점 관련 review_df

'''


# In[14]:


info_df['property_type'].unique()


# In[15]:


def categorize_property_type(value):
    value = value.lower()
    if 'private room' in value:
        return 'private'
    elif 'shared room' in value:
        return 'shared'
    elif 'entire' in value:
        return 'entire'
    elif 'hotel' in value:
        return 'hotel'
    else:
        return 'other'

info_df['property_category'] = info_df['property_type'].apply(categorize_property_type)


# In[16]:


info_df.head(5)


# In[17]:


property_df = info_df


# In[18]:


property_df['property_category'].value_counts()


# In[19]:


property_df[property_df['property_category'] == 'other']


# In[20]:


property_df.loc[property_df['property_type'] == 'Room in serviced apartment', 'property_category'] = 'hotel'


# In[21]:


property_df['room_type'].value_counts()


# In[22]:


property_df['property_category'].value_counts()


# In[23]:


property_df['property_category'].value_counts(normalize=True)*100


# In[24]:


import matplotlib.pyplot as plt

#한글 표시 관련
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 숙소유형별 개수 시각화
order = ['Private room', 'Entire home/apt', 'Hotel room', 'Shared room'] 
counts = property_df['room_type'].value_counts()
counts = counts.reindex(order)

# 비율 구하기
total = counts.sum()
percentages = counts / total * 100

ax = counts.plot(kind='bar', color='skyblue')
plt.title('숙소 유형별 개수')  
plt.xlabel('숙소 카테고리')
plt.ylabel('숙소 개수')
plt.xticks(rotation=45)

# 막대 위에 비율 표시
for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax.text(i, count + total*0.01, f'{pct:.1f}%', ha='center')

plt.tight_layout()
plt.show()


# In[25]:


price_df


# In[26]:


df_property_price = df[['room_type', 'price']]


# In[27]:


df_property_price


# In[28]:


df_property_price.info()


# In[29]:


df_property_price['price'] = (
    df_property_price['price']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False) #정규표현식이 아니라 문자 그대로 $로 인식
    .astype(float)
)


# In[30]:


import matplotlib.pyplot as plt

# 숙소 유형 평균 가격 계산
property_mean = df_property_price.groupby('room_type')['price'].mean().round(2)

# 그래프 크기 설정
plt.figure(figsize=(8, 5))

# 순서 설정
order = ['Private room', 'Entire home/apt', 'Hotel room', 'Shared room']
property_mean = property_mean.reindex(order)

# 막대 위에 수치 표시
ax = property_mean.plot(kind='bar', color='skyblue')
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height * 1.01, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=10)

# 그래프 제목과 축 이름
plt.title('숙소 유형별 평균 가격')
plt.xlabel('숙소 유형')
plt.ylabel('평균가격($)')

plt.xticks(rotation=45)
plt.show()



# In[31]:


property_mean = df_property_price.groupby('room_type')['price'].mean().round(2)
property_mean


# In[32]:


df_property_price


# 숙소유형별 가격의 분포도 확인하기

# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

# room type 필터링
filtered_df = df_property_price[df_property_price['room_type'].isin(['Private room', 'Entire home/apt'])]

sns.boxplot(data=filtered_df, x='room_type', y='price')
plt.ylim(0, 1000)


# In[34]:


#어느 가격대의 숙소가 많은지
#displot 히스토그램과 커널밀도추정을 포함한 분포 시각화
#kde 부드러운 곡선 형태의 커널밀도추정선, 데이터분포의 밀도(연속적 분포형태) 보여줌.
#col 룸타입열에 있는 룸타입별로 그래프를 분리해서 여러개를 그림
#col_wrap=2 2개씩 가로루 묶어서 여러 행으로 배치
#common_norm=True 전체 데이터 기준으로 정규화서 그려줌

sns.displot(data=filtered_df, 
             x='price',
             kde=False,
             col='room_type',
             col_wrap=2,
             common_norm=True)
plt.xlim(0, 1000)
plt.show()


# In[35]:


#숙소유형별 가격 밀집도 파악 가능
sns.stripplot(data=filtered_df, x='room_type', y='price', jitter=True, alpha=0.3)
plt.ylim(0, 2000)


# In[36]:


host_df.info()


# In[37]:


host_df.head(10)


# In[38]:


host_df['host_acceptance_rate'].nunique()


# In[39]:


host_df['host_acceptance_rate'].unique()


# In[40]:


host_df.info()


# In[41]:


host_df.isnull().sum()


# In[42]:


property_df['property_type'].nunique()


# In[43]:


property_df.info()


# In[44]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[45]:


pro_df = property_df
pro_df = pro_df.drop('property_category', axis=1)
pro_df['price'] = df['price']


# In[46]:


pro_df['price'] = pro_df['price'].replace('[\$,]', '', regex=True).astype(float)


# In[47]:


print(f"데이터 수: {pro_df.shape}")
pro_df.head(2)


# In[48]:


pro_df.info()


# 변수별 의미
# - property_type : 집 유형
# - room_type : 방 유형 (private, entire, hotel, shared)
# - accomodates : 수용가능한 인원
# - bedrooms : 침실
# - beds : 침대
# - bathrooms : 욕실
# - amenities : 어메니티 (히터, 티비, 드라이기, 와이파이 등)
# - price : 가격
# 

# In[49]:


#변수 분류

target_feature = ['price']
categorical_feature = [x for x in pro_df.columns if pro_df[x].dtype in ['object', 'category', 'bool']]
numerical_feature = [x for x in pro_df.columns if x not in categorical_feature and x not in target_feature]
print(f"target:{target_feature}")
print(f"categorical:{categorical_feature}")
print(f"numerical:{numerical_feature}")


# In[50]:


pro_df.info()


# In[51]:


pro_df.isnull().sum()

#


# In[52]:


pro_df['price'].describe()


# In[53]:


pro_df[numerical_feature].describe()


# In[54]:


#이산형변수
discrete_vars = ['accommodates', 'bedrooms', 'beds', 'bathrooms']

#시각화스타일
sns.set(style='whitegrid')
plt.figure(figsize=(16, 12))

#이산형 변수 시각화
for i, var in enumerate(discrete_vars):
    plt.subplot(2, 2, i+1) # 행, 열, 위치
    sns.countplot(data=df, x=var, color='skyblue')
    plt.title(f'Discrete Variable: {var}')
    plt.xlabel(var)
    plt.ylabel('Count')



# In[55]:


# 욕실수, 침실수, 침대수 중위값으로 결측치 채우기
pro_df['bathrooms'] = pro_df['bathrooms'].fillna(pro_df['bathrooms'].median())
pro_df['bedrooms'] = pro_df['bedrooms'].fillna(pro_df['bedrooms'].median())
pro_df['beds'] = pro_df['beds'].fillna(pro_df['beds'].median())


# In[56]:


#방 타입별 이산형 변수 평균/중위값 살펴보기
pro_df.groupby('room_type')[discrete_vars].agg([np.mean, np.median]).round(2)


# In[57]:


import seaborn as sns
import matplotlib.pyplot as plt

# 평균값 구하기
mean_df = pro_df.groupby('room_type')[discrete_vars].mean().reset_index()

# 데이터 변형
# melt는 여러 개의 열을 하나의 열로 합쳐서 변수명과 그 값을 표현하는 방식으로 데이터를 재구성
# id_vars = 'room_type' 이 열 그대로 유지
# var_name='variable': 새로 생길 열 이름 (이산형 변수 이름을 담음)
# value_name='mean_value': 새로 생길 값 열 이름 (평균값을 담음)

mean_df_melted = mean_df.melt(id_vars='room_type', var_name='variable', value_name='mean_value')

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(data=mean_df_melted, x='room_type', y='mean_value', hue='variable')
plt.title('Room Type별 수용가능원/침실/침대/욕실 수 평균 비교')
plt.ylabel('Mean Value')
plt.xlabel('Variable')
plt.legend(title='Room Type')
plt.tight_layout()
plt.show()



# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt

# 평균값 구하기
median_df = pro_df.groupby('room_type')[discrete_vars].median().reset_index()

# 데이터 변형
# melt는 여러 개의 열을 하나의 열로 합쳐서 변수명과 그 값을 표현하는 방식으로 데이터를 재구성
# id_vars = 'room_type' 이 열 그대로 유지
# var_name='variable': 새로 생길 열 이름 (이산형 변수 이름을 담음)
# value_name='mean_value': 새로 생길 값 열 이름 (평균값을 담음)

median_df_melted = median_df.melt(id_vars='room_type', var_name='variable', value_name='mean_value')

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(data=median_df_melted, x='room_type', y='mean_value', hue='variable')
plt.title('Room Type별 수용가능원/침실/침대/욕실 수 중위값 비교')
plt.ylabel('Mean Value')
plt.xlabel('Variable')
plt.legend(title='Room Type')
plt.tight_layout()
plt.show()



# In[59]:


import seaborn as sns
import matplotlib.pyplot as plt

# 평균값 구하기
median_df = pro_df.groupby('room_type')[discrete_vars].median()

# 전치
median_df_T = median_df.T

# 히트맵 시각화
plt.figure(figsize=(8, 5))
sns.heatmap(median_df_T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Room Type별 이산형 변수 평균 히트맵')
plt.ylabel('Room Type')
plt.xlabel('Variable')
plt.tight_layout()
plt.show()


# In[60]:


from scipy.stats import shapiro, normaltest
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

for var in ['accommodates', 'bedrooms', 'beds', 'bathrooms']:
    data = pro_df[var].dropna()

    print(f"\n📊 변수: {var}")

    # Q-Q Plot
    # 실제 분포와 이론적 정규분포의 분위를 비교
    # 점들이 대각선에 가까울수록 정규분포에 가깝다는 의미
    sm.qqplot(data, line='s')
    plt.title(f'Q-Q plot for {var}')
    plt.show()

    # 히스토그램 + 정규곡선
    # 실제 분포와 정규분포 곡선(norm.df)을 겹쳐 시각적으로 비교
    sns.histplot(data, kde=True, stat='density', bins=20, color='lightgray', label='실제 분포')
    x = np.linspace(data.min(), data.max(), 1000)
    from scipy.stats import norm
    plt.plot(x, norm.pdf(x, data.mean(), data.std()), 'r', label='정규분포', linewidth=2)
    plt.title(f'{var} - 분포 vs 정규분포')
    plt.xlabel(var)
    plt.ylabel('밀도')
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[61]:


get_ipython().system('pip install scipy')
get_ipython().system('pip install statsmodels')


# In[62]:


property_df.shape


# In[63]:


pro_df.columns


# In[64]:


# 선형 회귀 분석
# 독립변수 (수용인원, 침실수, 침대수, 욕실수)가 종속변수(가격)에 유의미한 영향을 주는지 확인

import numpy as np
import statsmodels.api as sm

# 종속변수 로그변환
# 가격은 일반적으로 양의 방향으로 치우친 분포를 가지는 경우가 많음. 그래서 log(price)를 사용하면, 분포가 정규분포에 가까워짐. 이상치의 영향이 줄어듦. 회귀계수가 비율적 해석이 가능해짐
pro_df['log_price'] =np.log(pro_df['price'])

# 변수 지정
X = pro_df[['accommodates', 'bedrooms', 'beds', 'bathrooms']]
y = pro_df['log_price']

# 상수항(절편) 추가 -> 절편을 넣어야지 회귀선이 데이터에 더 잘 맞음
X = sm.add_constant(X)

# 회귀 모델 적합
# 최소제곱법(OLS)사용 : 오차의 제곱의 합을 최소화헤서 회귀선을 찾음
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())


# In[65]:


#다중공선성 문제 확인 
#다중공선성은 독립변수들끼리 서로 상관이 높을 때 생기며, 회귀계수 해석이 불안정해질 수 있다. 
#이를 해결하기 위해 VIF를 확인 
#VIF > 5 다중공선성 의심
#VIF > 10 심각한 다중공선성
#이 수치상에서 bedrooms의 다중공선성 문제는 없는것으로 확인

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = pro_df[['accommodates', 'bedrooms', 'beds', 'bathrooms']]
X = add_constant(X)  # 상수항 추가
vif = pd.DataFrame()
vif['variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)


# In[66]:


# 따라서 bedrooms의 이상치를 제거해보고자 함
# 즉 침실 많고 가격 낮은 이상행 제거 후 재학습
import seaborn as sns
import matplotlib.pyplot as plt

# 이상치 포함한 박스플롯
plt.figure(figsize=(6, 4))
sns.boxplot(x=pro_df['bedrooms'])
plt.title("Boxplot of Bedrooms (with outliers)")
plt.show()



# In[67]:


# bedrooms 이상치 탐구

# bedrooms 이상치 탐지
Q1 = pro_df['bedrooms'].quantile(0.25)
Q3 = pro_df['bedrooms'].quantile(0.75)
IQR = Q3 - Q1

# 하한선, 상한선
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 제거된 DataFrame
pro_df_no_outliers = pro_df[(pro_df['bedrooms'] >= lower_bound) & (pro_df['bedrooms'] <= upper_bound)]

# 박스플롯 그리기
plt.figure(figsize=(6, 4))
sns.boxplot(x=pro_df_no_outliers['bedrooms'])
plt.title("Boxplot of Bedrooms (outliers removed)")
plt.show()


# In[68]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='bedrooms', data=pro_df_no_outliers)
plt.title("Bedroom Count (Outliers Removed)")
plt.show()


# In[69]:


#bedrooms 회귀계수가 음수로 튀는 원인을 파악하고
# 그것을 해결하기 위한 방법
# 
# pro_df['bedrooms'].value_counts()


# In[70]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# bedrooms별 평균 가격 및 개수 집계
# bedrooms 3~7구간은 bedrooms이 증가한다고 가격이 계속 올라가지 않는다는것을 뜻함. 회귀계수가 음수로 튈 수 있음.
bedroom_group = pro_df.groupby('bedrooms')['log_price'].agg(['mean']).reset_index()

# barplot 그리기
plt.figure(figsize=(8, 4))
sns.barplot(data=bedroom_group, x='bedrooms', y='mean', palette='viridis')
plt.title("Average log_price by Number of Bedrooms")
plt.xlabel("Bedrooms")
plt.ylabel("Average log_price")
plt.show()

#bedrooms 별 개수 분포 (희소성 확인용 표)
#4개 이상의 bedrooms는 전체 데이터의 3%미만
print((pro_df['bedrooms'].value_counts(normalize=True)*100).sort_index())

#bedrooms가 많아도 가격이 낮은 경우가 많다
plt.figure(figsize=(8, 4))
sns.scatterplot(data=pro_df, x='bedrooms', y='log_price', alpha=0.3)
plt.title("Scatterplot of Bedrooms vs. Log Price")
plt.xlabel("Bedrooms")
plt.ylabel("Log Price")
plt.show()




# In[71]:


# bedrooms - 범주형처리 4이상은 전체 데이터의 3%미만이었으므로 4+로 묶는다.
# 새 범주형 컬럼 생성
pro_df['bedrooms_cat'] = pro_df['bedrooms'].apply(
        lambda x: '0' if x == 0 else
                  '1' if x == 1 else
                  '2' if x == 2 else
                  '3' if x == 3 else
                  '4+'
)

#더비 변수 변환 
# drop_first=True는 더미변수 생성 시, 기준이 될 범주 하나를 자동으로 제외하라는 의미
# 이는 회귀 분석 시 다중공선성을 방지
pro_df = pd.get_dummies(pro_df, columns=['bedrooms_cat'], drop_first=True)


# In[72]:


import statsmodels.api as sm

# 독립변수 설정
X = pro_df[['accommodates', 'beds', 'bathrooms',
            'bedrooms_cat_1', 'bedrooms_cat_2', 'bedrooms_cat_3', 'bedrooms_cat_4+']]

# 숫자형으로 강제 변환
X = X.astype(float)

# 상수항 추가
X = sm.add_constant(X)

# 종속변수도 float 확인
y = pro_df['log_price'].astype(float)

# 회귀 실행
model = sm.OLS(y, X).fit()
print(model.summary())

# 예: room_type 더미 추가
pro_df = pd.get_dummies(pro_df, columns=['room_type'], drop_first=True)

#bedrooms는 “맥락에 따라” 영향력 달라짐 → 맥락 변수 없으면 왜곡


# In[ ]:




