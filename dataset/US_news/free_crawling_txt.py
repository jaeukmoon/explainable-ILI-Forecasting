import time
import random
import datetime
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from dateutil.relativedelta import relativedelta
from selenium.webdriver.chrome.options import Options

options=Options()
options.add_argument('--headless')
driver=webdriver.Chrome(options=options)

#검색 년, 월, 키워드
keyword="influenza"

start_year=2024
start_month=10

end_year=2024
end_month=10

#날짜 계산 함수 구현
# 1.일주일의 시작을 월요일->일요일
# 2.매년 첫주차의 기준을 첫번째 목요일->수요일 이 포함된 주차로함
def week_num(date):
    year=date.year
    month=date.month

    first_day_weekday=datetime.date(year,1,1).weekday()
    after_first_day_weekday=datetime.date(year+1,1,1).weekday()
    
    last_week=datetime.date(year,12,28).isocalendar().week
    before_last_week=datetime.date(year-1,12,28).isocalendar().week

    if first_day_weekday==3:
        date_week=(date+datetime.timedelta(days=1)).isocalendar().week-1
        if date_week==0:
            date_week=before_last_week+1
    elif after_first_day_weekday==3:
        date_week=(date+datetime.timedelta(days=1)).isocalendar().week
        if month==12 and date_week==1:
            date_week=last_week+1
    else:
        date_week=(date+datetime.timedelta(days=1)).isocalendar().week
    
    return date_week

#검색 하기위한 변수 수정
start_day=datetime.date(start_year,start_month,7)#일자
end_day=datetime.date(end_year,end_month,1)+relativedelta(months=1)-datetime.timedelta(days=1)

start_day_sunday=start_day-datetime.timedelta(days=start_day.weekday()+1) #일요일로 맞춰줌, weekday는 월요일기준이기에 -1을 더 해줌

while start_day_sunday+datetime.timedelta(days=6)<=end_day: #일주일 단위로 날짜를 슬라이싱
    #검색할 날짜 url형식으로 변환
    search_start_day=f'{start_day_sunday.month}/{start_day_sunday.day}/{start_day_sunday.year}'
    end_day_saturday=start_day_sunday+datetime.timedelta(days=6)
    search_end_day=f'{end_day_saturday.month}/{end_day_saturday.day}/{end_day_saturday.year}'

    base_xpath='//*[@id="rso"]/div/div/div'#링크 따오는 기본 주소
    url_save=[]#링크 저장하는 배열

    break_count=0
    for i in range(10):#최대 10페이지 정보를 추출
        url=f'https://www.google.com/search?q={keyword}&hl=en&gl=us&tbm=nws&tbs=cdr:1,cd_min:{search_start_day},cd_max:{search_end_day}&start={i*10}'
        driver.get(url)
        time.sleep(random.uniform(0.8,1.5))
        for j in range(1,11):
            try:
                xpath=f'{base_xpath}[{j}]/div/div/a'
                temp=driver.find_element(By.XPATH,xpath)
                href=temp.get_attribute('href')
                url_save.append(href)
            except Exception as ex:
                print('해당페이지 없음')
                break_count=1
                break
        if break_count==1:
            break
    
    #년도,주차 계산
    save_year=start_day_sunday.year
    save_week=week_num(start_day_sunday)
    print(f" {save_year}년도 {save_week}주차 데이터 수집 ({start_day_sunday} -> {start_day_sunday+datetime.timedelta(days=6)})")

    texts=''
    # 1개 주차 url들에서 텍스트 추출 후 texts변수에 모두 저장
    full_count=len(url_save)
    count=0
    for temp_url in url_save:
        count+=1
        try:
            headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'}
            response = requests.get(temp_url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding='utf-8-sig'
            data = requests.get(temp_url, headers=headers)
            soup = BeautifulSoup(data.text, 'html.parser')
            text=' '.join(soup.text.split())
            texts+=text
            print(f"{save_year}년 {save_week}주차 저장중 : {count}/{full_count}")
        except Exception as ex:
            print("pass")
            continue

    print('끝')

    texts=texts.replace('\n','')
    texts=texts.replace('\r','')
    texts=texts.replace('\t','')
    texts=texts.replace('\r\n','')
    
    texts=texts.replace('/n','')
    texts=texts.replace('/r','')
    texts=texts.replace('/t','')
    texts=texts.replace('/r/n','')

    # 테스트 글자수 손실없이 저장이 되는지 검증 = 검증완 주차별 24만 글자~ 80만 글자
    # with open('test.txt',"w",encoding="utf-8")as file:
    #     file.write(texts)

    #해당 주차 csv파일에 저장 /년도,주차에 맞는 행에 저장
    save_week="{0:02d}".format(save_week)
    with open(f'{save_year}{save_week}.txt','w',encoding='utf-8') as file:
        file.write(texts)

    time.sleep(random.uniform(0.8,1.2))
    print(f"------{save_year}--{save_week}-------")
    start_day_sunday+=datetime.timedelta(days=7)

time.sleep(random.uniform(3,5))
