from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import csv

options = webdriver.ChromeOptions()
options.add_argument("--headless=new") # 크롤링 창 숨기기(백그라운드 작동)

driver = webdriver.Chrome(options = options)
driver.get('https://smartclean.nowon.kr/online/bulky/item') # 스마트클린 노원 대형폐기물 분리배출 신청 사이트
time.sleep(6) # 페이지 전체 로딩 여유 시간

# 표 css 선택자로 가져옴
table = driver.find_element(By.CSS_SELECTOR, ".tui-grid-rside-area > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > table:nth-child(1)")
text_raw_dict: list = table.text.splitlines()

driver.quit() # 크롤링 종료

csv_ = ['분류','품목','규격','가격']
tmp = []
cnt = 1

FILE_PATH = "대형폐기물분류표_노원_crawler.csv"
with open(FILE_PATH, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(csv_)
    for i in text_raw_dict:
        i.replace("(1599-0903 단일 무상 수거 품목)", "")
        if cnt % 4 != 0:
            tmp.append(i)
        else:
            tmp.append(int(i.replace(',', '')))
            w.writerow(tmp)
            tmp.clear()
        cnt += 1

# f = open("대형폐기물분류표_노원_crawler.csv", 'r')
# r = csv.reader(f)
# for i in r:
#     print(i)