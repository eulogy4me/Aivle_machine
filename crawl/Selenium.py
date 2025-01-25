from seleniumwire import webdriver  # selenium-wire를 사용하도록 수정
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
import time
import json
from webdriver_manager.chrome import ChromeDriverManager
from io import BytesIO
import gzip
from xpath import *

class SeleniumParser():
    BASE_URL = "https://www.i-sh.co.kr/houseinfo/map"

    def __init__(self):
        opts = Options()
        opts.add_argument('--headless')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument('--enable_har')
        opts.add_argument("--disable-gpu")
        opts.add_argument('--ignore-certificate-errors')
        opts.add_argument('--ignore-ssl-errors')
        opts.add_argument("--auto-open-devtools-for-tabs")

        # selenium-wire를 사용하도록 설정
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts
        )

        # CDP 명령어로 네트워크 캡처 활성화
        self.driver.get('https://www.i-sh.co.kr/houseinfo/main/mainPage.do')
        self.driver.maximize_window()

        # 응답 데이터를 저장할 딕셔너리
        self.data = {
            "house_details": [],
            "management_costs": [],
            "house_supply_types": []
        }

        # 네트워크 요청 응답 캡처를 위한 콜백 함수 설정
        # request_interceptor를 사용하지 않고, 응답만 처리하도록 수정
        self.driver.response_interceptor = self._capture_response

    def save_json(self, name="response_data"):
        with open(f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"Saved JSON data to {name}.json")

    def action_click(self, element):
        webdriver.ActionChains(self.driver).move_to_element(element).perform()
        self.driver.execute_script("arguments[0].click();", element)

    def _capture_response(self, request, response):
        """
        특정 URL에 대한 응답을 캡처하여 딕셔너리에 저장합니다.
        """
        url = request.url
        try:
            if "selectHouseBscDetail.do" in url:
                response_body = self._decode_response_body(response.body, response.headers)
                house_details = json.loads(response_body)
                self.data["house_details"].append(house_details)
            elif "selectHouseHsTyDetailInfo.do" in url:
                response_body = self._decode_response_body(response.body, response.headers)
                house_supply_types = json.loads(response_body)
                self.data["house_supply_types"].append(house_supply_types)
            elif "selectManaCostInfo.do" in url:
                response_body = self._decode_response_body(response.body, response.headers)
                management_costs = json.loads(response_body)
                self.data["management_costs"].append(management_costs)
        except json.JSONDecodeError:
            print(f"Response for {url} is not valid JSON.")
        except Exception as e:
            print(f"Error processing response for {url}: {e}")

    def _decode_response_body(self, body, headers):
        """
        응답 본문을 디코딩합니다. (gzip 디코딩 처리)
        """
        # 응답 헤더에서 압축 여부 확인
        content_encoding = headers.get('Content-Encoding', '')
        if 'gzip' in content_encoding:
            return self._gunzip(body)
        else:
            # 기본적으로 바이트 스트림을 문자열로 변환
            return body.decode('utf-8')

    def _gunzip(self, body):
        """
        gzip으로 압축된 데이터를 디코딩합니다.
        """
        buf = BytesIO(body)
        with gzip.GzipFile(fileobj=buf) as f:
            return f.read().decode('utf-8')

    def prepare(self):
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, BOARD_AREA))
        )
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, ANNOUNCE_ITEM))
        )
        announce_item = self.driver.find_element(By.XPATH, ANNOUNCE_ITEM)
        print("Get announce_item")
        self.action_click(announce_item)
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, ANNOUNCE_TITLE))
        )
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, NAVIGATOR_CONTAINER))
        )
        pages = self.driver.find_elements(By.XPATH, NAVIGATORS)
        print(f"Total Pages: {len(pages)}")

        for i in range(2, len(pages)):
            jutak_items = self.driver.find_elements(By.XPATH, JUTAK_ITEMS)
            for item in jutak_items:
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, JUTAK))
                )
                jutaks = item.find_elements(By.TAG_NAME, 'a')

                for jutak in jutaks:
                    self.action_click(jutak)

                    WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, INFO_TYPE_SPLY_BTN))
                    )
                    splyty_btn = self.driver.find_element(By.XPATH, INFO_TYPE_SPLY_BTN)
                    WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, INFO_TYPE_MANAGE_BTN))
                    )
                    manage_btn = self.driver.find_element(By.XPATH, INFO_TYPE_MANAGE_BTN)
                    self.action_click(splyty_btn)
                    self.action_click(manage_btn)
                    
                    print("Web Clicking Jutak Info")

            if i != len(pages):
                page = pages[i]
                self.action_click(page)
                time.sleep(2)
                pages = self.driver.find_elements(By.XPATH, NAVIGATORS)
                print(f"Pages Refreshed: {len(pages)}")

        # 마지막에 데이터를 JSON으로 저장
        self.save_json()

        print("Web Access Successfully")
        self.driver.quit()

if __name__ == "__main__":
    try:
        parser = SeleniumParser()
        parser.prepare()
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")
