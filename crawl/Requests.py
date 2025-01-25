import requests
import json

class HouseDataFetcher():
    BASE_URL = "https://www.i-sh.co.kr/houseinfo/map"
    HEADERS = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Ajax": "true",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Dnt": "1",
        "Host": "www.i-sh.co.kr",
        "Origin": "https://www.i-sh.co.kr",
        "Pragma": "no-cache",
        "Referer": "https://www.i-sh.co.kr/houseinfo/main/mainPage.do",
        "Sec-CH-UA": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
    }
        
    def __init__(self):
        self.session = requests.Session()
        self.session.headers = self.HEADERS
        self.local_storage = None
        self.session_storage = None
        
    def update(self, session_data):
        if "cookies" in session_data:
            self.session.cookies.update(session_data["cookies"])
        if "localStorage" in session_data:
            self.local_storage = session_data["localStorage"]
        if "sessionStorage" in session_data:
            self.session_storage = session_data["sessionStorage"] 

    '''
    Request Functions
    ''' 
            
    def save(self, name, data):
        """ 저장 """
        with open(f'{name}.json', 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    def post_request(self, url, payload=None):
        try:
            if payload:
                response = self.session.post(url, headers=self.HEADERS, json=payload)
            else:
                response = self.session.post(url, headers=self.HEADERS)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 오류: {e}")
        except ValueError as e:
            print(f"JSON 처리 오류: {e}")
        
        return None
    
    def fetch_announcement_list(self):
        """ 메인 페이지 공고들 추출 """
        url = f"{self.BASE_URL}/selectNoticeInProgressList.do"
        return self.post_request(url)
    
    def post_manage_cost(self, biznsCd):
        """ 관리비 데이터 추출 """
        url = f"{self.BASE_URL}/selectManaCostInfo.do"
        payload = {
            "biznsCds": biznsCd,
            "searchYear": "",
            "searchMonth": ""
        }
        data = self.post_request(url, payload)
        if not data:
            print(f"{biznsCd} : ManageCost is Empty")
        
        return self.post_request(url, payload)
    
    def post_house_type(self, biznsCd, hscl, splyty):
        """ 공급 형별 정보 추출 """
        url = f"{self.BASE_URL}/selectHouseHsTyDetailInfo.do"
        
        payload = {
            "biznsCds": biznsCd,
            "hsCl": hscl,
            "splyTy": splyty.strip()
        }
        data = self.post_request(url, payload)
        if not data:
            print(f"{biznsCd} : HouseType is Empty")

        return self.post_request(url, payload)
    
    def post_house_detail(self, biznsCd):
        """ 집 상세 정보 추출 """
        url = f"{self.BASE_URL}/selectHouseBscDetail.do"
        payload = {
            "biznsCd": biznsCd,
            "ctgrCd": 10
        }
        
        house_detail_data = self.post_request(url, payload)
        result = house_detail_data["result"]
        
        if result is None:
            print(f"HouseDetail : 'result' is None for biznsCd {biznsCd}")
            return None

        hsTy = result.get("hsTy", "")
        hsTyLst = result.get("hsTyLst", "").split(",") if result.get("hsTyLst") else []

        house_type_data = []
        for splyty in hsTyLst:
            house_type_response = self.post_house_type(biznsCd, hsTy, splyty)
            if house_type_response:
                house_type_data.append(house_type_response)

        return {"house_detail": house_detail_data, "house_type": house_type_data}
    
    def fetch_and_save_data(self):
        print("Fetching announcements list...")
        
        anno_list_data = self.fetch_announcement_list()
        anno_list_data = anno_list_data["result"]
        selected_anno = [item for item in anno_list_data if "청년안심주택" in item.get("recrnotiNm", "")]
        if not selected_anno:
            print("조건에 맞는 공고가 없습니다.")
            return

        biznsCds = selected_anno[0].get("biznsCd")
        if not biznsCds:
            print("선택된 공고에 biznsCds 정보가 없습니다.")
            return

        print(f"선택된 공고: {selected_anno[0]}")

        all_data = {}
        for biznsCd in map(str, biznsCds.split(',')):
            print("Run Post_house_detail")
            house_detail_and_type = self.post_house_detail(biznsCd)
            if house_detail_and_type:
                print("Run Post_manage_cost")
                manage_cost = self.post_manage_cost(biznsCd)
                all_data[biznsCd] = {
                    "house_detail_and_type": house_detail_and_type,
                    "manage_cost": manage_cost
                }
        print("HouseData Saved")
        self.save("All_House_Data", all_data)