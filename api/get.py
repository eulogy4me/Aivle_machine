import requests
import xml.etree.ElementTree as ET
import pandas as pd

class PublicData():
    def __init__(self, service) :
        self.service = service
        self.url = 'http://apis.data.go.kr/5390000/uiryeongcdnursetree'

    def list_info_inqire(self,path):
        data = []
        params = {
            'serviceKey': "IvrMdzqTlI6gqf0IavquGTueiCFJUOxaPugfaOcLf0WWZtVwh7cMTjeDBpG07Uz2CMVGDBXtBns6dlhQSvwjOA%3D%3D",
            'pageNo': '1',
            'numOfRows': '1000',
            'entld': '',
            'roadAddr' : ''
        }
        try:
            response = requests.get(self.url, params=params)
            response.raise_for_status()

            if response.status_code == 200:
                root = ET.fromstring(response.text)

                for item in root.findall('.//item'):
                    data.append({
                        "entId": item.findtext('entId'),
                        "familyNm": item.findtext('familyNm'),
                        "height": item.findtext('height'),
                        "mngAgency": item.findtext('mngAgency'),
                        "type": item.findtext('type'),
                        "ownerType": item.findtext('ownerType'),
                        "lat": item.findtext('lat'),
                        "lon": item.findtext('lon'),
                        "regDate": item.findtext('regDate'),
                        "roadAddr": item.findtext('roadAddr'),
                        "round": item.findtext('round'),
                        "scientificNm": item.findtext('scientificNm'),
                        "sido": item.findtext('sido'),
                        "sigungu": item.findtext('sigungu'),
                        "trees": item.findtext('trees'),
                        "address": item.findtext('address'),
                        "appointDate": item.findtext('appointDate'),
                        "appointNo": item.findtext('appointNo'),
                        "area": item.findtext('area'),
                        "kind": item.findtext('kind'),
                        "diameter": item.findtext('diameter'),
                        "dignityNm": item.findtext('dignityNm'),
                        "pointedNm": item.findtext('pointedNm'),
                        "age": item.findtext('age')
                    })
        except requests.exceptions.RequestException as e:
            print(e)
        df = pd.DataFrame(data)
        df.to_csv(path)