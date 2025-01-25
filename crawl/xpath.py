# XPaths as constants
## first page
BOARD_AREA = '//*[@id="page-cont-main"]/div/div/div[4]'
ANNOUNCE_EXPAND_CONTAINER = '//*[@id="page-cont-main"]/div/div/div[4]/div'
ANNOUNCE_EXPAND_BTN = '//*[@id="page-cont-main"]/div/div/div[4]/div/button'
ANNOUNCE_LIST = '//*[@id="page-cont-main"]/div/div/div[4]/ul'
ANNOUNCE_ITEMS = ANNOUNCE_LIST + '//li'
ANNOUNCE_ITEM = '//*[@id="page-cont-main"]/div/div/div[4]/ul/li[2]/a'

## second page
CONTENT_AREA = '//*[@id="jutakInfoLeft"]'
ANNOUNCE_TITLE = '//*[@id="sigEmdNm"]/p/span'
JUTAK_LIST ='//*[@id="jutakList"]'
JUTAK_ITEMS = '//*[@id="jutakList"]/li'
JUTAK = '//*[@id="btn-modal-open"]'

INFO_AREA = '//*[@id="detailPop"]'
INFO_TITLE = '//*[@id="biznsViewTile"]'
INFO_TYPE_SPLY_BTN = '//*[@id="splyTyInfo"]'
INFO_TYPE_MANAGE_BTN = '//*[@id="manageCostDiv"]/button'

NAVIGATOR_CONTAINER = '//*[@id="pagingNav-jutakinfoLeft"]'
NAVIGATORS = NAVIGATOR_CONTAINER + '/a'
PAGE1 = '//*[@id="pagingNav-jutakinfoLeft"]/a[2]'
PAGE2 = '//*[@id="pagingNav-jutakinfoLeft"]/a[3]'
PAGE3 = '//*[@id="pagingNav-jutakinfoLeft"]/a[4]'

### Jutak Info
LIST_SUMMARY = '//*[@id="list-summary"]/button' # 단지 개요 버튼
CONT_SUMMARY = '//*[@id="cont-summary"]/div/ul/li[8]' # 입지 조건 확장 버튼
JTK_NAME = '//*[@id="biznsViewTile"]' # 주택 이름
ADDR = '//*[@id="addr"]' # 주소
HO_CNT = '//*[@id="hoCnt"]' # 임대 세대수
HS_TY_LST = '//*[@id="hsTyLst"]' # 공급 유형
HET_TY_NM = '//*[@id="hetTyNm"]' # 난방 방식
PSGE_TY_NM = '//*[@id="psgeTyNm"]' # 복도 유형
CNSTRT_CORP = '//*[@id="cnstrtCorp"]' # 시공사
CENT_TEL = '//*[@id="centTel"]' # 관리사무소 연락처
IF_CON_DESC = '//*[@id="lfCondDesc"]' # 입지조건

### 공급형 별 주소
SPLY_TY_INFO = '//*[@id="splyTyInfo"]' # 펼치기 버튼
SELECT_SPLY_TY = '//*[@id="selectSplyTy"]' # 공급 형별 -> option 순회
TAB_TABLE_VIEW = '//*[@id="tab-table-view"]/table/caption' # 공급형 정보 테이블
TAB_TABLE_DATA = '//*[@id="tab-table-view"]/table'
#### 공급형 정보들 -> table/tbody/tr 순회 td -> text

### 관리비
MANAGE_COST_BTN = '//*[@id="manageCostDiv"]/button' # 펼치기 버튼
MANAGE_COST_IS_NONE = '//*[@id="manageCostNoVal"]' # 관리비가 존재하지 않을때,
MANAGE_COST_TABLE = '//*[@id="manageCostDiv"]/div'
MANAGE_COST_TABLE_ROWS = '//*[@id="manageCostBg"]/div[2]/table/tbody'