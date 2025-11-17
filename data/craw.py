import requests
from bs4 import BeautifulSoup
import os
import time
import re




def fetch_post(nttId, menuNo="", view_path = ""):
    """게시글 HTML 요청"""
    params = {
        "nttId": str(nttId),
        "menuNo": menuNo
    }
    url = BASE + view_path
    resp = requests.get(url, params=params)
    return resp


def get_audio_links(soup):
    """HTML 내 오디오(mp3) 관련 링크 추출"""
    links = []

    # <video> 또는 <audio> 태그 처리
    for tag in soup.find_all(["video", "audio"]):
        src = tag.get("src")
        if src:
            if src.startswith("/"):
                src = BASE + src
            links.append(src)

    # fileDown.do 링크 (직접 다운로드 링크)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "fileDown.do" in href:
            if href.startswith("/"):
                href = BASE + href
            links.append(href)

    return list(set(links))  # 중복 제거


def download_mp3(url, folder="./mp3_files", filename=None):
    """
    개선된 다운로드 함수 — 상세 로그 + MP3 포맷 검사.
    반환: (saved_path, info_dict)
    info_dict에는 status_code, content_type, content_length, saved_bytes, is_mp3(boolean) 등이 들어있음.
    """
    os.makedirs(folder, exist_ok=True)

    # 요청 (짧은 타임아웃 권장)
    try:
        resp = requests.get(url, stream=True, timeout=30, allow_redirects=True)
    except Exception as e:
        print("요청 실패:", e)
        return None, {"error": str(e)}

    print("  HTTP 응답 코드:", resp.status_code)
    print("  최종 URL:", resp.url)
    ct = resp.headers.get("Content-Type", "")
    cl = resp.headers.get("Content-Length", None)
    cd = resp.headers.get("Content-Disposition", None)
    print("  Content-Type:", ct)
    print("  Content-Length:", cl)
    print("  Content-Disposition:", cd)

    # 파일명 결정
    server_filename = None
    if cd:
        m = re.search(r'filename\*?=.*?\'\'?([^;"]+)', cd)
        if m:
            server_filename = os.path.basename(m.group(1))
        else:
            m2 = re.search(r'filename="?([^";]+)"?', cd)
            if m2:
                server_filename = os.path.basename(m2.group(1))

    if filename:
        fname = filename if filename.endswith(".mp3") else filename + ".mp3"
    elif server_filename:
        fname = server_filename
    else:
        # URL 경로 마지막 조각
        from urllib.parse import urlparse, unquote, parse_qs
        parsed = urlparse(resp.url)
        qs = parse_qs(parsed.query)
        # atchFileId 등으로 이름 생성 시도
        if "atchFileId" in qs:
            fname = qs.get("atchFileId")[0] + ".mp3"
        else:
            base = os.path.basename(parsed.path)
            fname = (unquote(base) or "downloaded_file") + ".mp3"

    # 안전한 파일명
    fname = fname.replace("/", "_").replace("\\", "_").replace(":", "_")
    save_path = os.path.abspath(os.path.join(folder, fname))
    print("  저장 경로:", save_path)

    # 실제로 쓰기
    saved_bytes = 0
    try:
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    saved_bytes += len(chunk)
    except Exception as e:
        print("  파일 저장 중 오류:", e)
        return None, {"error": str(e)}

    print("  저장 완료 — 바이트:", saved_bytes)

    # 파일이 비었거나 아주 작으면 경고
    if saved_bytes == 0:
        print("  *** 경고: 저장된 파일 크기 0 bytes 입니다. HTML 에러가 내려왔을 가능성 있음.")

    # 앞부분 검사 (MP3 흔적: "ID3" 또는 frame sync 0xFF 0xFB / 0xFF 0xF3 / 0xFF 0xF2 등)
    is_mp3 = False
    try:
        with open(save_path, "rb") as f:
            header = f.read(64)
        if header.startswith(b"ID3"):
            is_mp3 = True
        elif len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            # 간단한 프레임 sync 체크 (0xFF Ex)
            is_mp3 = True
        else:
            is_mp3 = False
    except Exception as e:
        print("  파일 읽기 오류:", e)
        is_mp3 = False

    if is_mp3:
        print("  ✔ 파일이 MP3 포맷으로 보입니다.")
    else:
        print("  ✖ MP3 포맷이 아닌 것 같습니다.")
        # 만약 Content-Type이 text/html이면, 파일을 .html로 바꿔 저장하도록 권장
        if "text/html" in ct or save_path.endswith(".mp3") and saved_bytes < 2000:
            alt_path = save_path + ".html"
            os.replace(save_path, alt_path)
            print(f"  (주의) 저장된 내용이 HTML일 가능성 있어 파일명을 {alt_path} 로 변경했습니다. 열어보세요.")
            save_path = alt_path

    info = {
        "status_code": resp.status_code,
        "content_type": ct,
        "content_length_header": cl,
        "saved_bytes": saved_bytes,
        "is_mp3": is_mp3,
        "final_url": resp.url,
        "content_disposition": cd
    }
    return save_path, info


def crawl_range(start_id=36603, end_id=36744, menuNo="", view_path="", count = 0, download=True):
    """게시글 범위 순회하며 mp3 자동 다운로드"""
    results = []

    for ntt in range(start_id, end_id + 1):
        resp = fetch_post(ntt, menuNo, view_path)
        print(f"[DEBUG] nttId={ntt}, status={resp.status_code}")
        if resp.status_code != 200:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        mp3_links = get_audio_links(soup)
        print(f"[DEBUG] nttId={ntt}, mp3_links={mp3_links}")

        if not mp3_links:
            print(f"nttId={ntt}: mp3 없음")
            continue

        print(f"nttId={ntt}: mp3 {len(mp3_links)}개 발견")

        saved = []
        if download:
            for mp3url in mp3_links:
                # 게시글 제목 일부를 파일명으로 활용 가능
                filename = f"{count}.mp3"

                path = download_mp3(url=mp3url, filename=filename)
                saved.append(path)
                count += 1

        results.append({
            "nttId": ntt,
            "mp3_urls": mp3_links,
            "downloaded_paths": saved
        })

        time.sleep(1.0)  # 서버 부담 줄이기

    return results, count


BASE = "https://www.fss.or.kr"

if __name__ == "__main__":


    #대출사기형
    view_path_1 = "/fss/bbs/B0000206/view.do"
    menuNo_1 = "200690"
    start_id_1 = 36337
    end_id_1 = 36744

    #수사기관 사칭형
    view_path_2 = "/fss/bbs/B0000207/view.do"
    menuNo_2 = "200690"
    start_id_2 = 36334
    end_id_2 = 36745

    count = 0
    data1, count = crawl_range(start_id_1, end_id_1, menuNo_1, view_path_1, count, download=True)
    data2, _ = crawl_range(start_id_2, end_id_2, menuNo_2, view_path_2, count, download=True)

    all_data = data1 + data2  # 결과 합치기

    import json
    print(json.dumps(all_data, ensure_ascii=False, indent=2))