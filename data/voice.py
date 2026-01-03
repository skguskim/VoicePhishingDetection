import pandas as pd
def remove_short_yes_lines(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    if len(lines) == 1:
        return ""
        
    filtered = []
    for line in lines:
        # 길이 5 이하 + "네" 또는 "예" 포함 → 제거
        if len(line) <= 10 and ("네" in line or "예" in line):
            continue
        elif len(line) <= 8:
            continue
        filtered.append(line)

    return "\n".join([s for s in filtered if s])





def main():
    # 1. CSV 읽기
    data = pd.read_csv("phish_data.csv")

    # 2. transcript 컬럼에 변환 적용 → 새로운 컬럼 생성
    data['transcript_filtered'] = data['transcript'].apply(remove_short_yes_lines)

    # 3. 새 DataFrame 생성 (원본 + 필터링본)
    new_df = data['transcript_filtered']

    # 4. CSV로 저장
    new_df.to_csv("phish_data_filtered7-1.csv", index=True)

    print("Saved to phish_data_filtered7-1.csv")


if __name__ == "__main__":
    main()