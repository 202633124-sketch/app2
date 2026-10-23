import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from kiwipiepy import Kiwi
from wordcloud import WordCloud
from collections import Counter

# --- 0. 한글 폰트 설정 ---
plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# --- 1. 감성 사전 정의 (각 10개 이상) ---
POSITIVE_DICT = {'좋다', '기쁘다', '훌륭하다', '성공', '만족', '행복', '최고', '감사', '편리', '희망', '유익하다', '완벽'}
NEGATIVE_DICT = {'나쁘다', '슬프다', '실망', '오류', '불편', '최악', '아쉽다', '어렵다', '실패', '부족하다', '문제', '지루하다'}


# --- 2. 전처리 함수 (사용자 사전 등록 + 명사/동사/형용사 추출) ---
@st.cache_resource
def get_kiwi():
    return Kiwi()


def extract_tokens_and_nouns(text, user_words_str):
    kiwi = get_kiwi()

    # [사용자 사전] 쉼표로 입력받은 단어를 Kiwi 사용자 사전에 등록
    if user_words_str.strip():
        user_words = [w.strip() for w in user_words_str.split(',') if w.strip()]
        for word in user_words:
            kiwi.add_user_word(word, tag='NNG')

    tokens = kiwi.tokenize(text)

    nouns = []  # 2글자 이상 명사 (시각화용)
    all_lemmas = []  # 명사 + 동사/형용사 기본형 (감성 분석용)

    for token in tokens:
        # 명사 (NNG: 일반명사, NNP: 고유명사)
        if token.tag in ('NNG', 'NNP'):
            all_lemmas.append(token.form)
            if len(token.form) >= 2:
                nouns.append(token.form)
        # 동사(VV), 형용사(VA) -> 기본형(~다)으로 변환하여 저장
        elif token.tag in ('VV', 'VA'):
            all_lemmas.append(token.form + "다")

    return nouns, all_lemmas


# --- 3. 빈도 분석 함수 ---
def get_word_counts(nouns, top_n):
    counts = Counter(nouns)
    df_counts = pd.DataFrame(counts.most_common(top_n), columns=['단어', '빈도수'])
    return df_counts


# --- 4. 막대그래프 생성 함수 ---
def plot_bar_chart(df_counts):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df_counts.iloc[::-1]
    ax.barh(df_sorted['단어'], df_sorted['빈도수'], color='skyblue')

    ax.set_title("단어 빈도수 상위 그래프", fontproperties=font_prop, fontsize=14)
    ax.set_xlabel("빈도수", fontproperties=font_prop)
    ax.set_ylabel("단어", fontproperties=font_prop)
    ax.tick_params(axis='y', labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    st.pyplot(fig)


# --- 5. 워드클라우드 생성 함수 ---
def plot_wordcloud(nouns):
    text_space = " ".join(nouns)
    wc = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        width=800,
        height=500
    ).generate(text_space)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# --- 6. 감성 분석 & 원그래프 생성 함수 ---
def analyze_sentiment(lemmas):
    pos_words = [w for w in lemmas if w in POSITIVE_DICT]
    neg_words = [w for w in lemmas if w in NEGATIVE_DICT]
    return pos_words, neg_words


def plot_sentiment_pie(pos_count, neg_count):
    fig, ax = plt.subplots(figsize=(6, 4))

    if pos_count == 0 and neg_count == 0:
        ax.text(0.5, 0.5, "감성 단어가 탐지되지 않았습니다.",
                ha='center', va='center', fontproperties=font_prop, fontsize=12)
        ax.axis("off")
    else:
        labels = ['긍정', '부정']
        counts = [pos_count, neg_count]
        colors = ['#66b3ff', '#ff9999']

        ax.pie(
            counts, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontproperties': font_prop}
        )
        ax.set_title("긍정 / 부정 감성 비율", fontproperties=font_prop, fontsize=14)

    st.pyplot(fig)


# --- 7. Streamlit 메인 화면 구성 ---
def main():
    st.title("단어 빈도 및 감성 시각화 프로그램")

    # [1] 파일 업로드
    uploaded_file = st.file_uploader("텍스트 파일(.txt)을 업로드하세요", type=["txt"])

    # [사용자 사전 입력창]
    user_words_input = st.text_input("사용자 사전 단어 입력 (쉼표로 구분)", placeholder="예: 인공지능, 워드클라우드")

    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")

        # [2] 파일 내용 보기
        st.subheader("1. 파일 내용 보기")
        st.text_area("원본 텍스트", raw_text, height=150)

        # [3] 전처리 결과
        nouns, all_lemmas = extract_tokens_and_nouns(raw_text, user_words_input)
        st.subheader("2. 전처리 결과 (2글자 이상 명사)")
        st.write(f"총 추출된 명사 수: **{len(nouns)}**개")
        st.write(nouns[:30])

        if nouns:
            # [4] 단어 빈도
            st.subheader("3. 단어 빈도")
            top_n = st.slider("상위 단어 개수 선택", min_value=5, max_value=50, value=10, step=5)
            df_counts = get_word_counts(nouns, top_n)
            st.dataframe(df_counts, use_container_width=True)

            # [5] 시각화 (탭 구성)
            st.subheader("4. 시각화")
            tab1, tab2 = st.tabs(["가로 막대그래프", "워드클라우드"])
            with tab1:
                plot_bar_chart(df_counts)
            with tab2:
                plot_wordcloud(nouns)

            # [6] 감성 분석
            st.subheader("5. 감성 분석")
            pos_words, neg_words = analyze_sentiment(all_lemmas)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("긍정 단어 수", f"{len(pos_words)}개")
                st.write("**추출된 긍정 단어:**", pos_words)
            with col2:
                st.metric("부정 단어 수", f"{len(neg_words)}개")
                st.write("**추출된 부정 단어:**", neg_words)

            plot_sentiment_pie(len(pos_words), len(neg_words))
        else:
            st.warning("추출된 2글자 이상의 명사가 없습니다.")


if __name__ == "__main__":
    main()
