import csv
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import itertools
import collections
from gensim.corpora import Dictionary

##前処理 preprocessing_nlk ~ WordNetLemmatizer_lemmatize_pos
def preprocessing_nlk(text):
    #分かち書き http://haya14busa.com/python-nltk-natural-language-processing/
    morph = nltk.word_tokenize(text)
    #品詞分析
    pos = nltk.pos_tag(morph)
    #小文字化
    morph_low = [(i[0].lower(),i[1]) for i in pos]
    #分かち書き & 文の分割 & 品詞選別 & ストップワード除去
    pos_pre = pos_preprocessing(morph_low)
    # print("################################################################################")
    # #SnowballStemmer -> dies -> die(死ぬ) 複数形や色々を原型に直す
    # stemmer_nltk(pos_pre)

    return WordNetLemmatizer_lemmatize_pos(pos_pre)

#NN, NNS(名詞,複数形名詞)-> n(レンマ化.lemmatize(s, pos="n"))
#VB, VBD, VBG, VBN, VBP, VBZ  -> v
#(動詞(原型),動詞(過去形),動詞(動名詞,現在分詞),動詞(過去分詞),動詞(三人称単数外の現在系),動詞(三人称単数現在系))
#JJ, JJR, JJS(形容詞,比較級,最上級) -> a
#RB, RBR, RBS (副詞,比較級,最上級) -> r
def pos_preprocessing(morph_low):
    #品詞辞書 https://qiita.com/m__k/items/ffd3b7774f2fde1083fa
    dic_NVJR = {"NN":"n", "NNS":"n",#名詞
                "VB":"v", "VBD":"v", "VBG":"v", "VBN":"v", "VBP":"v", "VBZ":"v",#動詞
                "JJ":"a", "JJR":"a", "JJS":"a",#形容詞
                "RB":"r", "RBR":"r", "RBS":"r"}#副詞
    #ストップワード https://gist.github.com/sebleier/554280#file-nltk-s-list-of-english-stopwords
    sw = stopwords.words("english")
    pos_pre = []
    pos_pre.append([])
    text_n = 0#文数 ピリオドでインクリ

    for p in morph_low:
        if p[0] not in sw:
            if p[1] in dic_NVJR:
                pos_pre[text_n].append((p[0], dic_NVJR[p[1]]))
        #ピリオドで次の文
        if p[0] == ".":
            text_n += 1
            pos_pre.append([])
    #余分な最後の要素削除
    del pos_pre[-1]

    return pos_pre


#SnowballStemmer -> dies -> die(死ぬ) 複数形や色々を原型に直す
#使わない
def stemmer_nltk(pos_pre):
    stemmer = SnowballStemmer('english')
    stem_pre = []
    for pos in pos_pre:
        stem_pre.append([stemmer.stem(i[0]) for i in pos])

    print(stem_pre)
    return 0
##went -> go　見出し語化, レンマ化
#文脈を考慮した語幹化  こっちの方が良さそう
def WordNetLemmatizer_lemmatize_pos(pos_pre):
    wordNetL_lem = []
    for pos in pos_pre:
        wordNetL_lem.append([WordNetLemmatizer().lemmatize(i[0], pos=i[1]) for i in pos])

    return wordNetL_lem

#areaにあるホテルのIDをとってくる
def reviews_va_csv(area):
    area_id = []

    filename = "data/listings.csv"
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[5] == area:#####################ここを==, inで変更
                if row[0] not in area_id:
                    area_id.append(row[0])
    #指定エリア内の全施設数,全施設のID
    return area_id
#ホテルIDのあったレビューを取得 -> listing_id??
def reviews_id_area(area_id):
    area_reviwes = []
    filename = "data/reviews_va.csv"

    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        # n = 0
        for row in reader:
            if row[0] in area_id:
                area_reviwes.append(row[5])
            # else:
            #     n += 1

    #使用可能な全レビュー数:199295
    print(len(area_reviwes))
    # #ID無記名3989
    # print(n)
    return area_reviwes
def bow_file(text):
    ###########################################
    #一次元配列に名詞[名詞1,名詞2,名詞1]
    ts = []
    ts = itertools.chain.from_iterable(text)
    c = collections.Counter(ts)
    #####################BOWファイル書き出し########################
    path_w = "data/" + are_vac[22] + "/bow.txt"
    with open(path_w,mode='x') as f:#ファイルがない場合
    #with open(path_w,mode='w') as f: #ファイルがある場合
        for i in c.most_common():
            f.write(str(i[0]) + ":" + str(i[1]) + " ")

    ##"Kerrisdale"は出現回数１が54%,,2が12%,3が5%#1を消したら2が28%ぐらい
    #"Downtown"は出現回数1が45%,2が13%,3が6%##消したら26%

    return 0

def bow_corpus_dct(text):
    #辞書作成
    #text=[['a','a','b','c'],['a','a','c']] -> [a,b,c]
    dct = Dictionary(text)
    #no_below: 出現文書数が閾値以上になるような単語のみを保持します
    #no_above: 出現文書数/全文書数が閾値以下になるような単語のみを保持します
    dct.filter_extremes(no_below=3, no_above=0.8)#no_above=0.8)

    #辞書をID化
    #{'maso': 0, 'mele': 1, 'máma': 2, 'ema': 3, 'má': 4, 'is': 5, 'sparta': 6, 'this': 7, 'joking': 8, 'just': 9}
    #dct.token2id

    #2次元配列[[(0ID, 1全文書内出現回数), (1, 1), (2, 1), (3, 1)],[(4, 1), (5, 1), (6, 1), (7, 1)]]
    corpus = [dct.doc2bow(t) for t in text]

    print(dct[0])
    print(dct[16])
    print(text[2])
    print(dct.doc2bow(text[2]))
    print(corpus[2])

    return corpus, dct

def main():
    are_vac=["Downtown", "Riley Park", "Downtown Eastside", "Kensington-Cedar Cottage", "Hastings-Sunrise",
    "Renfrew-Collingwood",  "Mount Pleasant", "Grandview-Woodland",  "Kitsilano", "West End",
    "Fairview", "Marpole", "Arbutus Ridge",  "Sunset", "Dunbar Southlands",  "Killarney",
    "South Cambie",  "Shaughnessy", "Victoria-Fraserview",  "Strathcona", "West Point Grey", "Oakridge",
    "Kerrisdale"]
    #指定エリアにあるホテルIDの取得
    area_id = reviews_va_csv(are_vac[22])

    reviews = []
    reviews = reviews_id_area(area_id)#####レビュー所得
    ###########################################
    ##前処理
    text = []
    for t in reviews:#######全レビューを前処理[1レビューに出現される名詞],[..]..]
        #####text=[['a','a','b','c'],['a','a','c']]
        text.append([n for i in preprocessing_nlk(t) for n in i])

    print(text[0])
    #bowコーパス作成
    corpus, dct = bow_corpus_dct(text)

    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
    ##bow書き出し，分析用
    #bow_file(text)

if __name__ == "__main__":
    #地域を指定して，出現名詞の数を集計
    main()
